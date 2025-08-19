# dataset/audio_rir_dataset.py  — 改訂版
import random, math, torchaudio, torch, pandas as pd
import yaml
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import numpy as np
import soundfile   as sf
from functools import lru_cache
try:
    if torchaudio.get_audio_backend() != "sox_io":
        torchaudio.set_audio_backend("sox_io")
except Exception:
    pass
# ToDo;SALSAの論文の実装に合わせる。0のときの処理を
 #ToDo: A-format to B-format
 # #ToDo: captionも空間拡張する
MAX_DURATION_SEC = 10.0
FOA_SR = 48_000      # FOA / omni target SR
IV_SR  = 16_000      # intensity-vector SR

@lru_cache(maxsize=16)
def _cached_hann(n_fft: int, device_str: str):
    return torch.hann_window(n_fft, device=torch.device(device_str))

@lru_cache(maxsize=8192)   # メモリに余裕があれば大きめでも可
def _load_dry_cached(path: str):
    wav, sr = torchaudio.load(path)
    if sr != FOA_SR:
        wav = torchaudio.functional.resample(wav, sr, FOA_SR)
    if wav.shape[-1] < int(FOA_SR*MAX_DURATION_SEC):
        rpt = math.ceil(FOA_SR*MAX_DURATION_SEC / wav.shape[1])
        wav = wav.repeat(1, rpt)[:, : int(FOA_SR*MAX_DURATION_SEC)]
    else:
        wav = wav[:, : int(FOA_SR*MAX_DURATION_SEC)]
    return wav

#random.seed(5)  # 再現性のため

# -------------- mapping helpers (from SpatialCaps.py) -----------------
def map_direction(az):
    if -35 <= az <= 35: return 'front'
    if 55 <= az <= 125: return 'right'
    if -125 <= az <= -55: return 'left'
    return 'back' if az >= 145 or az <= -145 else ''

def map_elevation(el): return 'up' if el > 40 else 'down' if el < -40 else ''
def map_distance(d):   return 'near' if d < 1 else 'far' if d > 2 else ''
def map_size(a):       return 'small' if a < 50 else 'large' if a > 100 else 'mid-sized'
def map_reverb(t30):   return 'acoustically dampened' if t30 < 200 else \
                        'highly reverberant' if t30 > 1000 else ''
def rewrite_caption(orig: str, meta: dict) -> str:
    dist_txt   = map_distance(meta["source_distance_m"])
    dir_txt    = map_direction(meta["azimuth_deg"])
    ele_txt    = map_elevation(meta["elevation_deg"])
    size_txt   = map_size(meta["area_m2"])
    reverb_txt = map_reverb(meta["fullband_T30_ms"])

    loc = ' '.join(filter(None, [dist_txt, ele_txt, dir_txt])).strip()
    room = f"a {size_txt} room"
    if reverb_txt: room += f" that is {reverb_txt}"
    if loc: room = f"{loc} part of {room}"

    return f"{orig} in {room}.".replace("  ", " ")

# ランダム性があっても良い. 文章の順番を入れ替えたり、
# ------- foa_to_iv (同じものをインポートしても可) --------------------------
def foa_to_iv(
    foa_wave: torch.Tensor,
    n_fft: int = 400,
    hop: int = 100,
    eps: float = 1e-6,
):
    B, C, T = foa_wave.shape
    assert C == 4, "FOA wav must be (B,4,T)"
    win = _cached_hann(n_fft, str(foa_wave.device))
    spec = (
        torch.stft(
            foa_wave.view(-1, T),
            n_fft=n_fft,
            hop_length=hop,
            window=win,
            center=True,
            return_complex=True,
        )
        .view(B, 4, n_fft // 2 + 1, -1)
    )
    W, Y, Z, X = spec[:, 0], spec[:, 1], spec[:, 2], spec[:, 3]
    conjW = W.conj()
    I_act = torch.stack([(conjW * Y).real,
                         (conjW * Z).real,
                         (conjW * X).real], dim=1)
    I_rea = torch.stack([(conjW * Y).imag,
                         (conjW * Z).imag,
                         (conjW * X).imag], dim=1)
    norm = torch.linalg.norm(I_act, dim=1, keepdim=True)

    # ToDo;SALSAの論文の実装に合わせる。0のときの処理を
    I_act = torch.where(norm > eps, I_act / norm, I_act)
    I_rea = torch.where(norm > eps, I_rea / norm, I_rea)
    return I_act.float(), I_rea.float()

class AudioRIRDataset(Dataset):
    def __init__(self,
                 csv_audio: str,
                 base_dir: str, # dry音源のベースディレクトリ
                 csv_rir: str,
                 n_views: int = 1,
                 split: str = "train",
                 n_fft: int = 400,
                 share_rir: bool = True,
                 batch_size : int | None = None,
                 stats_path: str = "/home/takamichi-lab-pc09/DELSA/RIR_dataset/stats.pt",
                 hop: int =100,
                 bad_log_path: str | None = "bad_mp3_ids.txt",
                 defer_convolution: bool = True):
        super().__init__()
        # ── 設定ロード ──

        self.share_rir = share_rir
        self.batch_size = batch_size
        self._emmited = 0 # なんサンプルか返したか
        self._batch_rir = None # 現バッチ用RIRをキャッシュする
        self.base_dir = Path(base_dir)
        self.split = split
        self.n_views = n_views
        self.n_fft = n_fft
        self.hop = hop
        self.foa_len = int(FOA_SR*MAX_DURATION_SEC)
        self.defer_convolution = defer_convolution
        # audio_csv読み込み
        df = pd.read_csv(csv_audio)
        df = df[df["audiocap_id"].apply(
            lambda i :(self.base_dir/self.split/f"{i}.mp3").exists()
        )].reset_index(drop=True)
        self.audio_df = df
        self.source_map = {i: i for i in range(len(df))}

        # rir_csv読み込み
        rir_df = pd.read_csv(csv_rir)
        self.rir_paths = rir_df["rir_path"].tolist()
        self.space_map = {p: i for i,p in enumerate(self.rir_paths)}

        # RIR ごとのメタ dict
        self.rir_meta  = {row["rir_path"]: row.to_dict() for _, row in rir_df.iterrows()}
        self._bad_ids = set()
        self._bad_log_path = Path(bad_log_path) if bad_log_path else None    
        #物理量を正規化するためののコード
        stats = torch.load(stats_path)
        self.area_mean = stats["area_m2"]["mean"]
        self.area_std  = stats["area_m2"]["std"]
        self.dist_mean = stats["distance"]["mean"]
        self.dist_std  = stats["distance"]["std"]
        self.t30_mean  = stats["fullband_T30_ms"]["mean"]
        self.t30_std   = stats["fullband_T30_ms"]["std"]


    def __len__(self):
        return len(self.audio_df)
    def _mark_bad(self, audiocap_id: int, err: Exception):
        if audiocap_id in self._bad_ids:
            return
        self._bad_ids.add(audiocap_id)
        msg = f"[WARN] skip {audiocap_id}: {err}"
        print(msg)
        if self._bad_log_path:
            with self._bad_log_path.open("a") as f:
                f.write(f"{audiocap_id}\t{type(err).__name__}\t{err}\n")
 
    def _load_dry(self, audiocap_id: int) -> torch.Tensor:
        path = self.base_dir / self.split / f"{audiocap_id}.mp3"
        wav, sr = torchaudio.load(path)
        if sr != FOA_SR:
            wav = torchaudio.functional.resample(wav, sr, FOA_SR)
        if wav.shape[-1] < self.foa_len:
            rpt = math.ceil(self.foa_len / wav.shape[1])
            wav = wav.repeat(1, rpt)[:, : self.foa_len]
        else: wav = wav[:, : self.foa_len]
        return wav
    
    def _apply_rir(self, dry: torch.Tensor, rir_path: str) -> torch.Tensor:
        rir, sr = torchaudio.load(rir_path)            # [4,Tr] (FOA)
        wet = torchaudio.functional.fftconvolve(dry, rir)  # [4,T+Tr-1]
        wet = wet[..., :dry.shape[-1]]               # クロップ → 10 s
    # ---------- RIR 畳み込み（A-format 4ch） ----------
  
        m0, m1, m2, m3 = wet[0], wet[1], wet[2], wet[3]

        # ---------- A → B (First-order Ambisonics, FOA) ----------
        W =  (m0 +  m1 +  m2 +  m3)/2
        X =  (m0 +  m1 -  m2 -  m3)/2
        Y =  (m0 -  m1 +  m2 -  m3)/2
        Z =  (m0 -  m1 -  m2 +  m3)/2
        foa = torch.stack([W, Y, Z, X])
        #print(sr)
        #sf.write('foa.wav', foa.T, sr)
        return foa


    def __getitem__(self, idx: int):
        row = self.audio_df.iloc[idx]
        audiocap_id = row["audiocap_id"]
        caption = row["caption"]
        try:
            dry = self._load_dry(audiocap_id)
        except Exception as e:
            self._mark_bad(audiocap_id, e)
            return None  # ← ここがポイント：丸ごとスキップ

        audio_features_list = []
        src_ids, spa_ids, texts, rir_meta = [], [], [], []
        if self.share_rir:
            # バッチ頭だったらRIRを選び直す
            if self._batch_rir is None:
                self._batch_rir = random.sample(self.rir_paths, k = self.n_views)
            rir_paths = self._batch_rir

            # サンプルを返すたびに、カウンタを進める。バッチ終端でキャッシュを破棄
            self._emmited += 1
            if self.batch_size and self._emmited % self.batch_size == 0:
                self._batch_rir = None
            
            # エポック最終サンプルのとき
            if self._emmited == len(self):
                self._batch_rir = None # 次エポック用にリセット
                self._emmited = 0
        else:
            rir_paths = random.sample(self.rir_paths, k=self.n_views)

        for rir_path in rir_paths:
            if self.defer_convolution:
                audio_features_list.append({"dry": dry, "rir_path": rir_path}) #[4, T10]
                meta = self.rir_meta[rir_path].copy()      # シャローコピーで安全に複製 :contentReference[oaicite:5]{index=5}
                meta["area_m2_norm"]  = (meta["area_m2"]           - self.area_mean) / self.area_std
                meta["distance_norm"] = (meta["source_distance_m"] - self.dist_mean) / self.dist_std
                meta["t30_norm"]      = (meta["fullband_T30_ms"]   - self.t30_mean)  / self.t30_std
                azimuth = meta["azimuth_deg"]
                elevation = meta["elevation_deg"]
                direction_vec = torch.tensor(
                    [np.deg2rad(azimuth), np.deg2rad(elevation)],
                    dtype=torch.float32
                )
                meta["direction_vec"] = direction_vec
                caption_spatial = rewrite_caption(caption, meta)

            else: 
                wet = self._apply_rir(dry, rir_path) #[4, T10]
                #ToDo済: A-format to B-format　　　Spatial_AudioCaps/scripts/SpatialAudio.pyを参考に
                #ToDo済: captionも空間拡張する(ルールべースの書き換え)
                meta = self.rir_meta[rir_path].copy()      # シャローコピーで安全に複製 :contentReference[oaicite:5]{index=5}
                meta["area_m2_norm"]  = (meta["area_m2"]           - self.area_mean) / self.area_std
                meta["distance_norm"] = (meta["source_distance_m"] - self.dist_mean) / self.dist_std
                meta["t30_norm"]      = (meta["fullband_T30_ms"]   - self.t30_mean)  / self.t30_std
                azimuth = meta["azimuth_deg"]
                elevation = meta["elevation_deg"]
                direction_vec = torch.tensor(
                    [np.deg2rad(azimuth), np.deg2rad(elevation)],
                    dtype=torch.float32
                )
                meta["direction_vec"] = direction_vec
                caption_spatial = rewrite_caption(caption, meta)
                omni_48k = wet[0]  # [T10]
                
                # 16kにリサンプリング
                wet_16k = torchaudio.functional.resample(wet, orig_freq=FOA_SR, new_freq=IV_SR)
                i_act, i_rea = foa_to_iv(wet_16k.unsqueeze(0), n_fft=self.n_fft, hop=self.hop)
                i_act, i_rea = i_act.squeeze(0), i_rea.squeeze(0)

                audio_features_list.append({"i_act": i_act, "i_rea": i_rea, "omni_48k": omni_48k})
            src_ids.append(self.source_map[idx])
            spa_ids.append(self.space_map[rir_path])
            texts.append(caption_spatial)
            rir_meta.append(meta)
        return{
            "audio": audio_features_list,  # 各ビューの音声特徴
            "texts": texts,                # キャプション
            "source_id": torch.tensor(src_ids, dtype=torch.long), 
            "space_id": torch.tensor(spa_ids, dtype=torch.long),
            "rir_meta": rir_meta,          # RIRメタデータ
        }   
    
# ---------------- collate_fn (4 ch → 特徴辞書) -----------------------------
def collate_fn(batch):
    # 失敗サンプル（None）を取り除く
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    # 以降は従来どおり flatten
    # 既存の flatten 処理
    audio_list, text_list, src_list, spa_list, rir_meta_list = [], [], [], [], []
    for sample in batch:
        audio_list += sample["audio"]
        text_list  += sample["texts"]
        src_list.append(sample["source_id"])
        spa_list.append(sample["space_id"])
        rir_meta_list += sample["rir_meta"]
    # ------------------------

    # RIR メタデータを key ごとにまとめる
    meta_keys = rir_meta_list[0].keys()
    rir_meta_dict = {}
    for key in meta_keys:
        
        vals = [m[key] for m in rir_meta_list]
        first = vals[0]

        # 1) Tensor の場合 → stack
        if isinstance(first, torch.Tensor):
            rir_meta_dict[key] = torch.stack(vals, dim=0)

        # 2) 数値（int/float）の場合 → float tensor
        elif isinstance(first, (int, float)):
            rir_meta_dict[key] = torch.tensor(vals, dtype=torch.float32)

        # 3) 文字列などその他 → そのままリストで保持
        else:
            rir_meta_dict[key] = vals

    return {
        "audio": audio_list,
        "texts": text_list,
        "source_id": torch.vstack(src_list),
        "space_id":  torch.vstack(spa_list),
        "rir_meta":  rir_meta_dict,  # dict of tensors or lists
    }
