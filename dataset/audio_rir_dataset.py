# dataset/audio_rir_dataset.py  — 改訂版
import random, math, torchaudio, torch, pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
# ToDo;SALSAの論文の実装に合わせる。0のときの処理を
 #ToDo: A-format to B-format
 # #ToDo: captionも空間拡張する
MAX_DURATION_SEC = 10.0
FOA_SR = 48_000      # FOA / omni target SR
IV_SR  = 16_000      # intensity-vector SR



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


# ------- foa_to_iv (同じものをインポートしても可) --------------------------
def foa_to_iv(
    foa_wave: torch.Tensor,
    n_fft: int = 400,
    hop: int = 100,
    eps: float = 1e-6,
):
    B, C, T = foa_wave.shape
    assert C == 4, "FOA wav must be (B,4,T)"
    win = torch.hann_window(n_fft, device=foa_wave.device)
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
                 hop: int =100):
        super().__init__()

        self.base_dir = Path(base_dir)
        self.split = split
        self.n_views = n_views
        self.n_fft = n_fft
        self.hop = hop
        self.foa_len = int(FOA_SR*MAX_DURATION_SEC)
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

    def __len__(self):
        return len(self.audio_df)
    
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
        rir, _ = torchaudio.load(rir_path)            # [4,Tr] (FOA)
        wet = torchaudio.functional.fftconvolve(dry, rir)  # [4,T+Tr-1]
        wet = wet[..., :dry.shape[-1]]               # クロップ → 10 s
        return wet


    def __getitem__(self, idx: int):
        row = self.audio_df.iloc[idx]
        audiocap_id = row["audiocap_id"]
        caption = row["caption"]

        dry = self._load_dry(audiocap_id)

        audio_features_list = []
        src_ids, spa_ids, texts = [], [], []

        for _ in range(self.n_views):
            rir_path = random.choice(self.rir_paths)
            wet = self._apply_rir(dry, rir_path) #[4, T10]   
            #ToDo: A-format to B-format
             #ToDo: captionも空間拡張する
            meta = self.rir_meta[rir_path]
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

        return{
            "audio": audio_features_list,  # 各ビューの音声特徴
            "texts": texts,                # キャプション
            "source_id": torch.tensor(src_ids, dtype=torch.long), 
            "space_id": torch.tensor(spa_ids, dtype=torch.long),
        }
    
# ---------------- collate_fn (4 ch → 特徴辞書) -----------------------------
def collate_fn(batch):
    # “audio” は list[n_views] × B をフラット化して返す
    audio_list, text_list, src_list, spa_list = [], [], [], []
    for sample in batch:
        audio_list += sample["audio"]
        text_list  += sample["texts"]
        src_list.append(sample["source_id"])
        spa_list.append(sample["space_id"])

    # 辞書の中身 (i_act etc.) はテンソルなのでそのままリストで扱い

    print(text_list)
    return {
        "audio": audio_list,              # len=B*n_views
        "texts": text_list,               # len=B*n_views
        "source_id": torch.vstack(src_list),  # shape [B,n_views]
        "space_id" : torch.vstack(spa_list),
    }
