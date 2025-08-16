import random, math, torchaudio, torch, pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
import time

try:
    if torchaudio.get_audio_backend() != "sox_io":
        torchaudio.set_audio_backend("sox_io")
except Exception:
    pass

MAX_DURATION_SEC = 10.0
FOA_SR = 48_000
IV_SR  = 16_000

# ======== FOA → Intensity Vector (変更なし) ========
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
    I_act = torch.where(norm > eps, I_act / norm, I_act)
    I_rea = torch.where(norm > eps, I_rea / norm, I_rea)
    return I_act.float(), I_rea.float()


class AudioRIRDatasetProfiled(Dataset):
    def __init__(self,
                 csv_audio: str,
                 base_dir: str,
                 csv_rir: str,
                 n_views: int = 1,
                 split: str = "train",
                 n_fft: int = 400,
                 share_rir: bool = False,
                 batch_size: int = 8,
                 stats_path: str = "RIR_dataset/stats.pt",
                 hop: int =100,
                 profile_io: bool=False):
        super().__init__()
        self.base_dir = Path(base_dir)
        self.split = split
        self.n_views = n_views
        self.n_fft = n_fft
        self.hop = hop
        self.profile_io = profile_io
        self._clock = time.perf_counter

        # ==================== 🚀 変更点 (1) ====================
        # Deviceを決定し、transformsを一度だけ初期化してGPUに配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # FFT畳み込み用のインスタンスを生成
        self.fftconv = torchaudio.transforms.FFTConvolve(mode="full").to(self.device)
        
        # 再サンプリング用のインスタンスを生成
        self.resampler_48_to_16 = torchaudio.transforms.Resample(
            orig_freq=FOA_SR, new_freq=IV_SR
        ).to(self.device)
        # dry音源を48kHzに変換するためのリサンプラーも用意
        # 注: 入力サンプリングレートが多様な場合は、辞書で管理するなどの工夫も有効
        self.resamplers_to_48k = {}
        # =======================================================

        df = pd.read_csv(csv_audio)
        df = df[df["audiocap_id"].apply(
            lambda i :(self.base_dir/self.split/f"{i}.mp3").exists()
        )].reset_index(drop=True)
        self.audio_df = df
        self.source_map = {i: i for i in range(len(df))}

        rir_df = pd.read_csv(csv_rir)
        self.rir_paths = rir_df["rir_path"].tolist()
        self.space_map = {p: i for i,p in enumerate(self.rir_paths)}
        self.rir_meta  = {row["rir_path"]: row.to_dict() for _, row in rir_df.iterrows()}

        # (統計情報の読み込み部分は変更なし)
        # ...

    def __len__(self): return len(self.audio_df)

    def _load_dry(self, audiocap_id: int, _prof=None):
        path = self.base_dir / self.split / f"{audiocap_id}.mp3"
        t0_load = self._clock()
        wav, sr = torchaudio.load(path)
        if _prof is not None:
            _prof["load_mp3"] = self._clock() - t0_load

        if sr != FOA_SR:
            t0_resample = self._clock()
            # ==================== 🚀 変更点 (2) ====================
            # Resampleインスタンスをキャッシュして使い回す
            if sr not in self.resamplers_to_48k:
                self.resamplers_to_48k[sr] = torchaudio.transforms.Resample(sr, FOA_SR).to(self.device)
            resampler = self.resamplers_to_48k[sr]
            wav = resampler(wav.to(self.device)).cpu() # GPUでリサンプルし、結果をCPUに戻す
            # =======================================================
            if _prof is not None:
                _prof["resample_48k"] = self._clock() - t0_resample

        # (音声長の調整部分は変更なし)
        if wav.shape[-1] < int(FOA_SR*MAX_DURATION_SEC):
            rpt = math.ceil(int(FOA_SR*MAX_DURATION_SEC) / wav.shape[1])
            wav = wav.repeat(1, rpt)[:, : int(FOA_SR*MAX_DURATION_SEC)]
        else: wav = wav[:, : int(FOA_SR*MAX_DURATION_SEC)]
        return wav

    def __getitem__(self, idx: int):
        row = self.audio_df.iloc[idx]
        audiocap_id = row["audiocap_id"]; caption=row["caption"]
        _prof_sample = {} if self.profile_io else None
        
        # 1. Dry音声の読み込み
        dry_cpu = self._load_dry(audiocap_id, _prof_sample)
        
        # 2. RIRの選択と読み込み
        t0_rir_load = self._clock()
        rir_paths = random.sample(self.rir_paths, k=self.n_views)
        # ==================== 🚀 変更点 (3) ====================
        # 複数のRIRをリストとして読み込み、torch.stackでバッチ化する
        rirs_list = [torchaudio.load(p)[0] for p in rir_paths]
        rirs = torch.stack(rirs_list, dim=0).to(self.device) # (V, 4, T_rir)
        if _prof_sample is not None:
            _prof_sample["load_rir_batch"] = self._clock() - t0_rir_load

        # 3. 畳み込み (バッチ処理)
        t0_conv = self._clock()
        # Dry音声をGPUに送り、RIRの数 (V) だけ複製
        dry = dry_cpu.to(self.device)
        dry_v = dry.unsqueeze(0).repeat(self.n_views, 1, 1) # (V, 1, T)

        # バッチ畳み込みを一括実行
        wet_v = self.fftconv(dry_v, rirs) # (V, 4, T_conv)
        wet_v = wet_v[..., : dry.shape[-1]] # 長さを揃える
        if _prof_sample is not None:
            _prof_sample["rir_conv_batch"] = self._clock() - t0_conv

        # (B-format変換部分は変更なし、ただしバッチのまま処理)
        m0, m1, m2, m3 = wet_v[:,0], wet_v[:,1], wet_v[:,2], wet_v[:,3]
        W = (m0+m1+m2+m3)/2; X = (m0+m1-m2-m3)/2
        Y = (m0-m1+m2-m3)/2; Z = (m0-m1-m2+m3)/2
        wet_v_foa = torch.stack([W, Y, Z, X], dim=1) # (V, 4, T)

        # 4. 再サンプリング (バッチ処理)
        t0_resample16 = self._clock()
        wet_16k = self.resampler_48_to_16(wet_v_foa) # (V, 4, T_16k)
        if _prof_sample is not None:
            _prof_sample["resample_16k_batch"] = self._clock() - t0_resample16
        
        # 5. IV特徴量計算 (バッチ処理)
        t0_foa2iv = self._clock()
        i_act_v, i_rea_v = foa_to_iv(wet_16k, n_fft=self.n_fft, hop=self.hop) # (V, 3, F, T')
        if _prof_sample is not None:
            _prof_sample["foa_to_iv_batch"] = self._clock() - t0_foa2iv

        # 6. 結果の整形 (ループ処理)
        # 計算が終わった後、Pythonリストに格納するのは高速
        wet_v_foa_cpu = wet_v_foa.cpu() # 必要なデータのみCPUに戻す
        i_act_v_cpu = i_act_v.cpu()
        i_rea_v_cpu = i_rea_v.cpu()
        
        audio_features_list=[]; src_ids=[]; spa_ids=[]; texts=[]; rir_meta_list=[]
        for i, rir_path in enumerate(rir_paths):
            audio_features_list.append({
                "i_act": i_act_v_cpu[i],
                "i_rea": i_rea_v_cpu[i],
                "omni_48k": wet_v_foa_cpu[i, 0] # Wチャンネル (Omni)
            })
            src_ids.append(self.source_map[idx])
            spa_ids.append(self.space_map[rir_path])
            texts.append(caption)
            rir_meta_list.append(self.rir_meta[rir_path])
        # ==========================================================

        return {
            "audio": audio_features_list, "texts": texts,
            "source_id": torch.tensor(src_ids), "space_id": torch.tensor(spa_ids),
            "rir_meta": rir_meta_list,
            **({"_prof": _prof_sample} if _prof_sample is not None else {}),
        }

# (collate_fn は変更なし)

# ==== collate_fn ====
def collate_fn_profiled(batch):
    batch=[b for b in batch if b is not None]
    if len(batch)==0: return None
    audio_list=[]; text_list=[]; src_list=[]; spa_list=[]; rir_meta_list=[]
    for sample in batch:
        audio_list+=sample["audio"]; text_list+=sample["texts"]
        src_list.append(sample["source_id"]); spa_list.append(sample["space_id"])
        rir_meta_list+=sample["rir_meta"]
    out={
        "audio": audio_list,
        "texts": text_list,
        "source_id": torch.vstack(src_list),
        "space_id": torch.vstack(spa_list),
        "rir_meta": rir_meta_list,
    }
    prof_list=[s["_prof"] for s in batch if "_prof" in s]
    if len(prof_list)>0: out["_prof_list"]=prof_list
    return out
