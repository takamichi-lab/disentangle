import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

# W, Y, Z, XはFOAの形式によって変える必要あり
#############################################
# Constants
#############################################
# 10‑second fixed duration (cannot be changed)
MAX_DURATION_SEC: float = 10.0
# Fixed FOA sample‑rate
FOA_SR: int = 48_000  # Omni is kept at this rate
# Sample‑rate used for I_act / I_rea computation
IV_SR: int = 16_000


# -----------------------------------------------------------------------------
# Utility: FOA → (I_act, I_rea)
# -----------------------------------------------------------------------------

def foa_to_iv(
    foa_wave: torch.Tensor,
    sr: int = IV_SR,
    n_fft: int = 400,
    hop: int = 100,
    eps: float = 1e-6,
):
    """Convert FOA waveform to *active* / *reactive* intensity vectors."""
    B, C, T = foa_wave.shape
    assert C == 4, "FOA waveform must have 4 channels (W, Y, Z, X)"

    win = torch.hann_window(n_fft, device=foa_wave.device)

    # STFT: (B, 4, F, N)
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
        .contiguous()
    )

    W, Y, Z, X = spec[:, 0], spec[:,1], spec[:, 2], spec[:, 3]
    conjW = W.conj()

    # Active / Reactive intensity
    I_act = torch.stack([(conjW * Y).real, (conjW * Z).real, (conjW * X).real], dim=1)
    I_rea = torch.stack([(conjW * Y).imag, (conjW * Z).imag, (conjW * X).imag], dim=1)

    # Unit‑norm (skip zero‑vectors)
    norm = torch.linalg.norm(I_act, dim=1, keepdim=True)  # (B,1,F,N)
    I_act = torch.where(norm > eps, I_act / norm, I_act)
    I_rea = torch.where(norm > eps, I_rea / norm, I_rea)

    return I_act.float(), I_rea.float()


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class ELSADataset(Dataset):
    """
    Dataset for ELSA model.
    Returns a dictionary containing audio features, raw text caption,
    and spatial regression targets.
    """

# file: dataset/foa_dataset_caption.py

    def __init__(
        self,
        foa_folder: str,
        foa_metadata_csv: str,
        split: str,
        n_fft: int = 400,
        hop: int = 100,
        stats_path: str = "/home/takamichi-lab-pc09/elsa/Spatial_AudioCaps/takamichi09/SpatialAudioCaps/meta/stats.pt"
    ) -> None:
        super().__init__()

        self.audio_dir = Path(foa_folder)
        self.n_fft = n_fft
        self.hop = hop

        # 1. CSVファイルを読み込む
        try:
            full_df = pd.read_csv(foa_metadata_csv)
            # カラム名の前後の空白を念のため削除
            full_df.columns = full_df.columns.str.strip()
        except FileNotFoundError:
            raise IOError(f"FATAL: Metadata CSV file not found at: {foa_metadata_csv}")
        except Exception as e:
            raise IOError(f"FATAL: Error reading CSV file: {e}")

        # 2. 指定されたsplitでデータをフィルタリング
        if 'split' not in full_df.columns:
            raise ValueError("FATAL: 'split' column not found in the CSV file.")
        
        # この時点で self.meta を定義する
        self.meta = full_df[full_df['split'] == split].reset_index(drop=True)
        
        if len(self.meta) == 0:
            print(f"Warning: No data found for the split '{split}' in {foa_metadata_csv}. The dataset will be empty.")
            return # データがなければここで処理を終了

        print(f"Found {len(self.meta)} samples for split '{split}' in the CSV.")

        # 3. 存在する音声ファイルのみにメタデータをさらに絞り込む
        valid_indices = []
        for idx, row in self.meta.iterrows():
            expected_path = self.audio_dir / row['foa_filename']
            if expected_path.is_file():
                valid_indices.append(idx)
            # else:
            #     print(f"[Warning] Audio file not found, skipping: {row['foa_filename']}")

        original_count = len(self.meta)
        self.meta = self.meta.loc[valid_indices].reset_index(drop=True)
        final_count = len(self.meta)

        if final_count < original_count:
            print(f"Warning: Filtered out {original_count - final_count} samples because audio files were not found.")
        
        print(f"Dataset initialized with {final_count} valid samples.")

        # 4. 固定長を計算
        self.foa_len = int(FOA_SR * MAX_DURATION_SEC)
        self.iv_len = int(IV_SR * MAX_DURATION_SEC)

        self.n_fft = n_fft
        self.hop = hop
        stats = torch.load(stats_path) 
        self.area_mean = stats["meta/area"]["mean"]
        self.area_std = stats["meta/area"]["std"]
        self.dist_mean = stats["meta/distance"]["mean"]
        self.dist_std = stats["meta/distance"]["std"]

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> dict:
        row = self.meta.iloc[idx]
        fname = row["foa_filename"]
        caption_text = row["caption"]

        # --------------------------------------------------------------
        # 1) Load and process audio
        # --------------------------------------------------------------
        path = self.audio_dir / f"{fname}"
        wav, orig_sr = torchaudio.load(path)

        if orig_sr != FOA_SR:
            wav = torchaudio.functional.resample(wav, orig_freq=orig_sr, new_freq=FOA_SR)

        if wav.size(1) < self.foa_len:
            repeat_factor = math.ceil(self.foa_len / wav.size(1))
            wav = wav.repeat(1, repeat_factor)[:, : self.foa_len]
        else:
            wav = wav[:, : self.foa_len]

        omni_48k = wav[0]

        wav_16k = torchaudio.functional.resample(wav, orig_freq=FOA_SR, new_freq=IV_SR)
        if wav_16k.size(1) > self.iv_len:
            wav_16k = wav_16k[:, : self.iv_len]

        I_act, I_rea = foa_to_iv(
            wav_16k.unsqueeze(0), sr=IV_SR, n_fft=self.n_fft, hop=self.hop,
        )
        I_act, I_rea = I_act.squeeze(0), I_rea.squeeze(0)

        # --------------------------------------------------------------
        # 2) Get spatial regression targets from CSV
        # --------------------------------------------------------------
        azimuth = row['meta/azimuth']
        elevation = row['meta/elevation']
        area = row['meta/area']
        distance = row['meta/distance']

        direction_vec = torch.tensor(
            [np.deg2rad(azimuth), np.deg2rad(elevation)],
            dtype=torch.float32
        )
        area_val = torch.tensor(area, dtype=torch.float32)
        distance_val = torch.tensor(distance, dtype=torch.float32)

        area_norm = (area_val - self.area_mean)/self.area_std
        distance_norm = (distance_val - self.dist_mean)/self.dist_std
        # --------------------------------------------------------------
        # 3) Return all data in a dictionary
        # --------------------------------------------------------------
        audio_features = {
            "i_act": I_act,
            "i_rea": I_rea,
            "omni_48k": omni_48k
        }

        return {
            "audio": audio_features,
            # 【変更】テキストを生文字列のまま返す
            "text": caption_text,
           
            "has_spatial": torch.tensor(True), # 空間情報があることを示すフラグ
           
        }
        #以下の物理量は一旦無視
        # "direction": direction_vec,
        # "distance": distance_norm,
        # "area": area_norm