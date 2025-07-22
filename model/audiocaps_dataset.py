import torch
import torchaudio
import pandas as pd
import math
from pathlib import Path
from torch.utils.data import Dataset

# foa_dataset_caption.py から必要な関数と定数をインポート
from .foa_dataset_caption import ELSADataset, foa_to_iv, MAX_DURATION_SEC, FOA_SR, IV_SR


class AudioCapsDatasetForELSA(Dataset):
    """
    AudioCaps (1ch) をELSAモデルの評価用に変換するデータセット。
    - 1ch音声を4chに複製し、I_act / I_rea を計算。
    - 1ch音声をそのまま Omni チャンネルとして使用。
    """
    def __init__(
        self,
        mono_folder: str,
        mono_metadata_csv: str,
        n_fft: int = 400,
        hop: int = 100,
    ) -> None:
        super().__init__()
        self.audio_dir = Path(mono_folder)
        # AudioCapsのCSVは 'audiocap_id' と 'caption' 列を持つと想定
        self.meta = pd.read_csv(mono_metadata_csv)

        # 音声ファイルが存在するものだけをフィルタリング
        # AudioCapsのファイル名は 'audiocap_id' + '.wav' の形式を想定
        self.meta['wav_filename'] = self.meta['audiocap_id'].apply(lambda x: f"{x}.mp3")
        valid_idx = [
            idx for idx, row in self.meta.iterrows()
            if (self.audio_dir / row['wav_filename']).is_file()
        ]
        self.meta = self.meta.loc[valid_idx].reset_index(drop=True)
        print(f"Found {len(self.meta)} valid audio files in {mono_folder}")

        # 長さの定数を事前計算
        self.target_len_48k = int(FOA_SR * MAX_DURATION_SEC)  # 480,000
        self.target_len_16k = int(IV_SR * MAX_DURATION_SEC)     # 160,000
        self.n_fft = n_fft
        self.hop = hop

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int):
        row = self.meta.iloc[idx]
        caption_text = row["caption"]
        # .wavか.mp3かは以前の修正で対応済みと仮定
        filename = self.meta['wav_filename'][idx] 
        path = self.audio_dir / filename

        try:
            # --- ファイル読み込みから特徴量計算までをtryブロックで囲む ---
            wav_mono, orig_sr = torchaudio.load(path)

            # 48kHzにリサンプリング
            if orig_sr != FOA_SR:
                wav_mono = torchaudio.functional.resample(wav_mono, orig_freq=orig_sr, new_freq=FOA_SR)

            # 10秒の長さに揃える
            if wav_mono.size(1) < self.target_len_48k:
                repeat_factor = math.ceil(self.target_len_48k / wav_mono.size(1))
                wav_mono = wav_mono.repeat(1, repeat_factor)[:, :self.target_len_48k]
            else:
                wav_mono = wav_mono[:, :self.target_len_48k]
            
            omni_48k = wav_mono.squeeze(0)
            fake_foa_48k = wav_mono.repeat(4, 1)
            wav_16k = torchaudio.functional.resample(fake_foa_48k, orig_freq=FOA_SR, new_freq=IV_SR)
            if wav_16k.size(1) != self.target_len_16k:
                 wav_16k = wav_16k[:, :self.target_len_16k]

            I_act, I_rea = foa_to_iv(wav_16k.unsqueeze(0), sr=IV_SR, n_fft=self.n_fft, hop=self.hop)
            I_act, I_rea = I_act.squeeze(0), I_rea.squeeze(0)

            return {
                "audio": {
                    "i_act":   I_act.squeeze(0),
                    "i_rea":   I_rea.squeeze(0),
                    "omni_48k": omni_48k,
                },
                "text":           caption_text,
                "has_spatial":    torch.tensor(False), # 空間情報がないことを示すフラグ
                "direction":      torch.zeros(2, dtype=torch.float32),
                "distance":       torch.tensor(0.0, dtype=torch.float32), # スカラーテンソルに修正
                "area":           torch.tensor(0.0, dtype=torch.float32),   # スカラーテンソルに修正
            }

        except Exception as e:
            print(f"\n[Warning] Skipping unreadable file {path}: {e}")
            return None