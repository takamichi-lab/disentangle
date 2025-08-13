# dataset/precomputed_val_dataset.py
import math
from pathlib import Path
import pandas as pd
import torch, torchaudio
from torch.utils.data import Dataset

class PrecomputedValDataset(Dataset):
    """
    index_csv: precompute_val.py が出力した val_precomputed.csv
      必須列: audiocap_id, rir_path, foa_path, feat_path, caption
    rir_meta_csv: RIRカタログ（rir_catalog_val.csv）。rir_path で join してGT列を付与。
    root: index_csv のあるディレクトリ（相対パス解決用）
    """
    def __init__(self, index_csv: str, rir_meta_csv: str, root: str | None = None):
        self.root = Path(root) if root else Path(index_csv).parent
        self.df = pd.read_csv(index_csv)
        meta = pd.read_csv(rir_meta_csv)
        self.df = self.df.merge(meta, on="rir_path", how="left")
        # rir_path → space_id（安定な整数）
        self._space_map = {p: i for i, p in enumerate(sorted(self.df["rir_path"].unique().tolist()))}

    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        # ① 特徴読み込み（i_act/i_rea）
        feat = torch.load(self.root / row.feat_path, map_location="cpu")
        # ② FOA WAV から W(omni) だけ取り出し（必要なら）
        foa, sr = torchaudio.load(self.root / row.foa_path)  # (4, T)
        omni_48k = foa[0]  # W

        audio_dict = {"i_act": feat["i_act"].float(), "i_rea": feat["i_rea"].float(), "omni_48k": omni_48k}
        az = float(row.get("azimuth_deg", 0.0)) 
        el = float(row.get("elevation_deg", 0.0))
        direction_vec = torch.tensor([math.radians(az), math.radians(el)], dtype=torch.float32)
        rir_meta = {
            "azimuth_deg": float(row.get("azimuth_deg", 0.0)),
            "elevation_deg": float(row.get("elevation_deg", 0.0)),
            "direction_vec": direction_vec,
            "distance": float(row.get("source_distance_m", 0.0)),
            "area_m2": float(row.get("area_m2", 0.0)),
            "fullband_T30_ms": float(row.get("fullband_T30_ms", 0.0)),
        }
        return {
            "audio":   [audio_dict],                      # n_views=1 として扱う（collate_fn互換）
            "texts":   [row.caption],
            "source_id": torch.tensor([int(row.audiocap_id)]),                 # = 音源ID（dry）
            "space_id":  torch.tensor([self._space_map[row.rir_path]]),        # = 空間ID（RIR）
            "rir_meta":  [rir_meta],
        }
