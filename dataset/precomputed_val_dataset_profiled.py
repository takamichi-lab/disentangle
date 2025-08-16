import math, time
from pathlib import Path
import pandas as pd
import torch, torchaudio
from torch.utils.data import Dataset

class PrecomputedValDatasetProfiled(Dataset):
    """
    index_csv: precompute_val.py が出力した val_precomputed.csv
      必須列: audiocap_id, rir_path, foa_path, feat_path, caption
    rir_meta_csv: RIRカタログ
    """
    def __init__(self, index_csv: str, rir_meta_csv: str, root: str | None = None, profile_io: bool=False):
        self.root = Path(root) if root else Path(index_csv).parent
        self.df = pd.read_csv(index_csv)
        meta = pd.read_csv(rir_meta_csv)
        self.df = self.df.merge(meta, on="rir_path", how="left")
        self._space_map = {p: i for i, p in enumerate(sorted(self.df["rir_path"].unique().tolist()))}
        self.profile_io = profile_io
        self._clock = time.perf_counter

    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int):
        row=self.df.iloc[idx]; _prof={} if self.profile_io else None
        if _prof is not None:
            t0=self._clock(); feat=torch.load(self.root/row.feat_path,map_location="cpu")
            _prof["load_feat_pt"]=self._clock()-t0
        else:
            feat=torch.load(self.root/row.feat_path,map_location="cpu")
        if _prof is not None:
            t0=self._clock(); foa,sr=torchaudio.load(self.root/row.foa_path)
            _prof["load_foa_wav"]=self._clock()-t0
        else:
            foa,sr=torchaudio.load(self.root/row.foa_path)
        omni_48k=foa[0]
        az=float(row.get("azimuth_deg",0.0)); el=float(row.get("elevation_deg",0.0))
        direction_vec=torch.tensor([math.radians(az), math.radians(el)],dtype=torch.float32)
        rir_meta={
            "azimuth_deg": az, "elevation_deg": el,
            "direction_vec": direction_vec,
            "distance": float(row.get("source_distance_m",0.0)),
            "area_m2": float(row.get("area_m2",0.0)),
            "fullband_T30_ms": float(row.get("fullband_T30_ms",0.0)),
        }
        return {
            "audio":[{"i_act": feat["i_act"].float(),"i_rea": feat["i_rea"].float(),"omni_48k": omni_48k}],
            "texts":[row.caption],
            "source_id": torch.tensor([int(row.audiocap_id)]),
            "space_id": torch.tensor([self._space_map[row.rir_path]]),
            "rir_meta":[rir_meta],
            **({"_prof": _prof} if _prof is not None else {}),
        }
