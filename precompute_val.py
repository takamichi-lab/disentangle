# scripts/precompute_val.py
import argparse, json, torchaudio, torch, pandas as pd
from pathlib import Path
from dataset.audio_rir_dataset import AudioRIRDataset, foa_to_iv, rewrite_caption
import torch 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import random
import wandb
import math
import sys
import yaml
random.seed(42); torch.manual_seed(42)
def _select_device(raw: str |None) -> str:
    if raw is None or raw.lower() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return raw

def load_config(path: str | None = None) -> dict:
    """Load hyper‑parameters from a YAML file and fill in sane defaults."""

    # 1️⃣  determine YAML path (CLI arg 1 or default "config.yaml")
    if path is None:
        if len(sys.argv) > 1 and sys.argv[1].endswith((".yml", ".yaml")):
            path = sys.argv[1]
        else:
            path = "config.yaml"

    # 2️⃣  read YAML
    try:
        with open(path, "r", encoding="utf‑8") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError as e:
        raise SystemExit(f"[ERR] YAML not found: {path}") from e

    # 3️⃣  defaults
    defaults = {
        "split": "train",
        "batch_size": 8,
        "n_views": 4,
        "epochs": 5,
        "lr": 0.0001,
        "device": "auto",
        "wandb": True,
        "proj": "delsa-sup-contrast",
        "run_name": None,
        "audio_csv_train": "AudioCaps_csv/train.csv",
        "rir_csv_train": "RIR_dataset/rir_catalog_train.csv",
        "audio_csv_val": "AudioCaps_csv/val.csv",
        "rir_csv_val": "RIR_dataset/rir_catalog_val.csv",
        "base_dir": "RIR_dataset",
        "audio_base": None,

    }
    for k, v in defaults.items():
        cfg.setdefault(k, v)

    # 4️⃣  post‑process
    cfg["device"] = _select_device(cfg["device"])
    return cfg

def main(args):
    cfg = load_config()
    if args.csv_audio: cfg["audio_csv_val"] = args.csv_audio
    if args.csv_rir: cfg["rir_csv_val"] = args.csv_rir
    if args.base_dir: cfg["audio_base"] = args.base_dir

    cfg["out_dir"] = args.out_dir or cfg.get("out_dir", "data/val_precomputed")
    if args.n_views is not None:
        cfg["n_views_val"] = args.n_views

    ds = AudioRIRDataset(
        csv_audio=cfg["audio_csv_val"], base_dir=cfg["audio_base"],
        csv_rir=cfg["rir_csv_val"],     split="val",
        n_views=cfg["n_views_val"],     share_rir=False,  # ←ランダム性を排除
        batch_size=None)
    out_root = Path(cfg["out_dir"]); out_root.mkdir(parents=True, exist_ok=True)
    records = []

    rir_list = sorted(ds.rir_paths)
    # n_views <=0を「全部のRIRを使う」と解釈
    n_views = len(rir_list) if cfg["n_views"] <= 0 else min(cfg["n_views"],len(rir_list))

    torch.set_grad_enabled(False)
    for idx in range(len(ds)):

        row = ds.audio_df.iloc[idx]
        dry   = ds._load_dry(row["audiocap_id"])
        for rir_path in rir_list[:n_views]:
            foa   = ds._apply_rir(dry, rir_path)            # 4-ch FOA:contentReference[oaicite:0]{index=0}
            foa16 = torchaudio.functional.resample(foa, 48_000, 16_000)
            # 1) タプル全体を受け取る
            i_act, i_rea = foa_to_iv(foa16.unsqueeze(0))  # ← [0] を削除

            # 2) バッチ次元 (B=1) を潰す
            i_act, i_rea = i_act.squeeze(0), i_rea.squeeze(0)
            # --- 保存 ---
            key   = f"{row.audiocap_id}_{Path(rir_path).stem}"
            (out_root / "foa").mkdir(exist_ok=True)
            (out_root / "feat").mkdir(exist_ok=True)
            torchaudio.save(out_root/f"foa/{key}.wav", foa, 48_000)
            torch.save({'i_act': i_act, 'i_rea': i_rea}, out_root/f"feat/{key}.pt")

            meta   = ds.rir_meta[rir_path].copy()
            caption= rewrite_caption(row["caption"], meta)
            records.append({
                "audiocap_id": row.audiocap_id,
                "rir_path": rir_path,
                "foa_path": f"foa/{key}.wav",
                "feat_path": f"feat/{key}.pt",
                "caption": caption
            })

    pd.DataFrame(records).to_csv(out_root/"val_precomputed.csv", index=False)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv_audio"); p.add_argument("--base_dir")
    p.add_argument("--csv_rir");  p.add_argument("--out_dir")
    p.add_argument("--n_views", type=int, help="<=0 なら RIR 全部を使用")
    main(p.parse_args())
