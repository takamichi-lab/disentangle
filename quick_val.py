# quick_val.py
import torch
from pathlib import Path
from glob import glob

from dataset.precomputed_val_dataset import PrecomputedValDataset   # valセット（前計算）:contentReference[oaicite:5]{index=5}
from dataset.audio_rir_dataset import collate_fn                   # collate（GPU転送はeval側で実施）:contentReference[oaicite:6]{index=6}
from torch.utils.data import DataLoader
from model.delsa_model import DELSA                                # モデル本体:contentReference[oaicite:7]{index=7}
from utils.metrics import eval_retrieval                           # 取得評価（内部で .to(device)）:contentReference[oaicite:8]{index=8}
import yaml

def load_cfg(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def latest_ckpt():
    files = sorted(glob("checkpoints/ckpt_sup_ep*.pt"))
    return files[-1] if files else None

def main():
    cfg = load_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Val dataset/loader（前計算をそのまま読む） ---
    val_root = cfg.get("val_precomp_root")
    val_csv  = cfg.get("val_index_csv") or (str(Path(val_root)/"val_precomputed.csv") if val_root else None)
    if not val_csv or not Path(val_csv).exists():
        raise SystemExit(f"[ERR] val_precomputed.csv が見つかりません: {val_csv}")  # :contentReference[oaicite:9]{index=9}

    val_ds = PrecomputedValDataset(index_csv=val_csv, rir_meta_csv=cfg["rir_csv_val"], root=val_root)  #:contentReference[oaicite:10]{index=10}
    val_dl = DataLoader(val_ds, batch_size=cfg.get("val_batch_size", 16),
                        shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=False)         #:contentReference[oaicite:11]{index=11}

    # --- Model ---
    model = DELSA(audio_encoder_cfg={}, text_encoder_cfg={}).to(device)  # 入力キーは i_act/i_rea/omni_48k:contentReference[oaicite:12]{index=12}
    ckpt = latest_ckpt()
    if ckpt:
        print(f"[i] Load checkpoint: {ckpt}")
        sd = torch.load(ckpt, map_location=device)
        model.load_state_dict(sd["model"], strict=False)
    model.eval()

    # --- Retrieval evaluation（内部で各バッチを .to(device) 済）---
    mets = eval_retrieval(model, val_dl, device, use_wandb=False, epoch=None)  #:contentReference[oaicite:13]{index=13}
    print("\n=== Retrieval (val) ===")
    for k in sorted(mets.keys()):
        print(f"{k}: {mets[k]:.4f}" if isinstance(mets[k], float) else f"{k}: {mets[k]}")

if __name__ == "__main__":
    main()
