
#!/usr/bin/env python3
"""
Quick validation runner for DELSA (real precomputed val set).

- Reads the same paths from config.yaml (like train.py).
- Builds PrecomputedValDataset/DataLoader.
- Runs the val loop (space/source contrastive + physical losses).
- Computes retrieval metrics (SRC/SPA + X-SRC/X-SPA) identical to train.py.
- Computes invariance ratios (IR) on the FULL val set (not just the last batch).

Usage:
    python3 quick_val_full.py                 # uses ./config.yaml
    python3 quick_val_full.py path/to/config.yaml

Notes:
- Expects config.yaml to define: val_precomp_root (or val_index_csv), rir_csv_val, val_batch_size.
- You can optionally pass a checkpoint via --ckpt to evaluate a trained model.
"""
from tqdm.auto import tqdm
import argparse, math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Reuse project modules
from train_for_singleGPU import load_config, sup_contrast, physical_loss, eval_retrieval,recursive_to
from dataset.precomputed_val_dataset import PrecomputedValDataset
from dataset.audio_rir_dataset_old import collate_fn
from model.delsa_model import DELSA
from utils.metrics import invariance_ratio, cosine_sim
def _wrap(loader, desc: str, leave=False):
    try:
        total = len(loader)
    except TypeError:
        total = None
    return tqdm(loader, total=total, desc=desc, dynamic_ncols=True, mininterval=0.2, leave=leave)
def _device(auto: str) -> torch.device:
    if auto in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(auto)

@torch.no_grad()
def _embed_all(model, loader, device):
    """
    Pass the whole val set and collect embeddings & IDs (for global IR).
    """
    bufs = {"t_spa":[], "a_spa":[], "t_src":[], "a_src":[], "ids_src":[], "ids_spa":[]}
    model.eval()
    for batch in _wrap(loader, "Embed all"):
        if batch is None:
            continue
        audio = {k: torch.stack([d[k] for d in batch["audio"]]).to(device)
                 for k in ("i_act","i_rea","omni_48k")}
        texts = batch["texts"]
        out = model(audio, texts)
        bufs["t_spa"].append(out["text_space_emb"].detach().cpu())
        bufs["a_spa"].append(out["audio_space_emb"].detach().cpu())
        bufs["t_src"].append(out["text_source_emb"].detach().cpu())
        bufs["a_src"].append(out["audio_source_emb"].detach().cpu())
        bufs["ids_src"].append(batch["source_id"].reshape(-1).cpu())
        bufs["ids_spa"].append(batch["space_id"].reshape(-1).cpu())
    for k in bufs:
        bufs[k] = torch.cat(bufs[k], dim=0) if len(bufs[k]) else torch.empty(0)
    return bufs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default=None, help="config.yaml path (optional)")
    ap.add_argument("--ckpt", type=str, default=None, help="checkpoint path to evaluate")
    args = ap.parse_args()

    # --- Load config (reuse same helper as train.py) ---
    cfg = load_config(args.config)
    device = _device(cfg.get("device", "auto"))
    print(f"[Info] Device = {device}")

    # --- Build val dataset/loader (same as train.py) ---
    val_root = cfg.get("val_precomp_root")
    val_csv  = cfg.get("val_index_csv") or (str(Path(val_root)/"val_precomputed.csv") if val_root else None)
    if not val_csv or not Path(val_csv).exists():
        raise SystemExit(f"[ERR] val_precomputed.csv が見つかりません: {val_csv}")
    rir_csv_val = cfg["rir_csv_val"]
    if not Path(rir_csv_val).exists():
        raise SystemExit(f"[ERR] rir_csv_val が見つかりません: {rir_csv_val}")

    val_ds = PrecomputedValDataset(index_csv=val_csv, rir_meta_csv=rir_csv_val, root=val_root)
    val_bs = int(cfg.get("val_batch_size", cfg.get("batch_size", 8)))
    val_dl = DataLoader(val_ds, batch_size=val_bs, shuffle=False, num_workers=4,
                        collate_fn=collate_fn, pin_memory=False)
    # val_ds 作成の下あたりに追加
    from dataset.audio_rir_dataset_old import AudioRIRDataset

    try:
        train_ds = AudioRIRDataset(
            csv_audio=cfg["audio_csv_train"],
            base_dir=cfg["audio_base"],
            csv_rir=cfg["rir_csv_train"],
            n_views=1, split=cfg.get("split","train"), batch_size=1
        )
        train_stats = {
            "area_mean": train_ds.area_mean, "area_std": train_ds.area_std,
            "dist_mean": train_ds.dist_mean, "dist_std": train_ds.dist_std,
            "t30_mean":  train_ds.t30_mean,  "t30_std":  train_ds.t30_std,
        }
        print("[Info] Loaded train stats for de-normalization.")
    except Exception as e:
        print("[Warn] Could not load train stats; using neutral stats. ->", e)
    # --- Model ---
    model = DELSA(audio_encoder_cfg={}, text_encoder_cfg={}).to(device)
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt.get("model", ckpt), strict=False)
        print(f"[Info] Loaded ckpt: missing={len(missing)}, unexpected={len(unexpected)}")

    # === Validation loop (losses), mirroring train.py ===
    # Stats for de-normalization on val (train stats are expected; here we don't train,
    # so pass via config or keep neutral to report raw-scale loss numerics).
    # train_stats = {
    #     "area_mean": cfg.get("area_mean", 0.0),
    #     "area_std":  cfg.get("area_std",  1.0),
    #     "dist_mean": cfg.get("dist_mean", 0.0),
    #     "dist_std":  cfg.get("dist_std",  1.0),
    #     "t30_mean":  cfg.get("t30_mean",  0.0),
    #     "t30_std":   cfg.get("t30_std",   1.0),
    # }

    val_losses = {"space":0.0,"source":0.0,"physical":0.0,
                  "direction":0.0,"distance":0.0,"area":0.0,"reverb":0.0,"count":0}

    model.eval()
    with torch.no_grad():
        for batch in _wrap(val_dl, "VAL (loss)"):
            if batch is None:
                continue
            batch_data = {k: recursive_to(v, device) for k, v in batch.items() if k not in ["audio","texts"]}
            audio = {k: torch.stack([d[k] for d in batch["audio"]]).to(device)
                     for k in ("i_act","i_rea","omni_48k")}
            texts  = batch["texts"]
            src_lb = batch["source_id"].reshape(-1).to(device)
            spa_lb = batch["space_id"].reshape(-1).to(device)

            out = model(audio, texts)
            a_spa  = F.normalize(out["audio_space_emb"],  dim=-1)
            t_spa  = F.normalize(out["text_space_emb"],   dim=-1)
            a_src  = F.normalize(out["audio_source_emb"], dim=-1)
            t_src  = F.normalize(out["text_source_emb"],  dim=-1)
            logit_s = out["logit_scale"]

            l_sp  = sup_contrast(a_spa,  t_spa,  spa_lb, logit_s)
            l_sr  = sup_contrast(a_src,  t_src,  src_lb, logit_s)
            phys_log, l_phys = physical_loss(out, batch_data, isNorm=False, dataloader=val_dl, stats=train_stats)

            val_losses["space"]    += float(l_sp.item())
            val_losses["source"]   += float(l_sr.item())
            val_losses["physical"] += float(l_phys.item())
            val_losses["direction"]+= float(phys_log["loss_dir"])
            val_losses["distance"] += float(phys_log["loss_distance"])
            val_losses["area"]     += float(phys_log["loss_area"])
            val_losses["reverb"]   += float(phys_log["loss_reverb"])
            val_losses["count"]    += 1
            n = max(1, val_losses["count"])
            tqdm.write(
                f"[val-step] space={val_losses['space']/n:.4f} src={val_losses['source']/n:.4f} phys={val_losses['physical']/n:.4f}"
            )
    n = max(1, val_losses["count"])
    print(f"[VAL] space={val_losses['space']/n:.4f}  src={val_losses['source']/n:.4f}  phys={val_losses['physical']/n:.4f}")
    print(f"      dir={val_losses['direction']/n:.4f}  dist={val_losses['distance']/n:.4f}  area={val_losses['area']/n:.4f}  reverb={val_losses['reverb']/n:.4f}")

    # === Retrieval metrics (same routine as train.py) ===
    mets = eval_retrieval(model, _wrap(val_dl, "VAL (retrieval)", leave=False), device, use_wandb=False, epoch=0)
    print("\n[Retrieval metrics]")
    for k in sorted(mets.keys()):
        print(f"{k}: {mets[k]:.6f}")

    # === Global Invariance Ratios (IR) over the full val set ===
    bufs = _embed_all(model, val_dl, device)
    ir_audio_space  = invariance_ratio(bufs["a_spa"],  bufs["ids_src"], bufs["ids_spa"])
    ir_audio_source = invariance_ratio(bufs["a_src"],  bufs["ids_src"], bufs["ids_spa"])
    ir_text_space   = invariance_ratio(bufs["t_spa"],  bufs["ids_src"], bufs["ids_spa"])
    ir_text_source  = invariance_ratio(bufs["t_src"],  bufs["ids_src"], bufs["ids_spa"])

    print("\n[Invariance Ratio] (higher is better for the corresponding embedding)")
    for name, ir in [
        ("IR/audio_space",  ir_audio_space),
        ("IR/audio_source", ir_audio_source),
        ("IR/text_space",   ir_text_space),
        ("IR/text_source",  ir_text_source),
    ]:
        print(f"{name}: IR_space={ir['IR_space']:.6f}, IR_source={ir['IR_source']:.6f}, "
              f"num_ss_ds={ir['num_ss_ds']}, num_sd_ss={ir['num_sd_ss']}")

if __name__ == "__main__":
    main()
