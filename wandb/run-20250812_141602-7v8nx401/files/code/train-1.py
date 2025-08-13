#!/usr/bin/env python3
# train.py — supervised contrastive (source/space) + per-epoch & quick validation (retrieval)
import math, random, sys, yaml
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.audio_rir_dataset import AudioRIRDataset, collate_fn  # existing

from model.delsa_model import DELSA  # repo root

# Added utilities (new files)
from dataset.precomputed_val_dataset import PrecomputedValDataset
from utils.metrics import cosine_sim, recall_at_k

random.seed(42)

# -------------------- config --------------------
def _select_device(raw: str | None) -> str:
    if raw is None or raw.lower() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return raw

def load_config(path: str | None = None) -> dict:
    if path is None:
        if len(sys.argv) > 1 and sys.argv[1].endswith((".yml", ".yaml")):
            path = sys.argv[1]
        else:
            path = "config.yaml"
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError as e:
        raise SystemExit(f"[ERR] YAML not found: {path}") from e

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
        # train/val csvs
        "audio_csv_train": "AudioCaps_csv/train.csv",
        "rir_csv_train":   "RIR_dataset/rir_catalog_train.csv",
        "audio_csv_val":   "AudioCaps_csv/val.csv",
        "rir_csv_val":     "RIR_dataset/rir_catalog_val.csv",
        "audio_base": "AudioCaps_mp3",
        # precomputed val
        "val_precomp_root": None,   # e.g., "Spatial_AudioCaps/.../for_delsa_spatialAudio"
        "val_index_csv":    None,   # if None -> <root>/val_precomputed.csv
        "val_batch_size":   16,
        # quick val
        "val_on_start": True,
        "val_every_steps": 0,
        "val_subset_batches": 2,
    }
    for k, v in defaults.items():
        cfg.setdefault(k, v)
    cfg["device"] = _select_device(cfg["device"])
    return cfg

# -------------------- losses --------------------
def sup_contrast(a, b, labels, logit_scale, eps=1e-8, *, symmetric=True, exclude_diag=False):
    """Supervised contrastive loss (InfoNCE-style) given labels defining positives."""
    a = F.normalize(a, dim=1); b = F.normalize(b, dim=1)
    scale = torch.clamp(logit_scale, max=math.log(1e2)).exp()
    logits_t2a = (a @ b.T) * scale
    logits_a2t = logits_t2a.T if symmetric else None

    def _dir_loss(logits):
        B = logits.size(0)
        pos_mask = labels[:, None].eq(labels[None, :])
        max_sim, _ = logits.max(dim=1, keepdim=True)
        logits = logits - max_sim.detach()
        diag_mask = torch.eye(B, dtype=torch.bool, device=logits.device) if exclude_diag else torch.zeros(B,B,dtype=torch.bool, device=logits.device)
        exp_sim = torch.exp(logits) * (~diag_mask)
        denom = exp_sim.sum(dim=1, keepdim=True) + eps
        log_prob = logits - denom.log()
        mean_log_pos = (log_prob * pos_mask.float()).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
        valid = pos_mask.any(dim=1)
        return -mean_log_pos[valid].mean()

    loss_t2a = _dir_loss(logits_t2a)
    if symmetric:
        return 0.5 * (loss_t2a + _dir_loss(logits_a2t))
    return loss_t2a

def physical_loss(model_output, batch_data, isNorm=True, dataloader=None):
    """Direction cosine loss + MSEs for area/distance/reverb; normalized or original scale."""
    if isNorm:
        pred = {"direction": model_output["direction"], "area": model_output["area"],
                "distance": model_output["distance"], "reverb": model_output["reverb"]}
        true = {"direction": batch_data["rir_meta"]["direction_vec"],
                "area": batch_data["rir_meta"]["area_m2_norm"],
                "distance": batch_data["rir_meta"]["distance_norm"],
                "reverb": batch_data["rir_meta"]["t30_norm"]}
    else:
        area_mean, area_std   = dataloader.dataset.area_mean, dataloader.dataset.area_std
        dist_mean, dist_std   = dataloader.dataset.dist_mean, dataloader.dataset.dist_std
        t30_mean, t30_std     = dataloader.dataset.t30_mean, dataloader.dataset.t30_std
        pred = {"direction": model_output["direction"],
                "area": model_output["area"] * area_std + area_mean,
                "distance": model_output["distance"] * dist_std + dist_mean,
                "reverb": model_output["reverb"] * t30_std + t30_mean}
        true = {"direction": batch_data["rir_meta"]["direction_vec"],
                "area": batch_data["rir_meta"]["area_m2"],
                "distance": batch_data["rir_meta"]["distance"],
                "reverb": batch_data["rir_meta"]["fullband_T30_ms"]}
    loss_dir = (-1 * F.cosine_similarity(pred["direction"], true["direction"], dim=1)).mean()
    loss_area = F.mse_loss(pred["area"].squeeze(-1), true["area"])
    loss_distance= F.mse_loss(pred["distance"].squeeze(-1), true["distance"])
    loss_reverb = F.mse_loss(pred["reverb"].squeeze(-1), true["reverb"])
    total = loss_dir + loss_area + loss_distance + loss_reverb
    return {"loss_dir": loss_dir.item(),
            "loss_distance": loss_distance.item(),
            "loss_area": loss_area.item(),
            "loss_reverb": loss_reverb.item()}, total

def recursive_to(obj, device):
    if isinstance(obj, torch.Tensor): return obj.to(device)
    if isinstance(obj, dict):  return {k: recursive_to(v, device) for k,v in obj.items()}
    if isinstance(obj, list):  return [recursive_to(v, device) for v in obj]
    return obj

def to_audio_tensors(audio_field, device):
    """Accepts collate output as list[dict] or dict[str, Tensor]; returns dict[str, Tensor] to device."""
    if isinstance(audio_field, list):
        keys = audio_field[0].keys()
        out = {k: torch.stack([d[k] for d in audio_field]) for k in keys}
    elif isinstance(audio_field, dict):
        out = audio_field
    else:
        raise TypeError("Unexpected audio field type")
    return {k: v.to(device, non_blocking=True) for k,v in out.items()}

# -------------------- quick val --------------------
@torch.no_grad()
def eval_val_losses_quick(model, loader, device, max_batches=2):
    """Smoketest: compute losses on a few batches quickly."""
    model.eval()
    sums = {"space":0.0,"source":0.0,"physical":0.0,"count":0}
    for bidx, batch in enumerate(loader):
        if max_batches is not None and bidx >= max_batches: break
        if batch is None: continue
        batch_data = {k: recursive_to(v, device) for k, v in batch.items() if k not in ["audio","texts"]}
        audio = to_audio_tensors(batch["audio"], device)
        texts  = batch["texts"]
        src_lb = batch["source_id"].reshape(-1).to(device)
        spa_lb = batch["space_id"].reshape(-1).to(device)
        out = model(audio, texts)
        a_s  = F.normalize(out["audio_space_emb"],  dim=-1)
        t_s  = F.normalize(out["text_space_emb"],   dim=-1)
        a_sr = F.normalize(out["audio_source_emb"], dim=-1)
        t_sr = F.normalize(out["text_source_emb"],  dim=-1)
        T = out["logit_scale"].exp()
        l_sp  = sup_contrast(a_s,  t_s,  spa_lb, T)
        l_sr  = sup_contrast(a_sr, t_sr, src_lb, T)
        _, l_phys = physical_loss(out, batch_data, isNorm=False, dataloader=loader)
        sums["space"] += l_sp.item(); sums["source"] += l_sr.item()
        sums["physical"] += l_phys.item(); sums["count"] += 1
    n = max(1, sums["count"])
    return { "quick/val_loss_space": sums["space"]/n,
             "quick/val_loss_source": sums["source"]/n,
             "quick/val_loss_physical": sums["physical"]/n }

@torch.no_grad()
def eval_retrieval(model, loader, device, use_wandb=True, epoch=None, max_batches=None):
    """Compute source/space retrieval (view→group mean→R@K)."""
    try:
        import wandb
    except Exception:
        use_wandb = False
    model.eval()
    bufs = {"t_spa":[], "a_spa":[], "t_src":[], "a_src":[], "ids_src":[], "ids_spa":[]}
    for bidx, batch in enumerate(loader):
        if max_batches is not None and bidx >= max_batches: break
        if batch is None: continue
        audio = to_audio_tensors(batch["audio"], device)
        texts = batch["texts"]
        out = model(audio, texts)
        bufs["t_spa"].append(out["text_space_emb"].detach().cpu())
        bufs["a_spa"].append(out["audio_space_emb"].detach().cpu())
        bufs["t_src"].append(out["text_source_emb"].detach().cpu())
        bufs["a_src"].append(out["audio_source_emb"].detach().cpu())
        bufs["ids_src"].append(batch["source_id"].reshape(-1).cpu())
        bufs["ids_spa"].append(batch["space_id"].reshape(-1).cpu())
    for k in bufs: bufs[k] = torch.cat(bufs[k], dim=0)

    def group_mean(emb, gids):
        uniq = gids.unique(sorted=True)
        pooled = torch.stack([emb[gids==u].mean(0) for u in uniq])
        return pooled, uniq

    t_src, id_src = group_mean(bufs["t_src"], bufs["ids_src"])
    a_src, _      = group_mean(bufs["a_src"], bufs["ids_src"])
    t_spa, id_spa = group_mean(bufs["t_spa"], bufs["ids_spa"])
    a_spa, _      = group_mean(bufs["a_spa"], bufs["ids_spa"])

    S_src = cosine_sim(a_src, t_src)  # [N_t, N_a]
    S_spa = cosine_sim(a_spa, t_spa)

    src_T2A = recall_at_k(S_src,   id_src, id_src, ks=(1,5,10))
    src_A2T = recall_at_k(S_src.T, id_src, id_src, ks=(1,5,10))
    spa_T2A = recall_at_k(S_spa,   id_spa, id_spa, ks=(1,5,10))
    spa_A2T = recall_at_k(S_spa.T, id_spa, id_spa, ks=(1,5,10))

    mets = {**{f"SRC/T2A/{k}":v for k,v in src_T2A.items()},
            **{f"SRC/A2T/{k}":v for k,v in src_A2T.items()},
            **{f"SPA/T2A/{k}":v for k,v in spa_T2A.items()},
            **{f"SPA/A2T/{k}":v for k,v in spa_A2T.items()}}
    if use_wandb:
        wandb.log({"epoch": epoch, **mets})
    return mets

# -------------------- main --------------------
def main():
    cfg = load_config()
    device = cfg["device"]
    # wandb
    if cfg["wandb"]:
        import wandb
        wandb.init(project=cfg["proj"], name=cfg["run_name"], config=cfg, save_code=True, mode="online")

    # ---- Train loader (original on-the-fly) ----
    train_ds = AudioRIRDataset(csv_audio=cfg["audio_csv_train"], base_dir=cfg["audio_base"],
                               csv_rir=cfg["rir_csv_train"], n_views=cfg["n_views"],
                               split=cfg["split"], batch_size=cfg["batch_size"])
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=4,
                          collate_fn=collate_fn, pin_memory=False, persistent_workers=True)

    # ---- Val loader (precomputed fixed) ----
    val_root = cfg.get("val_precomp_root")
    val_csv  = cfg.get("val_index_csv") or (str(Path(val_root)/"val_precomputed.csv") if val_root else None)
    if not val_csv or not Path(val_csv).exists():
        raise SystemExit(f"[ERR] val_precomputed.csv が見つかりません: {val_csv}")
    val_ds = PrecomputedValDataset(index_csv=val_csv, rir_meta_csv=cfg["rir_csv_val"], root=val_root)
    val_dl = DataLoader(val_ds, batch_size=cfg.get("val_batch_size", cfg["batch_size"]),
                        shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=False, persistent_workers=True)

    # ---- model / optim ----
    model = DELSA(audio_encoder_cfg={}, text_encoder_cfg={}).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

    # ---- quick val at start ----
    if cfg.get("val_on_start", True):
        kb = cfg.get("val_subset_batches", 2)
        print(f"[Warmup] quick val on start: {kb} batch(es)")
        qloss = eval_val_losses_quick(model, val_dl, device, max_batches=kb)
        print("[Warmup] " + " | ".join(f"{k}:{v:.4f}" for k,v in qloss.items()))
        qret = eval_retrieval(model, val_dl, device, use_wandb=False, epoch=0, max_batches=kb)
        print("[Warmup] retrieval " + " | ".join(f"{k}:{v:.3f}" for k,v in qret.items()))

    # ---- training loop ----
    for ep in range(1, cfg["epochs"]+1):
        model.train()
        for step, batch in enumerate(train_dl, 1):
            if batch is None: continue
            batch_data = {k: recursive_to(v, device) for k, v in batch.items() if k not in ["audio","texts"]}
            audio = to_audio_tensors(batch["audio"], device)
            texts  = batch["texts"]
            src_lb = batch["source_id"].reshape(-1).to(device)
            spa_lb = batch["space_id"].reshape(-1).to(device)

            out = model(audio, texts)
            a_s  = F.normalize(out["audio_space_emb"],  dim=-1)
            t_s  = F.normalize(out["text_space_emb"],   dim=-1)
            a_sr = F.normalize(out["audio_source_emb"], dim=-1)
            t_sr = F.normalize(out["text_source_emb"],  dim=-1)
            T = out["logit_scale"].exp()

            loss_space  = sup_contrast(a_s,  t_s,  spa_lb, T)
            loss_source = sup_contrast(a_sr, t_sr, src_lb, T)
            phys_log, phys_loss = physical_loss(out, batch_data, isNorm=True, dataloader=train_dl)

            loss = loss_space + loss_source + phys_loss
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

            if step % 10 == 0:
                print(f"Epoch {ep} Step {step}/{len(train_dl)}  space={loss_space:.4f}  src={loss_source:.4f}  phys={phys_loss:.4f}  total={loss:.4f}")
                if cfg["wandb"]:
                    import wandb
                    wandb.log({
                        "loss/space": loss_space.item(), "loss/source": loss_source.item(),
                        "loss/physical": phys_loss.item(), "loss/dir": phys_log["loss_dir"],
                        "loss/distance": phys_log["loss_distance"], "loss/area": phys_log["loss_area"],
                        "loss/reverb": phys_log["loss_reverb"], "logit_scale": out["logit_scale"].item(),
                        "loss/mean": loss.item(), "epoch": ep, "step": step + (ep-1)*len(train_dl)
                    })

            # quick val during training
            if cfg.get("val_every_steps", 0) and (step % cfg["val_every_steps"] == 0):
                kb = cfg.get("val_subset_batches", 2)
                qloss = eval_val_losses_quick(model, val_dl, device, max_batches=kb)
                print(f"[QuickVal step {step}] " + " | ".join(f"{k}:{v:.4f}" for k,v in qloss.items()))
                qret = eval_retrieval(model, val_dl, device, use_wandb=False, epoch=ep, max_batches=kb)
                print(f"[QuickVal step {step}] retrieval " + " | ".join(f"{k}:{v:.3f}" for k,v in qret.items()))

        # ---- full validation per epoch ----
        model.eval()
        val_losses = {"space":0.0,"source":0.0,"physical":0.0,"direction":0.0,"distance":0.0,"area":0.0,"reverb":0.0,"count":0}
        with torch.no_grad():
            for batch in val_dl:
                if batch is None: continue
                batch_data = {k: recursive_to(v, device) for k, v in batch.items() if k not in ["audio","texts"]}
                audio = to_audio_tensors(batch["audio"], device)
                texts  = batch["texts"]
                src_lb = batch["source_id"].reshape(-1).to(device)
                spa_lb = batch["space_id"].reshape(-1).to(device)

                out = model(audio, texts)
                a_s  = F.normalize(out["audio_space_emb"],  dim=-1)
                t_s  = F.normalize(out["text_space_emb"],   dim=-1)
                a_sr = F.normalize(out["audio_source_emb"], dim=-1)
                t_sr = F.normalize(out["text_source_emb"],  dim=-1)
                T = out["logit_scale"].exp()

                l_sp  = sup_contrast(a_s,  t_s,  spa_lb, T)
                l_sr  = sup_contrast(a_sr, t_sr, src_lb, T)
                phys_log, l_phys = physical_loss(out, batch_data, isNorm=False, dataloader=val_dl)

                val_losses["space"]    += l_sp.item()
                val_losses["source"]   += l_sr.item()
                val_losses["physical"] += l_phys.item()
                val_losses["direction"]+= phys_log["loss_dir"]
                val_losses["distance"] += phys_log["loss_distance"]
                val_losses["area"]     += phys_log["loss_area"]
                val_losses["reverb"]   += phys_log["loss_reverb"]
                val_losses["count"]    += 1

        n = max(1, val_losses["count"])
        val_mean = (val_losses["space"]/n + val_losses["source"]/n + val_losses["physical"]/n)
        print(f"Epoch {ep}  [VAL] space={val_losses['space']/n:.4f}  src={val_losses['source']/n:.4f}  phys={val_losses['physical']/n:.4f}")

        mets = eval_retrieval(model, val_dl, device, use_wandb=cfg.get("wandb", False), epoch=ep)
        if cfg["wandb"]:
            import wandb
            wandb.log({
                "val/loss_space": val_losses["space"]/n,
                "val/loss_source": val_losses["source"]/n,
                "val/loss_physical": val_losses["physical"]/n,
                "val/loss_direction": val_losses["direction"]/n,
                "val/loss_distance": val_losses["distance"]/n,
                "val/loss_area":     val_losses["area"]/n,
                "val/loss_reverb":   val_losses["reverb"]/n,
                "val/loss_mean":     val_mean,
                "epoch": ep, **{f"val/{k}": v for k,v in mets.items()}
            })

        # ---- checkpoint ----
        ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep},
                   ckpt_dir/f"ckpt_sup_ep{ep}.pt")
        print(f"[✓] Saved checkpoint for epoch {ep}")

if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            import wandb
            if wandb.run is not None: wandb.finish()
        except Exception:
            pass
