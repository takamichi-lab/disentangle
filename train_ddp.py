#!/usr/bin/env python3
# train_ddp.py — DDP (single-node multi-GPU) + AMP + Grad Accum (rank0-only eval)
import os, math, random, sys, yaml, torch, torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext
import wandb

from utils.metrics import invariance_ratio, leakage_probe_acc, linear_probe_regression
from dataset.audio_rir_dataset import AudioRIRDataset, collate_fn
from dataset.precomputed_val_dataset import PrecomputedValDataset
from utils.metrics import cosine_sim, recall_at_k, recall_at_k_multi, eval_retrieval
from model.delsa_model import DELSA

random.seed(42)

def _is_dist_initialized():
    return dist.is_available() and dist.is_initialized()

def _is_main_process():
    return (not _is_dist_initialized()) or dist.get_rank() == 0

def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    # Prefer LOCAL_RANK; fall back to RANK (ABCI/torchrun envs)
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    return distributed, local_rank

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
        "batch_size": 8,          # per-GPU batch size
        "n_views": 4,
        "epochs": 5,
        "lr": 1e-4,
        "device": "auto",
        "wandb": True,
        "proj": "delsa-sup-contrast",
        "run_name": None,
        # validation (precomputed set)
        "val_precomp_root": None,
        "val_index_csv":    None,
        "val_batch_size":   16,   # per-RANK val batch
        # new
        "use_amp": True,
        "grad_accum": 1,
        "weight_decay": 0.0,
        "max_grad_norm": None,
    }

    for k, v in defaults.items():
        cfg.setdefault(k, v)
    cfg["device"] = _select_device(cfg["device"])
    return cfg

def sup_contrast(a, b, labels, logit_scale, eps=1e-8, *, symmetric=True, exclude_diag=False):
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

def physical_loss(model_output, batch_data, isNorm=True, dataloader=None, stats=None):
    if isNorm:
        pred = {"direction": model_output["direction"], "area": model_output["area"],
                "distance": model_output["distance"], "reverb": model_output["reverb"]}
        true = {"direction": batch_data["rir_meta"]["direction_vec"],
                "area": batch_data["rir_meta"]["area_m2_norm"],
                "distance": batch_data["rir_meta"]["distance_norm"],
                "reverb": batch_data["rir_meta"]["t30_norm"]}
    else:
        area_mean, area_std = stats["area_mean"], stats["area_std"]
        distance_mean, distance_std = stats["dist_mean"], stats["dist_std"]
        t30_mean, t30_std = stats["t30_mean"], stats["t30_std"]
        pred = {"direction": model_output["direction"],
                "area": model_output["area"] * area_std + area_mean,
                "distance": model_output["distance"] * distance_std + distance_mean,
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
    if isinstance(obj, torch.Tensor): return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):  return {k: recursive_to(v, device) for k,v in obj.items()}
    if isinstance(obj, list):  return [recursive_to(v, device) for v in obj]
    return obj

def main():
    # perf flags
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = load_config()

    distributed, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # W&B (main process only)
    if cfg["wandb"] and _is_main_process():
        if cfg["run_name"] is not None:
            wandb.init(project=cfg["proj"], name=cfg["run_name"], config=cfg, save_code=True, mode="online")
        else:
            wandb.init(project=cfg["proj"], config=cfg, save_code=True, mode="online")
    else:
        os.environ["WANDB_MODE"] = "disabled"

    # -------- Train loader --------
    train_ds = AudioRIRDataset(csv_audio=cfg["audio_csv_train"], base_dir=cfg["audio_base"],
                               csv_rir=cfg["rir_csv_train"], n_views=cfg["n_views"],
                               split=cfg["split"], batch_size=cfg["batch_size"])
    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    train_dl = DataLoader(train_ds,
                          batch_size=cfg["batch_size"],   # per-GPU
                          shuffle=(train_sampler is None),
                          sampler=train_sampler,
                          num_workers=4,
                          collate_fn=collate_fn,
                          pin_memory=True,
                          persistent_workers=False)

    train_stats = {
        "area_mean": train_ds.area_mean, "area_std": train_ds.area_std,
        "dist_mean": train_ds.dist_mean, "dist_std": train_ds.dist_std,
        "t30_mean":  train_ds.t30_mean,  "t30_std":  train_ds.t30_std,
    }

    # -------- Val loader (precomputed) --------
    val_root = cfg.get("val_precomp_root")
    val_csv  = cfg.get("val_index_csv") or (str(Path(val_root)/"val_precomputed.csv") if val_root else None)
    if not val_csv or not Path(val_csv).exists():
        raise SystemExit(f"[ERR] val_precomputed.csv が見つかりません: {val_csv}")
    val_ds = PrecomputedValDataset(index_csv=val_csv, rir_meta_csv=cfg["rir_csv_val"], root=val_root)
    val_dl = DataLoader(val_ds,
                        batch_size=cfg.get("val_batch_size", cfg["batch_size"]),
                        shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=True)

    # -------- Model / Optim --------
    model = DELSA(audio_encoder_cfg={}, text_encoder_cfg={}).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))

    use_amp = bool(cfg.get("use_amp", True))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    grad_accum = max(1, int(cfg.get("grad_accum", 1)))

    def forward_once(batch):
        audio = {k: torch.stack([d[k] for d in batch["audio"]]).to(device, non_blocking=True)
                 for k in ("i_act","i_rea","omni_48k")}
        texts  = batch["texts"]
        out = model(audio, texts)
        return out

    for ep in range(1, cfg["epochs"]+1):
        if train_sampler is not None:
            train_sampler.set_epoch(ep)
        model.train()
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_dl, 1):
            if batch is None: continue
            batch_data = {k: recursive_to(v, device) for k, v in batch.items() if k not in ["audio","texts"]}
            src_lb = batch["source_id"].reshape(-1).to(device, non_blocking=True)
            spa_lb = batch["space_id"].reshape(-1).to(device, non_blocking=True)

            # DDP no_sync for gradient accumulation (all but last micro-step)
            ddp_sync = ( (step % grad_accum) == 0 )
            sync_ctx = nullcontext()
            if distributed and not ddp_sync:
                sync_ctx = model.no_sync  # type: ignore

            with sync_ctx():
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = forward_once(batch)
                    a_spa  = F.normalize(out["audio_space_emb"],  dim=-1)
                    t_spa  = F.normalize(out["text_space_emb"],   dim=-1)
                    a_src  = F.normalize(out["audio_source_emb"], dim=-1)
                    t_src  = F.normalize(out["text_source_emb"],  dim=-1)
                    logit_s = out["logit_scale"]

                    loss_space  = sup_contrast(a_spa,  t_spa,  spa_lb, logit_s)
                    loss_source = sup_contrast(a_src, t_src, src_lb, logit_s)
                    phys_log, phys_loss = physical_loss(out, batch_data, isNorm=True, dataloader=train_dl)

                    loss = (loss_space + loss_source + phys_loss) / grad_accum

                scaler.scale(loss).backward()

            if ddp_sync:
                if cfg.get("max_grad_norm"):
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            if step % 10 == 0 and _is_main_process():
                print(f"Epoch {ep} Step {step}/{len(train_dl)}  space={loss_space:.4f}  src={loss_source:.4f}")
                if cfg["wandb"]:
                    wandb.log({
                        "loss/space": loss_space.item(),
                        "loss/source": loss_source.item(),
                        "loss/physical": phys_loss.item(),
                        "loss/dir": phys_log["loss_dir"],
                        "loss/distance": phys_log["loss_distance"],
                        "loss/area": phys_log["loss_area"],
                        "loss/reverb": phys_log["loss_reverb"],
                        "logit_scale": out["logit_scale"].item(),
                        "loss/mean": (loss_space + loss_source + phys_loss).item(),
                        "epoch": ep, "step": step + (ep-1)*len(train_dl)
                    })

        # -------- Validation (rank0 only) --------
        if _is_dist_initialized():
            dist.barrier()
        model.eval()
        if _is_main_process():
            val_losses = {"space":0.0,"source":0.0,"physical":0.0,"direction":0.0,"distance":0.0,"area":0.0,"reverb":0.0,"count":0}
            with torch.no_grad():
                for batch in val_dl:
                    if batch is None: continue
                    batch_data = {k: recursive_to(v, device) for k, v in batch.items() if k not in ["audio","texts"]}
                    audio = {k: torch.stack([d[k] for d in batch["audio"]]).to(device, non_blocking=True)
                             for k in ("i_act","i_rea","omni_48k")}
                    texts  = batch["texts"]
                    src_lb = batch["source_id"].reshape(-1).to(device, non_blocking=True)
                    spa_lb = batch["space_id"].reshape(-1).to(device, non_blocking=True)

                    with torch.cuda.amp.autocast(enabled=use_amp):
                        out = model(audio, texts)
                        a_spa  = F.normalize(out["audio_space_emb"],  dim=-1)
                        t_spa  = F.normalize(out["text_space_emb"],   dim=-1)
                        a_src  = F.normalize(out["audio_source_emb"], dim=-1)
                        t_src  = F.normalize(out["text_source_emb"],  dim=-1)
                        logit_s = out["logit_scale"]

                        l_sp  = sup_contrast(a_spa,  t_spa,  spa_lb, logit_s)
                        l_sr  = sup_contrast(a_src, t_src, src_lb, logit_s)
                        phys_log, l_phys = physical_loss(out, batch_data, isNorm=False, dataloader=val_dl, stats=train_stats)

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

            mets = eval_retrieval(model.module if isinstance(model, DDP) else model,
                                  val_dl, device, use_wandb=False, epoch=ep)

            if cfg["wandb"]:
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

            # Invariance Ratio logging (optional quick view; last batch stats)
            try:
                ir_a = invariance_ratio(a_spa, src_lb, spa_lb)
                ir_b = invariance_ratio(a_src, src_lb, spa_lb)
                ir_tspa = invariance_ratio(t_spa, src_lb, spa_lb)
                ir_tsrc = invariance_ratio(t_src, src_lb, spa_lb)
                if cfg["wandb"]:
                    wandb.log({
                        "IR/audio_space":  ir_a["IR_space"],
                        "IR/audio_source": ir_b["IR_source"],
                        "IR/text_space":   ir_tspa["IR_space"],
                        "IR/text_source":  ir_tsrc["IR_source"],
                        "IR/count_ss_ds":  ir_a["num_ss_ds"],
                        "IR/count_sd_ss":  ir_a["num_sd_ss"],
                        "epoch": ep,
                    })
            except Exception:
                pass

            # checkpoint (main only)
            ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True)
            tosave = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            torch.save({"model": tosave, "opt": opt.state_dict(), "epoch": ep},
                       ckpt_dir/f"ckpt_sup_ep{ep}.pt")
            print(f"[✓] Saved checkpoint for epoch {ep}")

        if _is_dist_initialized():
            dist.barrier()

    if _is_dist_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    try:
        main()
    finally:
        if wandb.run is not None and _is_main_process():
            wandb.finish()
        print("[✓] Finished training.")
