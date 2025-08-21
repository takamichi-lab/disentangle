
#!/usr/bin/env python3
#(.venv) takamichi-lab-pc09@takamichi-lab-pc09:~/DELSA$ python3 evaluate_checkpoint.py   --ckpt Spatial_AudioCaps/takamichi09/checkpoints_delsa/ckpt_sup_ep9.pt   --wandb   --eval_amp   --exclude_diag

# evaluate_checkpoint.py — TEST-ONLY evaluation (no SupCon)
# - Computes: retrieval metrics, physical losses (norm or real scale), invariance ratios
# - GPU-accelerated (retrieval/IR on GPU), tqdm progress, optional W&B logging
# - Uses config.yaml keys: test_precomp_root, test_index_csv, rir_csv_test, stats_path, val_batch_size, device
# - W&B project: "delsa-sup-contrast_testset" (enable with --wandb)

import argparse, json, os, sys, yaml
from pathlib import Path
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset.precomputed_val_dataset import PrecomputedValDataset
from dataset.audio_rir_dataset import collate_fn
from model.delsa_model import DELSA
from utils.metrics import eval_retrieval, invariance_ratio

import wandb

def _pick_sd(obj):
    if isinstance(obj, dict):
        for k in ("model_state_dict", "state_dict", "model"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj

def _strip_module(sd: dict) -> dict:
    return {(k.replace("module.", "", 1) if k.startswith("module.") else k): v for k, v in sd.items()}

def recursive_to(obj, device):
    if isinstance(obj, torch.Tensor): return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):  return {k: recursive_to(v, device) for k,v in obj.items()}
    if isinstance(obj, list):  return [recursive_to(v, device) for v in obj]
    return obj

@torch.no_grad()
def _forward_batch(model, batch, device, use_amp=False):
    audio = {
        "i_act": torch.stack([d["i_act"] for d in batch["audio"]]).to(device, non_blocking=True),
        "i_rea": torch.stack([d["i_rea"] for d in batch["audio"]]).to(device, non_blocking=True),
        "omni_48k": torch.stack([d["omni_48k"] for d in batch["audio"]]).to(device, non_blocking=True),
    }
    texts = batch["texts"]
    with torch.autocast(device_type=("cuda" if device=="cuda" else "cpu"), enabled=use_amp):
        out = model(audio, texts)
        a_spa  = F.normalize(out["audio_space_emb"], dim=-1)
        t_spa  = F.normalize(out["text_space_emb"], dim=-1)
        a_src  = F.normalize(out["audio_source_emb"], dim=-1)
        t_src  = F.normalize(out["text_source_emb"], dim=-1)
    return out, a_spa, t_spa, a_src, t_src

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to checkpoint .pt")
    p.add_argument("--index_csv", type=str, default=None)
    p.add_argument("--rir_meta_csv", type=str, default=None)
    p.add_argument("--root", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", type=str, default=None, choices=["auto","cpu","cuda"])
    p.add_argument("--stats_path", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--wandb", action="store_true", help="Log to W&B (project=delsa-sup-contrast_testset)")
    p.add_argument("--eval_amp", action="store_true", help="Enable autocast during forward (evaluation)")
    return p

def _select_device(raw: str|None):
    if raw is None or raw=="auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return raw

def load_config(path):
    try:
        with open(path,"r",encoding="utf-8") as f: return yaml.safe_load(f) or {}
    except FileNotFoundError: return {}

def flatten_ir(prefix, d):
    return {f"{prefix}/{k}": v for k,v in d.items()}

def main():
    # Optional CUDA speed-ups
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    args = build_parser().parse_args()
    cfg = load_config(args.config)
    root = args.root or cfg.get("test_precomp_root")
    index_csv = args.index_csv or cfg.get("test_index_csv") or (str(Path(root)/"test_precomputed.csv") if root else None)
    rir_meta_csv = args.rir_meta_csv or cfg.get("rir_csv_test")
    stats_path = args.stats_path or cfg.get("stats_path")
    batch_size = args.batch_size or cfg.get("val_batch_size",16)
    device = _select_device(args.device or cfg.get("device","auto"))

    if not index_csv or not os.path.exists(index_csv):
        raise SystemExit(f"[ERR] test_precomputed.csv not found: {index_csv}")
    if not rir_meta_csv or not os.path.exists(rir_meta_csv):
        raise SystemExit(f"[ERR] rir_csv_test not found: {rir_meta_csv}")

    ds = PrecomputedValDataset(index_csv=index_csv, rir_meta_csv=rir_meta_csv, root=root)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=args.num_workers,
                    collate_fn=collate_fn, pin_memory=(device=="cuda"), persistent_workers=False, prefetch_factor=2)

    # Model
    model = DELSA(audio_encoder_cfg={}, text_encoder_cfg={}).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    sd = _strip_module(_pick_sd(ckpt))
    model.load_state_dict(sd, strict=False)
    model.eval()

    # Stats (optional: for real-scale physical metrics)
    stats = None
    if stats_path and os.path.isfile(stats_path):
        s = torch.load(stats_path,map_location="cpu")
        stats = {
            "area_mean": s["area_m2"]["mean"], "area_std": s["area_m2"]["std"],
            "dist_mean": s["distance"]["mean"], "dist_std": s["distance"]["std"],
            "t30_mean":  s["fullband_T30_ms"]["mean"], "t30_std": s["fullband_T30_ms"]["std"],
        }

    if args.wandb:
        run_name = f"eval_{Path(args.ckpt).stem}"
        wandb.init(project="delsa-sup-contrast_testset", name=run_name, config=cfg)

    # Accumulators
    val_losses = {"physical":0.0,"direction":0.0,"distance":0.0,"area":0.0,"reverb":0.0,"count":0}
    buf = {"a_spa":[],"a_src":[],"t_spa":[],"t_src":[],"src_lb":[],"spa_lb":[]}

    @torch.no_grad()
    def physical_loss_eval(out, batch_data):
        if stats is None:
            pred = {"direction": out["direction"],"area": out["area"],"distance": out["distance"],"reverb": out["reverb"]}
            true = {"direction": batch_data["rir_meta"]["direction_vec"],
                    "area": batch_data["rir_meta"]["area_m2_norm"],
                    "distance": batch_data["rir_meta"]["distance_norm"],
                    "reverb": batch_data["rir_meta"]["t30_norm"]}
        else:
            pred = {"direction": out["direction"],
                    "area": out["area"]*stats["area_std"]+stats["area_mean"],
                    "distance": out["distance"]*stats["dist_std"]+stats["dist_mean"],
                    "reverb": out["reverb"]*stats["t30_std"]+stats["t30_mean"]}
            true = {"direction": batch_data["rir_meta"]["direction_vec"],
                    "area": batch_data["rir_meta"]["area_m2"],
                    "distance": batch_data["rir_meta"]["distance"],
                    "reverb": batch_data["rir_meta"]["fullband_T30_ms"]}
        loss_dir = (-1*F.cosine_similarity(pred["direction"],true["direction"],dim=1)).mean()
        loss_area= F.mse_loss(pred["area"].squeeze(-1),true["area"])
        loss_dist= F.mse_loss(pred["distance"].squeeze(-1),true["distance"])
        loss_rev = F.mse_loss(pred["reverb"].squeeze(-1),true["reverb"])
        total= loss_dir+loss_area+loss_dist+loss_rev
        return {"loss_dir":loss_dir.item(),"loss_area":loss_area.item(),
                "loss_distance":loss_dist.item(),"loss_reverb":loss_rev.item()}, total

    # Iterate
    for batch in tqdm(dl, desc="Evaluating", unit="batch", dynamic_ncols=True):
        if batch is None: continue
        batch_data = recursive_to({k:v for k,v in batch.items() if k not in ["audio","texts"]}, device)
        out,a_spa,t_spa,a_src,t_src = _forward_batch(model,batch,device, use_amp=args.eval_amp)

        phys_log,l_phys=physical_loss_eval(out,batch_data)
        val_losses["physical"]+=l_phys.item()
        val_losses["direction"]+=phys_log["loss_dir"]; val_losses["distance"]+=phys_log["loss_distance"]
        val_losses["area"]+=phys_log["loss_area"]; val_losses["reverb"]+=phys_log["loss_reverb"]; val_losses["count"]+=1

        # Keep embeddings on GPU for fast retrieval/IR
        buf["a_spa"].append(a_spa.detach().cpu())
        buf["a_src"].append(a_src.detach().cpu())
        buf["t_spa"].append(t_spa.detach().cpu())
        buf["t_src"].append(t_src.detach().cpu())
        buf["src_lb"].append(batch["source_id"].reshape(-1).to(device, non_blocking=True))
        buf["spa_lb"].append(batch["space_id"].reshape(-1).to(device, non_blocking=True))

    n = max(1,val_losses["count"])
    losses_mean={
        "loss/physical":val_losses["physical"]/n,
        "loss/direction":val_losses["direction"]/n,
        "loss/distance":val_losses["distance"]/n,
        "loss/area":val_losses["area"]/n,
        "loss/reverb":val_losses["reverb"]/n,
    }

    # ===== Retrieval / IR on GPU =====
    A_SPA=torch.cat(buf["a_spa"],dim=0).cpu()
    A_SRC=torch.cat(buf["a_src"],dim=0).cpu()
    T_SPA=torch.cat(buf["t_spa"],dim=0).cpu()
    T_SRC=torch.cat(buf["t_src"],dim=0).cpu()
    SRC_LB=torch.cat(buf["src_lb"],dim=0).cpu()
    SPA_LB=torch.cat(buf["spa_lb"],dim=0).cpu()

    retrieval=eval_retrieval(A_SPA,A_SRC,T_SPA,T_SRC,SRC_LB,SPA_LB,device=device,use_wandb=False,epoch=None)
    ir_space=invariance_ratio(A_SPA,SRC_LB,SPA_LB)
    ir_source=invariance_ratio(A_SRC,SRC_LB,SPA_LB)
    ir_tspa=invariance_ratio(T_SPA,SRC_LB,SPA_LB)
    ir_tsrc=invariance_ratio(T_SRC,SRC_LB,SPA_LB)

    summary={**{k: float(v) for k,v in losses_mean.items()},
             **{f"retrieval/{k}": (float(v) if isinstance(v,(int,float)) else v) for k,v in retrieval.items()},
             **flatten_ir("IR/audio_space", {k:(float(v) if isinstance(v,(int,float)) else v) for k,v in ir_space.items()}),
             **flatten_ir("IR/audio_source",{k:(float(v) if isinstance(v,(int,float)) else v) for k,v in ir_source.items()}),
             **flatten_ir("IR/text_space",  {k:(float(v) if isinstance(v,(int,float)) else v) for k,v in ir_tspa.items()}),
             **flatten_ir("IR/text_source", {k:(float(v) if isinstance(v,(int,float)) else v) for k,v in ir_tsrc.items()}),
             "count/batches": int(n),
             "count/items": int(A_SPA.size(0))}

    print("\\n=== Evaluation (test) ===")
    for k,v in losses_mean.items(): print(f"{k}: {float(v):.6f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True,exist_ok=True)
        with open(args.out,"w",encoding="utf-8") as f: json.dump(summary,f,indent=2,ensure_ascii=False)
        print(f"[✓] wrote metrics to {args.out}")

    if args.wandb:
        wandb.log(summary)
        wandb.finish()

if __name__=="__main__":
    main()
