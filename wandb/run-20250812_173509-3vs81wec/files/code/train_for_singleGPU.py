#!/usr/bin/env python3
# train.py  ― supervised contrastive (source / space) + per-epoch retrieval eval
from utils.metrics import invariance_ratio
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import random, wandb, math, sys, yaml
from utils.metrics import invariance_ratio, leakage_probe_acc, linear_probe_regression

from dataset.audio_rir_dataset_old import AudioRIRDataset, collate_fn   # 既存
from dataset.precomputed_val_dataset import PrecomputedValDataset   # 追加①
from utils.metrics import cosine_sim, recall_at_k, recall_at_k_multi                   # 追加②
from model.delsa_model import DELSA                                # ← 修正

random.seed(42)

def _select_device(raw: str |None) -> str:
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
        # ↓ 追加（前計算Valの場所）
        "val_precomp_root": None,   # 例: "Spatial_AudioCaps/takamichi09/for_delsa_spatialAudio"
        "val_index_csv":    None,   # 例: 上記/root/val_precomputed.csv（未指定なら root を使う）
        "val_batch_size":   16,
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

def physical_loss(model_output, batch_data, isNorm=True, dataloader=None,stats = None):
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
    if isinstance(obj, torch.Tensor): return obj.to(device)
    if isinstance(obj, dict):  return {k: recursive_to(v, device) for k,v in obj.items()}
    if isinstance(obj, list):  return [recursive_to(v, device) for v in obj]
    return obj
@torch.no_grad()
def eval_retrieval(model, loader, device, use_wandb=False, epoch=None):
    """
    Retrieval (multi-positive; no pooling)
      On-task : SRC(text_source<->audio_source), SPA(text_space<->audio_space)
      Off-task: X-SRC(text_source<->audio_space; source一致),
                X-SPA(text_space<->audio_source; space一致)
      さらに Chance@K と Excess@K (= R@K - Chance@K) を出す
    """
    model.eval()
    bufs = {"t_spa":[], "a_spa":[], "t_src":[], "a_src":[], "ids_src":[], "ids_spa":[]}
    for batch in loader:
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
        bufs[k] = torch.cat(bufs[k], dim=0)

    a_src, t_src, ids_src = bufs["a_src"], bufs["t_src"], bufs["ids_src"]
    a_spa, t_spa, ids_spa = bufs["a_spa"], bufs["t_spa"], bufs["ids_spa"]

    def _chance_at_k(q_ids, g_ids, ks=(1,5,10)):
        Ng = int(g_ids.numel())
        P = torch.stack([(g_ids == q).sum() for q in q_ids]).float()  # [Nq]
        out = {}
        for k in ks:
            k = min(k, Ng)
            if k <= 0:
                out[f"Chance@{k}"] = 0.0
                continue
            j = torch.arange(k, dtype=torch.float32)
            denom_log = torch.sum(torch.log((Ng - j)))                 # スカラー
            num_log   = torch.sum(torch.log((Ng - P.unsqueeze(1) - j).clamp(min=1.0)), dim=1)  # [Nq]
            prob_no   = torch.exp(num_log - denom_log).clamp(0.0, 1.0) # [Nq]
            out[f"Chance@{k}"] = (1.0 - prob_no).mean().item()
        return out

    def _eval_block(prefix, q_emb, g_emb, q_ids, g_ids):
        S = cosine_sim(q_emb, g_emb)  # [Nq, Ng]
        R = recall_at_k_multi(S, q_ids, g_ids, ks=(1,5,10))
        C = _chance_at_k(q_ids, g_ids, ks=(1,5,10))
        mets = {f"{prefix}/T2A/{k}": v for k, v in R.items()}
        mets.update({f"{prefix}/T2A/{k}": v for k, v in C.items()})
        for k in (1,5,10):
            mets[f"{prefix}/T2A/Excess@{k}"] = R[f"R@{k}"] - C[f"Chance@{k}"]

        Rr = recall_at_k_multi(S.T, g_ids, q_ids, ks=(1,5,10))
        Cr = _chance_at_k(g_ids, q_ids, ks=(1,5,10))
        mets.update({f"{prefix}/A2T/{k}": v for k, v in Rr.items()})
        mets.update({f"{prefix}/A2T/{k}": v for k, v in Cr.items()})
        for k in (1,5,10):
            mets[f"{prefix}/A2T/Excess@{k}"] = Rr[f"R@{k}"] - Cr[f"Chance@{k}"]
        return mets

    mets = {}
    # On-task
    mets.update(_eval_block("SRC", t_src, a_src, ids_src, ids_src))
    mets.update(_eval_block("SPA", t_spa, a_spa, ids_spa, ids_spa))
    # Off-task（低いほど良い / Excess≈0 が理想）
    mets.update(_eval_block("X-SRC", t_src, a_spa, ids_src, ids_src))
    mets.update(_eval_block("X-SPA", t_spa, a_src, ids_spa, ids_spa))

    if use_wandb:
        wandb.log({"epoch": epoch, **mets})
    return mets


def main():
    cfg = load_config()
    if cfg["wandb"] and cfg["run_name"] is not None:
        wandb.init(project=cfg["proj"], name=cfg["run_name"], config=cfg, save_code=True, mode="online")
    elif cfg["wandb"]:
        wandb.init(project=cfg["proj"], config=cfg, save_code=True, mode="online")
    # -------- Train loader (従来通り) --------
    train_ds = AudioRIRDataset(csv_audio=cfg["audio_csv_train"], base_dir=cfg["audio_base"],
                               csv_rir=cfg["rir_csv_train"], n_views=cfg["n_views"],
                               split=cfg["split"], batch_size=cfg["batch_size"])
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=4,
                          collate_fn=collate_fn, pin_memory=False)
    train_stats = {
        "area_mean": train_ds.area_mean, "area_std": train_ds.area_std,
        "dist_mean": train_ds.dist_mean, "dist_std": train_ds.dist_std,
        "t30_mean":  train_ds.t30_mean,  "t30_std":  train_ds.t30_std,
    }
    # -------- Val loader (前計算を読む) --------
    val_root = cfg.get("val_precomp_root")
    val_csv  = cfg.get("val_index_csv") or (str(Path(val_root)/"val_precomputed.csv") if val_root else None)
    if not val_csv or not Path(val_csv).exists():
        raise SystemExit(f"[ERR] val_precomputed.csv が見つかりません: {val_csv}")
    val_ds = PrecomputedValDataset(index_csv=val_csv, rir_meta_csv=cfg["rir_csv_val"], root=val_root)
    val_dl = DataLoader(val_ds, batch_size=cfg.get("val_batch_size", cfg["batch_size"]),
                        shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=False)

    # -------- Model / Optim --------
    model = DELSA(audio_encoder_cfg={}, text_encoder_cfg={}).to(cfg["device"])
    opt   = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    for ep in range(1, cfg["epochs"]+1):
        model.train()
        for step, batch in enumerate(train_dl, 1):
            if batch is None: continue
            batch_data = {k: recursive_to(v, cfg["device"]) for k, v in batch.items() if k not in ["audio","texts"]}
            audio = {k: torch.stack([d[k] for d in batch["audio"]]).to(cfg["device"]) for k in ("i_act","i_rea","omni_48k")}
            texts  = batch["texts"]
            src_lb = batch["source_id"].reshape(-1).to(cfg["device"])
            spa_lb = batch["space_id"].reshape(-1).to(cfg["device"])

            out = model(audio, texts)
            a_spa  = F.normalize(out["audio_space_emb"],  dim=-1)
            t_spa  = F.normalize(out["text_space_emb"],   dim=-1)
            a_src  = F.normalize(out["audio_source_emb"], dim=-1)
            t_src  = F.normalize(out["text_source_emb"],  dim=-1)
            logit_s = out["logit_scale"]

            loss_space  = sup_contrast(a_spa,  t_spa,  spa_lb, logit_s)
            loss_source = sup_contrast(a_src, t_src, src_lb, logit_s)
            phys_log, phys_loss = physical_loss(out, batch_data, isNorm=True, dataloader=train_dl)

            loss = loss_space + loss_source + phys_loss
            opt.zero_grad(); loss.backward(); opt.step()

            if step % 10 == 0:
                print(f"Epoch {ep} Step {step}/{len(train_dl)}  space={loss_space:.4f}  src={loss_source:.4f}")
                if cfg["wandb"]:
                    wandb.log({"loss/space": loss_space.item(), "loss/source": loss_source.item(),
                               "loss/physical": phys_loss.item(), "loss/dir": phys_log["loss_dir"],
                               "loss/distance": phys_log["loss_distance"], "loss/area": phys_log["loss_area"],
                               "loss/reverb": phys_log["loss_reverb"], "logit_scale": out["logit_scale"].item(),
                               "loss/mean": loss.item(), "epoch": ep, "step": step + (ep-1)*len(train_dl)})

        # -------- Validation (Retrieval + 物理lossの平均) --------
        model.eval()
        val_losses = {"space":0.0,"source":0.0,"physical":0.0,"direction":0.0,"distance":0.0,"area":0.0,"reverb":0.0,"count":0}

        with torch.no_grad():
            for batch in val_dl:
                if batch is None: continue
                batch_data = {k: recursive_to(v, cfg["device"]) for k, v in batch.items() if k not in ["audio","texts"]}
                audio = {k: torch.stack([d[k] for d in batch["audio"]]).to(cfg["device"]) for k in ("i_act","i_rea","omni_48k")}
                texts  = batch["texts"]
                src_lb = batch["source_id"].reshape(-1).to(cfg["device"])
                spa_lb = batch["space_id"].reshape(-1).to(cfg["device"])

                out = model(audio, texts)
                a_spa  = F.normalize(out["audio_space_emb"],  dim=-1)
                t_spa  = F.normalize(out["text_space_emb"],   dim=-1)
                a_src = F.normalize(out["audio_source_emb"], dim=-1)
                t_src = F.normalize(out["text_source_emb"],  dim=-1)
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
        
        mets = eval_retrieval(model, val_dl, cfg["device"], use_wandb=False, epoch=ep)

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
                
        # audio側の分離
        ir_a = invariance_ratio(a_spa, src_lb, spa_lb)   # emb=audio_space
        # ここで ir_a["IR_space"] を見る（大きいほど良い）
        ir_b = invariance_ratio(a_src, src_lb, spa_lb)   # emb=audio_source
        # ここで ir_b["IR_source"] を見る（大きいほど良い）

        # （任意）text側も見たい場合
        ir_tspa = invariance_ratio(t_spa, src_lb, spa_lb)
        ir_tsrc = invariance_ratio(t_src, src_lb, spa_lb)

        # W&Bへ
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
        # -------- checkpoint -------------
        ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep},
                   ckpt_dir/f"ckpt_sup_ep{ep}.pt")
        print(f"[✓] Saved checkpoint for epoch {ep}")

if __name__ == "__main__":
    try:
        main()
    finally:
        if wandb.run is not None:
            wandb.finish()
        print("[✓] Finished training.")