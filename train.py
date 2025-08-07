#!/usr/bin/env python3
# train_sup.py  ― supervised contrastive (source / space) 版
import torch 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import random
import wandb
import math
import sys
import yaml




# 容積の分類のloss
# 音源の分類のloss 犬とか
# adversarial loss 
    # sourceの埋め込みからはspaceの埋め込みを予測できないようにする。
    # 音源の分類は　spatialの埋め込みからはできないようにする。

# 3週間くらいで結果揃える
# ToDo: 正規化をする. 


random.seed(42)  # 再現性のため


#ToDO:Supervised Contrasive Learningを読む。https://proceedings.neurips.cc/paper_files/paper/2020/file/d89a66c7c80a29b1bdbab0f2a1a94af8-Paper.pdf?utm_source=chatgpt.com
# ────────────────────────── CLI ──────────────────────────

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
    }
    for k, v in defaults.items():
        cfg.setdefault(k, v)

    # 4️⃣  post‑process
    cfg["device"] = _select_device(cfg["device"])
    return cfg





# ───────── Supervised cross-modal contrastive los ─────────

def sup_contrast(
    a: torch.Tensor,
    b: torch.Tensor,
    labels: torch.Tensor,
    logit_scale: torch.Tensor,
    eps: float = 1e-8,
    *,
    symmetric: bool = True,      # True: Text↔Audio 両方向を平均
    exclude_diag: bool = False   # True: 分母から自己相似を除外 (SCL 型)
) -> torch.Tensor:
    """
    Cross‑modal Supervised Contrastive Loss (InfoNCE‑style).

    Parameters
    ----------
    a, b        : [B, D]  L2‑normalized以外は何でも可。内部で normalize する。
    labels      : [B]     同じ値を持つサンプルを正例集合とみなす。
    logit_scale : ()      log(temperature^{-1})。CLIP と同じ扱い。
    eps         : float   分母のゼロ除避け。
    symmetric   : bool    True  -> (Text→Audio + Audio→Text)/2
    exclude_diag: bool    True  -> 自己相似を分母から除外 (SCL 推奨)

    Returns
    -------
    loss : torch.Tensor  scalar
    """

    # 1. L2‑normalize
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)

    # 2. temperature (clamp to avoid overflow)
    scale = torch.clamp(logit_scale, max=math.log(1e2)).exp()

    # 3. logits  (Text→Audio)
    logits_t2a = torch.matmul(a, b.T) * scale           # [B, B]

    # optional: Audio→Text (reuse transpose for efficiency)
    logits_a2t = logits_t2a.T if symmetric else None

    def _directional_loss(logits: torch.Tensor) -> torch.Tensor:
        B = logits.size(0)

        # --- positive mask ---
        pos_mask = labels[:, None].eq(labels[None, :])          # [B, B] bool

        # --- numerical stability (detach!) ---
        max_sim, _ = logits.max(dim=1, keepdim=True)
        logits_stable = logits - max_sim.detach()               # stop‑grad on max

        # --- denominator ---
        if exclude_diag:
            diag_mask = torch.eye(B, dtype=torch.bool, device=logits.device)
        else:
            diag_mask = torch.zeros(B, B, dtype=torch.bool, device=logits.device)

        exp_sim = torch.exp(logits_stable) * (~diag_mask)
        denom = exp_sim.sum(dim=1, keepdim=True) + eps

        # --- log‑probabilities ---
        log_prob = logits_stable - denom.log()
        mean_log_pos = (log_prob * pos_mask.float()).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)

        # --- drop rows with no positives ---
        valid = pos_mask.any(dim=1)
        return -mean_log_pos[valid].mean()

    loss_t2a = _directional_loss(logits_t2a)
    if symmetric:
        loss_a2t = _directional_loss(logits_a2t)
        return 0.5 * (loss_t2a + loss_a2t)
    else:
        return loss_t2a
# ToDo;sup_contrastの公式実装見る：https://github.com/HobbitLong/SupContrast?utm_source=chatgpt.com


# ── 物理量の予測損失 ──
def physical_loss(model_output, batch_data,isNorm=True ,dataloader=None) -> torch.Tensor:
    """
    物理量の予測損失を計算する。
    pred_physics: {
        "direction": [B, 2]  # azimuth, elevation
        "area": [B, 1]
        "distance": [B, 1]
        "reverb": [B, 1] """
    if isNorm:
        pred_physics = {
            "direction": model_output["direction"],  
            "area": model_output["area"],
            "distance": model_output["distance"],
            "reverb": model_output["reverb"]}  
   
        true_physics = {
            "direction": batch_data["rir_meta"]["direction_vec"],  
            "area": batch_data["rir_meta"]["area_m2_norm"],
            "distance": batch_data["rir_meta"]["distance_norm"],
            "reverb": batch_data["rir_meta"]["t30_norm"]} 
        
    else: #生の値で評価するとき
        area_mean, area_std = dataloader.dataset.area_mean, dataloader.dataset.area_std
        distance_mean, distance_std = dataloader.dataset.dist_mean, dataloader.dataset.dist_std
        t30_mean, t30_std = dataloader.dataset.t30_mean, dataloader.dataset.t30_std
        pred_physics = {
            "direction": model_output["direction"],  
            "area": model_output["area"] * area_std + area_mean,
            "distance": model_output["distance"] * distance_std + distance_mean,
            "reverb": model_output["reverb"] * t30_std + t30_mean
        }
        true_physics = {
            "direction": batch_data["rir_meta"]["direction_vec"],
            "area": batch_data["rir_meta"]["area_m2"],
            "distance": batch_data["rir_meta"]["distance"],
            "reverb": batch_data["rir_meta"]["fullband_T30_ms"]}    
        
    # 方向の損失は、cosine similarityを使う。
    loss_dir = (-1* F.cosine_similarity(pred_physics["direction"], true_physics["direction"], dim=1)).mean()

    #mse 自体がえ平均をとるので、平均をとる必要はない。
    loss_area = F.mse_loss(pred_physics["area"].squeeze(-1), true_physics["area"])
    loss_distance= F.mse_loss(pred_physics["distance"].squeeze(-1), true_physics["distance"])
    loss_reverb = F.mse_loss(pred_physics["reverb"].squeeze(-1), true_physics["reverb"])

    total_physical_loss = loss_dir + loss_area + loss_distance + loss_reverb
    physical_losses = {
        "total_loss": total_physical_loss, "loss_dir": loss_dir.item(),
        "loss_distance": loss_distance.item(), "loss_area": loss_area.item(), "loss_reverb": loss_reverb.item()

    }
    return physical_losses, total_physical_loss
    
def recursive_to(obj, device):
    """Tensor だけを再帰的に device へ送る"""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: recursive_to(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [recursive_to(v, device) for v in obj]
    return obj          # int / float / str など
# ─────────────────────────── Main ─────────────────────────
def main():
    cfg = load_config()  # config.yaml を読み込む
    from dataset.audio_rir_dataset import AudioRIRDataset, collate_fn
    if cfg["wandb"]:
        wandb.init(project=cfg["proj"],
                   name = cfg["run_name"],
                   config = cfg, save_code =True,
                   mode="online")
    train_ds = AudioRIRDataset(
        csv_audio=cfg["audio_csv_train"],
        base_dir=cfg["audio_base"],
        csv_rir=cfg["rir_csv_train"],
        n_views=cfg["n_views"],
        split=cfg["split"],
        batch_size=cfg["batch_size"]
    )
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"],
                    shuffle=True, num_workers=4,
                    collate_fn=collate_fn, pin_memory=False)
    
    # val 用  ★追加★
    val_ds = AudioRIRDataset(
        csv_audio=cfg["audio_csv_val"], base_dir=cfg["audio_base"],
        csv_rir=cfg["rir_csv_val"], split="val",             # ここだけ split=val
        n_views=cfg["n_views"],
        batch_size=cfg["batch_size"])
    val_dl = DataLoader(val_ds,  batch_size=cfg["batch_size"],
                        shuffle=False, num_workers=4,
                        collate_fn=collate_fn, pin_memory=False)
    from model.delsa_model import DELSA      # アップロード済みモデル
    model = DELSA(audio_encoder_cfg={}, text_encoder_cfg={}).to(cfg["device"])
    opt   = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

    for ep in range(1, cfg["epochs"]+1):
        model.train()
        for step, batch in enumerate(train_dl, 1):
            if batch is None:
                continue
            # ── データをデバイスに転送 ──
            #batch_data = {k: v.to(cfg["device"]) for k, v in batch.items() if k not in ["audio", "texts"]}
            batch_data = {k: recursive_to(v, cfg["device"])
              for k, v in batch.items() if k not in ["audio","texts"]}
            # ----------- flatten ------------

            audio_dict = {k: torch.stack([d[k] for d in batch["audio"]]).to(cfg["device"])
              for k in ("i_act","i_rea","omni_48k")}
            texts  = batch["texts"]
            #print(texts)
            src_lb = batch["source_id"].reshape(-1).to(cfg["device"])  # [B']
            meta = batch["rir_meta"]  # RIRメタデータ
            #print(meta)
            spa_lb = batch["space_id"].reshape(-1).to(cfg["device"])
            #print(spa_lb)
            # ----------- forward ------------
            model_out = model(audio_dict, texts)   # expect 4 embeddings + logit_scale

            a_s  = F.normalize(model_out["audio_space_emb"],  dim=-1)
            t_s  = F.normalize(model_out["text_space_emb"],   dim=-1)
            a_src= F.normalize(model_out["audio_source_emb"], dim=-1)
            t_src= F.normalize(model_out["text_source_emb"],  dim=-1)



            T = model_out["logit_scale"].exp()     # 温度
            loss_space  = sup_contrast(a_s,  t_s,  spa_lb, T)
            loss_source = sup_contrast(a_src,t_src,src_lb, T)

            physical_losses, total_physical_loss = physical_loss(model_out, batch_data, isNorm=True, dataloader=train_dl)

            loss = (loss_space + loss_source + total_physical_loss)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 10 == 0:
                print(f"Epoch {ep} Step {step}/{len(train_dl)}  "
                      f"space={loss_space:.4f}  src={loss_source:.4f}")
                if cfg["wandb"]:
                    wandb.log({
                        "loss/space": loss_space.item(),
                        "loss/source": loss_source.item(),
                        "loss/physical": total_physical_loss.item(),
                        "loss/dir": physical_losses["loss_dir"],
                        "loss/distance": physical_losses["loss_distance"],
                        "loss/area": physical_losses["loss_area"],  
                        "loss/reverb": physical_losses["loss_reverb"],
                        "logit_scale": model_out["logit_scale"].item(),
                        "loss/mean": loss.item(),
                        "epoch": ep,
                        "step": step + (ep-1)*len(train_dl)
                    })
        # ---------- Validation ----------  ★追加ブロック★
        model.eval()
        val_metrics = {"space": 0.0, "source": 0.0,
                    "physical": 0.0, "count": 0}
        with torch.no_grad():
            for batch in val_dl:
                if batch is None:  # collate_fn が None を返すケース対策
                    continue
                # --- デバイス転送／forward は train ループと同じ ---
                batch_data = {k: v.to(cfg["device"]) for k, v in batch.items()
                            if k not in ["audio","texts"]}
                audio_dict = {k: torch.stack([d[k] for d in batch["audio"]]).to(args.device)
                            for k in ("i_act","i_rea","omni_48k")}
                texts  = batch["texts"]
                src_lb = batch["source_id"].reshape(-1).to(cfg["device"])
                spa_lb = batch["space_id"].reshape(-1).to(cfg["device"])

                model_out = model(audio_dict, texts)

                a_s  = F.normalize(model_out["audio_space_emb"],  dim=-1)
                t_s  = F.normalize(model_out["text_space_emb"],   dim=-1)
                a_src= F.normalize(model_out["audio_source_emb"], dim=-1)
                t_src= F.normalize(model_out["text_source_emb"],  dim=-1)
                T = model_out["logit_scale"].exp()

                loss_space  = sup_contrast(a_s,  t_s,  spa_lb, T)
                loss_source = sup_contrast(a_src,t_src,src_lb, T)
                physical_losses, total_physical_loss = physical_loss(model_out, batch_data, isNorm=False, dataloader=val_dl)

                val_metrics["space"]    += loss_space.item()
                val_metrics["source"]   += loss_source.item()
                val_metrics["physical"] += total_physical_loss.item()
                val_metrics["direction"] += physical_losses["loss_dir"].item()
                val_metrics["distance"] += physical_losses["loss_distance"].item()
                val_metrics["area"]     += physical_losses["loss_area"].item()
                val_metrics["reverb"]   += physical_losses["loss_reverb"].item()
                val_metrics["count"]    += 1

        # ---- 平均を計算 ----
        n = val_metrics["count"]
        val_space    = val_metrics["space"]    / n
        val_source   = val_metrics["source"]   / n
        val_phys     = val_metrics["physical"] / n
        val_direction = val_metrics["direction"] / n
        val_distance = val_metrics["distance"] / n
        val_area     = val_metrics["area"]     / n
        val_reverb   = val_metrics["reverb"]   / n
        val_mean     = (val_space + val_source + val_phys)

        print(f"Epoch {ep}  [VAL] space={val_space:.4f} "
            f"src={val_source:.4f} phys={val_phys:.4f}")

        if cfg["wandb"]:
            wandb.log({
                "val/loss_space":    val_space,
                "val/loss_source":   val_source,
                "val/loss_physical": val_phys,
                "val/loss_direction": val_direction,
                "val/loss_distance": val_distance,
                "val/loss_area":     val_area,  
                "val/loss_reverb":   val_reverb,
                "logit_scale":       model_out["logit_scale"].item(),
                "val/loss_mean":     val_mean,
                "epoch": ep
            })

        # -------- checkpoint -------------
        ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True)
        torch.save({"model": model.state_dict(),
                    "opt":   opt.state_dict(),
                    "epoch": ep},
                   ckpt_dir/f"ckpt_sup_ep{ep}.pt")
        print(f"[✓] Saved checkpoint for epoch {ep}")


if __name__ == "__main__":
    try:
        main()
    finally:
        if wandb.run is not None:
            wandb.finish()