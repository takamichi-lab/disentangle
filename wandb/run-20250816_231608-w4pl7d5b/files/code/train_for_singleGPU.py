#!/usr/bin/env python3
# train_for_singleGPU.py  ― supervised contrastive (source / space) + per-epoch retrieval eval
from utils.metrics import invariance_ratio
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import random, wandb, math, sys, yaml
from utils.metrics import invariance_ratio

from dataset.audio_rir_dataset import AudioRIRDataset, collate_fn   # 既存
from dataset.precomputed_val_dataset import PrecomputedValDataset   # 追加①
from utils.metrics import cosine_sim, recall_at_k, recall_at_k_multi, eval_retrieval              # 追加②
from model.delsa_model import DELSA                                # ← 修正

random.seed(42)
import torchaudio, torch
from functools import lru_cache
from dataset.audio_rir_dataset import foa_to_iv, FOA_SR, IV_SR  # 既存実装を再利用:contentReference[oaicite:15]{index=15}

def _aformat_to_foa(wet):  # wet: [4,T] A-format -> FOA (W,Y,Z,X)
    m0, m1, m2, m3 = wet[0], wet[1], wet[2], wet[3]
    W = (m0+m1+m2+m3)/2; X = (m0+m1-m2-m3)/2
    Y = (m0-m1+m2-m3)/2; Z = (m0-m1-m2+m3)/2
    return torch.stack([W, Y, Z, X], dim=0)

@lru_cache(maxsize=4096)
def _load_rir_cpu_cached(path: str):
    rir, sr = torchaudio.load(path)  # [4,Tr] A-format
    return rir, sr

def _fft_convolve_batch_gpu(dry_b, rir_b):  # [B,1,T],[B,4,Tr] -> [B,4,T]
    if dry_b.size(1) == 1:
        dry_b = dry_b.repeat(1,4,1)          # mono -> 4ch
    n = dry_b.shape[-1] + rir_b.shape[-1] - 1
    n_fft = 1 << (n - 1).bit_length()
    D = torch.fft.rfft(dry_b, n_fft)         # CUDA対応のrFFT :contentReference[oaicite:16]{index=16}
    R = torch.fft.rfft(rir_b, n_fft)
    y = torch.fft.irfft(D * R, n_fft)[..., :n]
    return y[..., :dry_b.shape[-1]]

def _build_audio_from_defer(audio_list, device):
    """Datasetから {dry, rir_path} を受け取り、GPUで畳み込み→FOA→16k→foa_to_iv までを実行。
       戻り値は {"i_act","i_rea","omni_48k"} のテンソル（学習コードは無変更でOK）。"""
    # 1) dry をまとめて GPU 転送
    drys = torch.stack([a["dry"].squeeze(0) for a in audio_list]).unsqueeze(1).to(device, non_blocking=True)  # [B,1,T]

    # 2) RIR をCPUキャッシュ→必要なら48kにresample→GPU転送
    rirs_cpu = []
    for a in audio_list:
        rir, sr = _load_rir_cpu_cached(a["rir_path"])
        rirs_cpu.append(rir if sr == FOA_SR else torchaudio.functional.resample(rir, sr, FOA_SR))
    rirs = torch.stack(rirs_cpu).to(device, non_blocking=True)  # [B,4,Tr]

    # 3) GPUでFFT畳み込み（A-format）→ FOA 変換
    wet_a = _fft_convolve_batch_gpu(drys, rirs)                 # [B,4,T]
    foa   = torch.stack([_aformat_to_foa(w) for w in wet_a], dim=0)  # [B,4,T]
    omni_48k = foa[:, 0, :]

    # 4) 16kへリサンプル（環境により CPU 実装のため安全に .cpu() へ）
    foa_16k = torchaudio.functional.resample(foa.detach().cpu(), orig_freq=FOA_SR, new_freq=IV_SR)

    # 5) foa_to_iv（deviceに従ってCUDA/CPUで実行可。ここではCPUでも十分高速）:contentReference[oaicite:17]{index=17}
    i_act_list, i_rea_list = [], []
    for b in range(foa_16k.size(0)):
        i_act, i_rea = foa_to_iv(foa_16k[b].unsqueeze(0))  # (1,3,F,Tfrm)
        i_act_list.append(i_act.squeeze(0)); i_rea_list.append(i_rea.squeeze(0))
    i_act = torch.stack(i_act_list, dim=0)
    i_rea = torch.stack(i_rea_list, dim=0)

    return {"i_act": i_act.to(device, non_blocking=True),
            "i_rea": i_rea.to(device, non_blocking=True),
            "omni_48k": omni_48k}  # omniは48kでそのままdevice上

def _is_defer_format(audio_list):
    return len(audio_list) > 0 and ("dry" in audio_list[0] and "rir_path" in audio_list[0])
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
    """
    Traditional Supervised Contrastive (Khosla et al., 2020) に準拠した2塔版。
    - 同ラベルを正例。2塔なので a のアンカーに対して b 側の同ラベル全てが正例。
    - exclude_diag=True のときは、対角 (i,i) を【分子・分母の両方】から外す（完全一貫）。
      False のときは対角を分子・分母の両方に含める（同一インスタンス正例を含める）。
    """
    a = F.normalize(a, dim=1); b = F.normalize(b, dim=1)

    # CLIP式のlogスケールを渡しているならこちらを使う：
    # scale = torch.clamp(logit_scale, max=math.log(1e2)).exp()
    # そうでなければ logit_scale は線形スカラーとして渡す
    scale = logit_scale

    logits_t2a = (a @ b.T) * scale
    logits_a2t = logits_t2a.T if symmetric else None

    labels = labels.view(-1)

    def _dir_loss(logits):
        B = logits.size(0)

        # 正例マスク（同ラベル）。2塔なので「自分自身のベクトル」は存在しない。
        pos_mask = labels[:, None].eq(labels[None, :]).float()

        # 数値安定化
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        # 分母マスク：exclude_diag によって対角を分子・分母で一貫処理
        if exclude_diag:
            pos_mask = pos_mask.clone()
            pos_mask.fill_diagonal_(0.0)  # 分子からも対角を除外（完全一貫）
            denom_mask = (~torch.eye(B, dtype=torch.bool, device=logits.device)).float()
        else:
            denom_mask = torch.ones_like(pos_mask)

        exp_logits = torch.exp(logits) * denom_mask
        denom = exp_logits.sum(dim=1, keepdim=True) + eps
        log_prob = logits - denom.log()

        pos_counts = pos_mask.sum(dim=1, keepdim=True)
        # アンカーごとの正例平均（SupConの 1/|P(i)| Σ_{p∈P(i)} log p を実装）
        mean_log_pos = (pos_mask * log_prob).sum(dim=1, keepdim=True) / pos_counts.clamp(min=1.0)

        valid = (pos_counts.squeeze(1) > 0)
        if valid.any():
            return -(mean_log_pos[valid]).mean()
        # 極端にクラスが偏って正例ゼロ行しか無い場合のフォールバック
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

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
            audio_list = batch["audio"]

            if _is_defer_format(audio_list):
                audio = _build_audio_from_defer(audio_list, cfg["device"]) 
            else:
                audio = {k: torch.stack([d[k] for d in batch["audio"]]).to(cfg["device"]) for k in ("i_act","i_rea","omni_48k")}

            batch_data = {k: recursive_to(v, cfg["device"]) for k, v in batch.items() if k not in ["audio","texts"]}
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