#!/usr/bin/env python3
# train_for_singleGPU.py  ― supervised contrastive (source / space) + per-epoch retrieval eval
from utils.metrics import invariance_ratio
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import random, wandb, math, sys, yaml
from utils.metrics import invariance_ratio
from tqdm.auto import tqdm
from dataset.audio_rir_dataset import AudioRIRDataset, collate_fn   # 既存
from dataset.precomputed_val_dataset import PrecomputedValDataset   # 追加①
from utils.metrics import cosine_sim, recall_at_k, recall_at_k_multi, eval_retrieval              # 追加②
from model.delsa_model import DELSA                                # ← 修正
from dataset.precomputed_val_dataset import PrecomputedValDataset
from utils.catesian_sampler import CartesianBatchSampler
import torch, re
random.seed(42)
import torchaudio, torch
from functools import lru_cache
from dataset.audio_rir_dataset import foa_to_iv, FOA_SR, IV_SR  # 既存実装を再利用:contentReference[oaicite:15]{index=15}
import os
import torchaudio
torchaudio.set_audio_backend("sox_io")

def _pick_sd(obj):
    if isinstance(obj, dict):
        for k in ("model_state_dict", "state_dict", "model"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj

def _strip_module(sd):
    return { (k.replace("module.", "", 1) if k.startswith("module.") else k): v for k,v in sd.items() }

def _load_prefix(mod, full_sd, prefix, strict=False, tag=""):
    sub = { k[len(prefix):]: v for k,v in full_sd.items() if k.startswith(prefix) }
    if not sub:
        return 0
    missing, unexpected = mod.load_state_dict(sub, strict=strict)
    print(f"[load]{tag or prefix}: loaded={len(sub)} missing={len(missing)} unexpected={len(unexpected)}")
    return len(sub)

def load_backbones_only(model, pt_path, device="cpu", strict=False) -> bool:
    """戻り値: 何かしらロードできたらTrue（=初期重みとして使えた）"""
    if not pt_path or not os.path.isfile(pt_path):
        print(f"[load] baseline checkpoint not found: {pt_path} (skip)")
        return False
    sd = _strip_module(_pick_sd(torch.load(pt_path, map_location=device)))
    n_loaded = 0

    # Text backbone (RoBERTa)
    if hasattr(model, "text_encoder") and hasattr(model.text_encoder, "roberta"):
        for pref in ("text_encoder.roberta.", "roberta."):
            n_loaded += _load_prefix(model.text_encoder.roberta, sd, pref, strict=strict, tag="text.roberta")

    # Audio backbones (HTSAT & Spatial)
    if hasattr(model, "audio_encoder"):
        # HTSAT は shared_audio_encoder.HTSAT 内で self.model に本体が入る実装
        for attr, prefs in [
            ("htsat", ("audio_encoder.htsat.model.", "audio_encoder.htsat.", "htsat.model.", "htsat.")),
            ("spatial_branch", ("audio_encoder.spatial_branch.", "spatial_branch.")),
        ]:
            if hasattr(model.audio_encoder, attr):
                n_loaded += sum(_load_prefix(getattr(model.audio_encoder, attr), sd, p, strict=strict,
                                             tag=f"audio.{attr}") for p in prefs)

    print(f"[load] total loaded tensors: {n_loaded}")
    return n_loaded > 0

def freeze_module(m):
    for p in m.parameters(): p.requires_grad = False

def maybe_init_from_baseline(model, ckpt_path, device, freeze_if_loaded=True) -> bool:
    """ckptが読めた場合のみ凍結。読めなければ何もしない。戻り値: 読めたかどうか"""
    loaded = load_backbones_only(model, ckpt_path, device=device, strict=False)
    if loaded and freeze_if_loaded:
        if hasattr(model, "text_encoder") and hasattr(model.text_encoder, "roberta"):
            freeze_module(model.text_encoder.roberta)
        if hasattr(model, "audio_encoder"):
            if hasattr(model.audio_encoder, "htsat"):           freeze_module(model.audio_encoder.htsat)
            if hasattr(model.audio_encoder, "spatial_branch"):  freeze_module(model.audio_encoder.spatial_branch)
        print("[freeze] backbones are frozen (because baseline was loaded).")
    return loaded
_M_A2FOA = torch.tensor([
    [0.5, 0.5, 0.5, 0.5],  # W
    [0.5, -0.5, 0.5, -0.5],  # Y
    [0.5, -0.5, -0.5, 0.5],  # Z
    [0.5, 0.5, -0.5, -0.5],  # X
], dtype=torch.float32)  # [4,4] A-format -> FOA変換行列
# def _aformat_to_foa(wet_b):  # wet: [4,T] A-format -> FOA (W,Y,Z,X)
#     m0, m1, m2, m3 = wet_b[:,0], wet_b[:,1], wet_b[:,2], wet_b[:,3]
#     W = (m0+m1+m2+m3)/2; X = (m0+m1-m2-m3)/2
#     Y = (m0-m1+m2-m3)/2; Z = (m0-m1-m2+m3)/2
#     return torch.stack([W, Y, Z, X], dim=1)

def _aformat_to_foa(wet_b):  # wet_b: [B,4,T] A-format -> FOA (W,Y,Z,X)
    if wet_b.size(1) != 4:
        raise ValueError(f"wet_b must have 4 channels (A-format), but got {wet_b.size(1)} channels.")
    return torch.einsum('ij,bjt->bit', _M_A2FOA.to(wet_b.device, wet_b.dtype), wet_b)
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
    Tr_list = []
    for a in audio_list:
        rir, sr = _load_rir_cpu_cached(a["rir_path"])
        rirs_cpu.append(rir if sr == FOA_SR else torchaudio.functional.resample(rir, sr, FOA_SR))
        Tr_list.append(rir.shape[-1])

        
    Tr_max = max(Tr_list)                        # バッチ内の最大 RIR 長
    pad_rirs = []
    for rir in rirs_cpu:
        pad = Tr_max - rir.shape[-1]
        if pad > 0:
            rir = torch.nn.functional.pad(rir, (0, pad))  # 末尾に 0 詰め（[4, Tr_max]）
        pad_rirs.append(rir)


    rirs = torch.stack(pad_rirs).to(device, non_blocking=True)  # [B,4,Tr]

    # 3) GPUでFFT畳み込み（A-format）→ FOA 変換
    wet_a = _fft_convolve_batch_gpu(drys, rirs)                 # [B,4,T]
    foa   = _aformat_to_foa(wet_a).contiguous() # [B,4,T]
    omni_48k = foa[:, 0, :].detach().to("cpu")

    # 4) 16kへリサンプル（環境により GPU 前処理）
    with torch.no_grad():
        foa_16k = torchaudio.functional.resample(foa, orig_freq=FOA_SR, new_freq=IV_SR)
    # 5) foa_to_iv（deviceに従ってCUDA/CPUで実行可。ここではCPUでも十分高速）:contentReference[oaicite:17]{index=17}

        i_act, i_rea = foa_to_iv(foa_16k) 

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
        "resume_ckpt": None,
        "wandb_id": None,
        "ckpt_dir": None,
        "save_every": 1,
        "wandb": True,
        "proj": "delsa-sup-contrast",
        "ckpt_dir": "Spatial_AudioCaps/takamichi09/checkpoints_delsa_fromELSA",
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
    a = F.normalize(a, dim=-1, eps=eps); b = F.normalize(b, dim=-1, eps = eps)

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

def physical_loss(model_output, batch_data, isNorm=True,stats = None):
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
    torch.backends.cuda.matmul.allow_tf32 =False 
    torch.backends.cudnn.allow_tf32 =False
    torch.backends.cudnn.benchmark = False  # 可変長でなければ有効
    cfg = load_config()
    if cfg["wandb"]:
        wandb.init(project=cfg["proj"],
                   config=cfg, 
                   save_code=True,
                   id=cfg.get("wandb_id"),
                   resume="allow" if cfg.get("wandb_id") else None)
    # -------- Train loader (従来通り) --------
    train_ds = AudioRIRDataset(csv_audio=cfg["audio_csv_train"], base_dir=cfg["audio_base"],
                               csv_rir=cfg["rir_csv_train"], n_views=cfg["n_views"],
                               split=cfg["split"], batch_size=cfg["batch_size"],
                               pcm_base_dir=cfg.get("audio_base_pcm"), prefer_pcm=cfg.get("prefer_pcm", False)) 
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=8,
                          collate_fn=collate_fn, pin_memory=True, persistent_workers=False,prefetch_factor=1)
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
    dry_per_batch  = cfg.get("val_dry_per_batch", 24)
    rirs_per_batch = cfg.get("val_rirs_per_batch", 3)
    batch_sampler = CartesianBatchSampler(
        index_csv=val_csv,
        dry_per_batch=dry_per_batch,
        rirs_per_batch=rirs_per_batch,
        drop_last = False,
        shuffle_audio = False,
        shuffle_rir = False,
        seed = 0,
    )
    val_dl = DataLoader(val_ds, batch_sampler=batch_sampler, num_workers=4, collate_fn=collate_fn, pin_memory=False)

    # -------- Model / Optim --------
    model = DELSA(audio_encoder_cfg={}, text_encoder_cfg={}).to(cfg["device"])
    
    log_vars = torch.nn.Parameter(torch.zeros(3, device=cfg["device"]))
    loaded = maybe_init_from_baseline(
        model,
        cfg.get("baseline_ckpt_path"),
        device=cfg["device"],
        freeze_if_loaded=cfg.get("freeze_backbones_if_baseline_loaded", True),
    )

    base_params = (filter(lambda p: p.requires_grad, model.parameters())
                if (loaded and cfg.get("freeze_backbones_if_baseline_loaded", True))
                else model.parameters())

    opt = torch.optim.AdamW(
        [{"params": base_params, "weight_decay": 0.01},
        {"params": [log_vars], "lr": cfg["lr"], "weight_decay": 0.0}],
        lr=cfg["lr"]
    )
    total_steps = cfg["epochs"] * len(train_dl)
    warmup_steps = max(1, int(0.05 * total_steps))   # 5% warmup（3–10%で調整）
    hold   = int(0.30 * total_steps)
    min_lr_ratio = 0.3                               # 最終LR = 初期LRの30%
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        if step < warmup_steps + hold:
            return 1.0
        prog = (step - warmup_steps - hold) / max(1, total_steps - warmup_steps - hold)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * prog))

    
    # Scheduler（log_vars は一定LRに固定したい場合の例）
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lr_lambda=[lr_lambda, lambda step: 1.0]  # [model用, log_vars用]
    )
    use_amp = bool(cfg.get("use_amp", True))
    scaler = torch.amp.GradScaler(enabled=use_amp)
    epoch_bar = tqdm(range(1, cfg["epochs"]+1), desc="Epochs", unit="ep", dynamic_ncols=True)
    start_ep = 1
    if cfg.get("resume_ckpt") and os.path.isfile(cfg["resume_ckpt"]):
        ckpt = torch.load(cfg["resume_ckpt"], map_location=cfg["device"])
        if "model" in ckpt: model.load_state_dict(ckpt["model"])
        if "opt"   in ckpt: opt.load_state_dict(ckpt["opt"])
        if "scaler" in ckpt:
            try: scaler.load_state_dict(ckpt["scaler"])
            except Exception: pass
        prev_ep = int(ckpt.get("epoch", 0))
        start_ep = prev_ep + 1
        prev_steps = prev_ep * len(train_dl)
        scheduler.last_epoch = max(-1, prev_steps - 1)
        if cfg["wandb"]:
            wandb.run.summary["resumed_from_epoch"] = prev_ep
            wandb.run.summary["resumed_from_ckpt"] = cfg["resume_ckpt"]
        tqdm.write(f"[↩] Resumed from {cfg['resume_ckpt']} (epoch {prev_ep}), next epoch = {start_ep}")
    else: 
        if cfg["resume_ckpt"]:
            tqdm.write(f"[ERR] Resume checkpoint not found: {cfg['resume_ckpt']}")
    epoch_bar = tqdm(range(start_ep, cfg["epochs"]+1), desc="Epochs", unit="ep", dynamic_ncols=True)
    exclude_diag = cfg.get("exclude_diag", True)
    for ep in epoch_bar:
        model.train()
        train_bar = tqdm(enumerate(train_dl, 1),total=len(train_dl), desc="Training", unit="batch", dynamic_ncols=True)
        for step, batch in train_bar:
            if batch is None: continue
            audio_list = batch["audio"]

            if _is_defer_format(audio_list):
                audio = _build_audio_from_defer(audio_list, cfg["device"]) 
            else:
                audio = {
                    "i_act": torch.stack([d["i_act"] for d in batch["audio"]]).to(cfg["device"]),
                    "i_rea": torch.stack([d["i_rea"] for d in batch["audio"]]).to(cfg["device"]),
                    "omni_48k": torch.stack([d["omni_48k"] for d in batch["audio"]])  # ★ CPU
                }
            batch_data = {k: recursive_to(v, cfg["device"]) for k, v in batch.items() if k not in ["audio","texts"]}
            texts  = batch["texts"]
            src_lb = batch["source_id"].reshape(-1).to(cfg["device"])
            spa_lb = batch["space_id"].reshape(-1).to(cfg["device"])
            with torch.autocast(cfg["device"], enabled=use_amp):
                out = model(audio, texts)
                a_spa  = F.normalize(out["audio_space_emb"], dim = -1)
                t_spa  = F.normalize(out["text_space_emb"], dim = -1)
                a_src  = F.normalize(out["audio_source_emb"], dim = -1)
                t_src  = F.normalize(out["text_source_emb"], dim = -1)
                logit_s = out["logit_scale"]

                loss_space = sup_contrast(a_spa,  t_spa,  spa_lb, logit_s, exclude_diag=exclude_diag)
                loss_source = sup_contrast(a_src, t_src, src_lb, logit_s, exclude_diag=exclude_diag)
                phys_log, phys_loss = physical_loss(out, batch_data, isNorm=True, stats=train_stats)
                w = torch.exp(-log_vars)  # [3]
                loss_space_w  = w[0] * loss_space  + log_vars[0]
                loss_source_w = w[1] * loss_source + log_vars[1]
                loss_phys_w   = w[2] * phys_loss   + log_vars[2]
                loss = loss_space_w + loss_source_w + loss_phys_w
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            prev_scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            # オーバーフローが起きると scale が下がる → その時は scheduler を回さない
            if scaler.get_scale() >= prev_scale:   # つまり今回 step できた
                scheduler.step()
            if step % 10 == 0:
                train_bar.set_postfix(
                    space=float(loss_space.detach().cpu()),
                    src=float(loss_source.detach().cpu()),
                    phys=float(phys_loss.detach().cpu()),
                    mean=float(loss.detach().cpu())
                )

            if step % 10 == 0:
                tqdm.write(f"Epoch {ep} Step {step}/{len(train_dl)}  space={loss_space:.4f}  src={loss_source:.4f}")
                if cfg["wandb"]:
                    wandb.log({"loss/space": loss_space.item(), "loss/source": loss_source.item(),
                               "loss/physical": phys_loss.item(), "loss/dir": phys_log["loss_dir"],
                               "loss/distance": phys_log["loss_distance"], "loss/area": phys_log["loss_area"],
                               "loss/reverb": phys_log["loss_reverb"], "logit_scale": out["logit_scale"].item(),
                               "loss/mean": loss.item(), "epoch": ep, "step": step + (ep-1)*len(train_dl)})

        # # -------- Validation (Retrieval + 物理lossの平均) --------
        model.eval()
        
        val_losses = {"space":0.0,"source":0.0,"physical":0.0,"direction":0.0,"distance":0.0,"area":0.0,"reverb":0.0,"count":0}
        buf = {"a_spa": [], "a_src": [], "t_spa": [], "t_src": [], "src_lb": [], "spa_lb": []}
        with torch.no_grad():
            val_bar = tqdm(val_dl, total=len(val_dl), desc=f"Val   ep{ep}", unit="batch",
                           leave=False, dynamic_ncols=True)
            for batch in val_bar:

                if batch is None: continue
                batch_data = {k: recursive_to(v, cfg["device"]) for k, v in batch.items() if k not in ["audio","texts"]}
                audio = {k: torch.stack([d[k] for d in batch["audio"]]).to(cfg["device"]) for k in ("i_act","i_rea","omni_48k")}
                texts  = batch["texts"]
                src_lb = batch["source_id"].reshape(-1).to(cfg["device"])
                spa_lb = batch["space_id"].reshape(-1).to(cfg["device"])
                with torch.autocast(cfg["device"], enabled=use_amp):
                    out = model(audio, texts)
                    a_spa  = F.normalize(out["audio_space_emb"], dim = -1)
                    t_spa  = F.normalize(out["text_space_emb"], dim = -1)
                    a_src  = F.normalize(out["audio_source_emb"], dim = -1)
                    t_src  = F.normalize(out["text_source_emb"], dim = -1)
                    logit_s = out["logit_scale"]

                    l_sp  = sup_contrast(a_spa,  t_spa,  spa_lb, logit_s, exclude_diag=exclude_diag)
                    l_sr  = sup_contrast(a_src, t_src, src_lb, logit_s, exclude_diag=exclude_diag)
                    phys_log, l_phys = physical_loss(out, batch_data, isNorm=False, stats=train_stats)
                    buf["a_spa"].append(a_spa.detach().float().cpu())
                    buf["a_src"].append(a_src.detach().float().cpu())
                    buf["t_spa"].append(t_spa.detach().float().cpu())
                    buf["t_src"].append(t_src.detach().float().cpu())
                    buf["src_lb"].append(src_lb.detach().float().cpu())
                    buf["spa_lb"].append(spa_lb.detach().float().cpu())
                val_losses["space"]    += l_sp.item()
                val_losses["source"]   += l_sr.item()
                val_losses["physical"] += l_phys.item()
                val_losses["direction"]+= phys_log["loss_dir"]
                val_losses["distance"] += phys_log["loss_distance"]
                val_losses["area"]     += phys_log["loss_area"]
                val_losses["reverb"]   += phys_log["loss_reverb"]
                val_losses["count"]    += 1
                # ★ 検証のpostfix
                val_bar.set_postfix(
                    sp=float(val_losses["space"]/max(1,val_losses["count"])),
                    sr=float(val_losses["source"]/max(1,val_losses["count"])),
                    phys=float(val_losses["physical"]/max(1,val_losses["count"]))
                )

        A_SPA = torch.cat(buf["a_spa"], dim=0)
        A_SRC = torch.cat(buf["a_src"], dim=0)
        T_SPA = torch.cat(buf["t_spa"], dim=0)
        T_SRC = torch.cat(buf["t_src"], dim=0)
        SRC_LB = torch.cat(buf["src_lb"], dim=0)
        SPA_LB = torch.cat(buf["spa_lb"], dim=0)
        n = max(1, val_losses["count"])
        val_mean = (val_losses["space"]/n + val_losses["source"]/n + val_losses["physical"]/n)
        tqdm.write(f"Epoch {ep}  [VAL] space={val_losses['space']/n:.4f}  src={val_losses['source']/n:.4f}  phys={val_losses['physical']/n:.4f}")
        mets = eval_retrieval(a_spa=A_SPA, a_src=A_SRC, t_spa=T_SPA, t_src=T_SRC, src_lb=SRC_LB, spa_lb=SPA_LB, device=cfg["device"], use_wandb=False, epoch=ep)

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
                


        ir_a = invariance_ratio(A_SPA, SRC_LB, SPA_LB) # audio_space
        ir_b = invariance_ratio(A_SRC, SRC_LB, SPA_LB) # audio_source
        ir_tspa = invariance_ratio(T_SPA, SRC_LB, SPA_LB) # text_space
        ir_tsrc = invariance_ratio(T_SRC, SRC_LB, SPA_LB) # text_source
        tqdm.write(f"[IR] audio_space={ir_a.get('IR_space', None)}  audio_source={ir_b.get('IR_source', None)}")
        # W&Bへ
        if cfg["wandb"]:
            wandb.log({
                "IR/Z_audio_space/IR_Space":  ir_a["IR_space"],
                "IR/Z_audio_space/IR_source": ir_a["IR_source"],
                "IR/Z_audio_source/IR_Space": ir_b["IR_space"],
                "IR/Z_audio_source/IR_source": ir_b["IR_source"],
                "IR/Z_text_space/IR_Space":   ir_tspa["IR_space"],
                "IR/Z_text_space/IR_source":  ir_tspa["IR_source"],
                "IR/Z_text_source/IR_Space":  ir_tsrc["IR_space"],
                "IR/Z_text_source/IR_source": ir_tsrc["IR_source"],
                "IR/count_ss_ds":  ir_a["num_ss_ds"],
                "IR/count_sd_ss":  ir_a["num_sd_ss"],
                "epoch": ep,
            })
        # -------- checkpoint -------------
        ckpt_dir = Path(cfg.get("ckpt_dir", "Spatial_AudioCaps/takamichi09/checkpoints_delsa_fromELSA"))
        ckpt_dir.mkdir(exist_ok=True, parents=True)
        if (ep % int(cfg.get("save_every", 1))) == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep},
                       ckpt_dir/f"ckpt_sup_ep{ep}.pt")
            tqdm.write(f"[✓] Saved checkpoint for epoch {ep}")
    tqdm.write("[✓] Training loop finished.")

if __name__ == "__main__":
    try:
        main()
    finally:
        if wandb.run is not None:
            wandb.finish()
        print("[✓] Finished training.")
