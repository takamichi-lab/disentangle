#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate checkpoint + t-SNE visualization (GPU-first)
- GPU: RAPIDS cuML の TSNE を優先使用（cupy 直接入力）
- CPU: cuML が無い場合は scikit-learn に自動フォールバック
- 1) space_id：K色 + others灰
- 2) 方位×距離（8-ish離散色）
"""

"""
python3 evaluate_checkpoint.py \
  --config config.yaml \
  --ckpt Spatial_AudioCaps/takamichi09/checkpoints_delsa_fromELSA/ckpt_sup_ep9.pt \
  --device cuda \
  --tsne_device gpu \            # ★ ここをgpu（デフォはauto）
  --tsne_perplexity 30 \
  --tsne_max_points 4000 \
  --highlight_k 20 \
  --outdir eval_out \
  --wandb
"""
import argparse, os, random, yaml
from pathlib import Path
from typing import Tuple, List, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# ---- Project imports (既存コードに準拠) ----
from model.delsa_model import DELSA                      # 4埋め込みを返す:contentReference[oaicite:3]{index=3}
from dataset.precomputed_val_dataset import PrecomputedValDataset  # 物理メタ付き:contentReference[oaicite:4]{index=4}
from dataset.audio_rir_dataset import collate_fn

# ========= Utilities =========
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def l2norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)

def _dir_label(az_deg: float) -> str:
    if -35 <= az_deg <= 35:    return "front"
    if 55 <= az_deg <= 125:    return "right"
    if -125 <= az_deg <= -55:  return "left"
    if az_deg >= 145 or az_deg <= -145: return "back"
    return "mid"

def _dist_label(d_m: float) -> str:
    if d_m < 1.0: return "near"
    if d_m > 2.0: return "far"
    return "mid"

def build_dirxdist_label(az_deg: float, d_m: float) -> str:
    return f"{_dir_label(az_deg)}-{_dist_label(d_m)}"

def take_subset_balanced(labels: np.ndarray, max_n: int) -> np.ndarray:
    if len(labels) <= max_n:
        return np.arange(len(labels))
    uniq = np.unique(labels)
    per = max(1, max_n // len(uniq))
    idxs = []
    for u in uniq:
        pool = np.nonzero(labels == u)[0]
        if len(pool) <= per: idxs.extend(pool.tolist())
        else: idxs.extend(np.random.choice(pool, size=per, replace=False).tolist())
    rest = max_n - len(idxs)
    if rest > 0:
        remain = sorted(set(range(len(labels))) - set(idxs))
        idxs.extend(np.random.choice(remain, size=rest, replace=False).tolist())
    return np.array(sorted(idxs))

def color_palette_k(k: int) -> List:
    import itertools
    base = plt.get_cmap("tab20").colors
    return list(itertools.islice(itertools.cycle(base), k))

# ========= t-SNE backends =========
def tsne_fit_auto(X_torch: torch.Tensor, perplexity: int, seed: int, prefer: str = "auto"):
    """
    X_torch: [N, D] torch.Tensor（GPU or CPU）
    prefer: "gpu" / "cpu" / "auto"
    戻り値: np.ndarray [N, 2]
    """
    # Try cuML (GPU)
    use_gpu = (prefer in ("gpu", "auto")) and X_torch.is_cuda
    if use_gpu:
        try:
            import cupy as cp
            from cuml.manifold import TSNE as cuTSNE
            from torch.utils.dlpack import to_dlpack

            X_cu = cp.fromDlpack(to_dlpack(X_torch))  # ゼロコピーで cupy に変換
            tsne = cuTSNE(n_components=2, perplexity=perplexity, init="pca",
                          learning_rate="auto", random_state=seed, method="barnes_hut")
            Y_cu = tsne.fit_transform(X_cu)           # cupy [N,2]
            Y = cp.asnumpy(Y_cu)                      # 可視化保存のため numpy に戻す
            return Y
        except Exception as e:
            print(f"[warn] cuML TSNE unavailable or failed ({type(e).__name__}: {e}). Falling back to CPU.")
    # Fallback: scikit-learn (CPU)
    from sklearn.manifold import TSNE as skTSNE
    X_np = X_torch.detach().cpu().numpy()
    Y = skTSNE(n_components=2, perplexity=perplexity, init="pca",
               learning_rate="auto", random_state=seed).fit_transform(X_np)
    return Y

# ========= Plot helpers =========
def plot_spaceid_discrete(xy: np.ndarray, labels: np.ndarray, out_png: Path, title: str, K: int, s: int = 10):
    uniq = sorted(list(set(labels.tolist())))
    K = min(K, len(uniq))
    labeled = uniq[:K]  # 均等出現なら先頭KでOK
    cols = color_palette_k(K)
    cmap = {sid: cols[i] for i, sid in enumerate(labeled)}
    x, y = xy[:,0], xy[:,1]
    plt.figure(figsize=(7,6), dpi=160)
    mask_other = ~np.isin(labels, labeled)
    plt.scatter(x[mask_other], y[mask_other], c="lightgray", s=s, alpha=0.15, label="others")
    for sid in labeled:
        m = (labels == sid)
        if m.any():
            plt.scatter(x[m], y[m], s=s, color=cmap[sid], label=f"space:{sid}")
    plt.legend(markerscale=2, fontsize=8, ncol=2, frameon=False)
    plt.title(title); plt.tight_layout(); plt.savefig(out_png); plt.close()
    print(f"[✓] saved: {out_png}")

def plot_by_category(xy: np.ndarray, cat: List[str], out_png: Path, title: str, s: int = 10):
    uniq = sorted(list(set(cat)))
    base = plt.get_cmap("tab20").colors
    color = {u: base[i % len(base)] for i, u in enumerate(uniq)}
    x, y = xy[:,0], xy[:,1]
    plt.figure(figsize=(7,6), dpi=160)
    for u in uniq:
        m = np.array([c == u for c in cat])
        if m.any():
            plt.scatter(x[m], y[m], s=s, color=color[u], label=u)
    plt.legend(markerscale=2, fontsize=8, ncol=2, frameon=False)
    plt.title(title); plt.tight_layout(); plt.savefig(out_png); plt.close()
    print(f"[✓] saved: {out_png}")

# ========= Main =========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tsne_perplexity", type=int, default=30)
    ap.add_argument("--tsne_max_points", type=int, default=4000)
    ap.add_argument("--highlight_k", type=int, default=20)
    ap.add_argument("--tsne_device", type=str, default="auto", choices=["auto","gpu","cpu"],
                    help="t-SNE backend preference")
    ap.add_argument("--outdir", type=str, default="eval_out")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    root = cfg.get("val_precomp_root")
    index_csv = cfg.get("val_index_csv") or (str(Path(root)/"val_precomputed.csv") if root else None)
    rir_csv   = cfg["rir_csv_val"]
    assert index_csv and Path(index_csv).exists(), f"index_csv not found: {index_csv}"

    # ---- Data / Model ----
    ds = PrecomputedValDataset(index_csv=index_csv, rir_meta_csv=rir_csv, root=root)   # 物理メタ付き:contentReference[oaicite:5]{index=5}
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    device = torch.device(args.device)
    model = DELSA(audio_encoder_cfg={}, text_encoder_cfg={}).to(device)                # 埋め込みはGPUで計算:contentReference[oaicite:6]{index=6}

    # checkpoint load
    sd = torch.load(args.ckpt, map_location=device)
    if isinstance(sd, dict) and any(k in sd for k in ("model","state_dict","model_state_dict")):
        for k in ("model","state_dict","model_state_dict"):
            if k in sd: sd = sd[k]; break
    sd = { (k.replace("module.","",1) if k.startswith("module.") else k): v for k,v in sd.items() }
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)}  unexpected={len(unexpected)}")

    model.eval()

    a_spa, t_spa, spa_ids = [], [], []
    az_list, dist_list = [], []
    with torch.no_grad():
        for batch in dl:
            if batch is None: continue
            audio = {k: torch.stack([d[k] for d in batch["audio"]]).to(device)
                     for k in ("i_act","i_rea","omni_48k")}
            texts = batch["texts"]
            out = model(audio, texts)                                                    # forward → GPU埋め込み:contentReference[oaicite:7]{index=7}
            a_spa.append(l2norm(out["audio_space_emb"]).detach())  # GPUのまま保持
            t_spa.append(l2norm(out["text_space_emb"]).detach())   # GPUのまま保持
            spa_ids.append(batch["space_id"].reshape(-1).cpu())
            meta = batch["rir_meta"]                                                     # 物理メタ:contentReference[oaicite:8]{index=8}
            az_list.append(meta["azimuth_deg"].cpu())
            dist_list.append(meta["distance"].cpu())

    # ---- Stack ----
    A = torch.cat(a_spa, dim=0)                 # [N, D] on device
    T = torch.cat(t_spa, dim=0)                 # [N, D] on device
    S = torch.cat(spa_ids, dim=0).numpy().astype(int)
    AZ = torch.cat(az_list, dim=0).numpy().astype(float)
    DST= torch.cat(dist_list, dim=0).numpy().astype(float)

    # ---- Subset ----
    sel = take_subset_balanced(S, max_n=args.tsne_max_points)
    A2 = A[sel]
    T2 = T[sel]
    S2 = S[sel]
    AZ2 = AZ[sel]
    DST2= DST[sel]

    # ---- t-SNE ----
    XY_a = tsne_fit_auto(A2, perplexity=args.tsne_perplexity, seed=args.seed, prefer=args.tsne_device)
    XY_t = tsne_fit_auto(T2, perplexity=args.tsne_perplexity, seed=args.seed, prefer=args.tsne_device)

    # Joint（同一空間に audio/text を投影したい時はまとめて回す）
    from numpy import concatenate as npcat
    XY_j = tsne_fit_auto(torch.cat([A2, T2], dim=0), perplexity=args.tsne_perplexity, seed=args.seed, prefer=args.tsne_device)
    XY_j_a, XY_j_t = XY_j[:len(A2)], XY_j[len(A2):]

    # ---- Plots ----
    plot_spaceid_discrete(XY_a, S2, outdir/"tsne_audio_space_by_spaceid.png",
                          f"t-SNE (audio_space) [K={args.highlight_k} colored]", K=args.highlight_k)
    plot_spaceid_discrete(XY_t, S2, outdir/"tsne_text_space_by_spaceid.png",
                          f"t-SNE (text_space)  [K={args.highlight_k} colored]", K=args.highlight_k)

    # joint
    plt.figure(figsize=(7,6), dpi=160)
    mask_other = ~np.isin(S2, sorted(list(set(S2)))[:min(args.highlight_k, len(set(S2)))])
    plt.scatter(XY_j_a[mask_other,0], XY_j_a[mask_other,1], s=8,  c="lightgray", alpha=0.15)
    plt.scatter(XY_j_t[mask_other,0], XY_j_t[mask_other,1], s=16, c="darkgray", alpha=0.15, marker="^")
    # 簡易ラベル色
    uniq = sorted(list(set(S2.tolist())))
    K = min(args.highlight_k, len(uniq))
    cols = color_palette_k(K)
    cmap = {sid: cols[i] for i, sid in enumerate(uniq[:K])}
    for sid in uniq[:K]:
        m = (S2 == sid)
        if m.any():
            plt.scatter(XY_j_a[m,0], XY_j_a[m,1], s=8,  color=cmap[sid], label=f"a:space{sid}")
            plt.scatter(XY_j_t[m,0], XY_j_t[m,1], s=16, color=cmap[sid], marker="^", label=f"t:space{sid}")
    plt.legend(markerscale=2, fontsize=8, ncol=2, frameon=False)
    plt.title(f"t-SNE (audio/text space) [K={K} colored]")
    plt.tight_layout(); out_joint = outdir/"tsne_joint_space_by_spaceid.png"; plt.savefig(out_joint); plt.close()
    print(f"[✓] saved: {out_joint}")

    # dir×dist（8-ish色）
    cat = [build_dirxdist_label(az, d) for az, d in zip(AZ2, DST2)]
    def plot_cat(xy, name, title):
        uniq_c = sorted(list(set(cat)))
        base = plt.get_cmap("tab20").colors
        color = {u: base[i % len(base)] for i, u in enumerate(uniq_c)}
        x, y = xy[:,0], xy[:,1]
        plt.figure(figsize=(7,6), dpi=160)
        for u in uniq_c:
            m = np.array([c == u for c in cat])
            plt.scatter(x[m], y[m], s=10, color=color[u], label=u)
        plt.legend(markerscale=2, fontsize=8, ncol=2, frameon=False)
        plt.title(title); plt.tight_layout(); outp = outdir/name; plt.savefig(outp); plt.close()
        print(f"[✓] saved: {outp}")
    plot_cat(XY_a, "tsne_audio_space_by_dirxdist.png", "t-SNE (audio_space) by dir×dist")
    plot_cat(XY_t, "tsne_text_space_by_dirxdist.png",  "t-SNE (text_space)  by dir×dist")

    # ---- W&B (optional) ----
    if args.wandb:
        import wandb
        run = wandb.init(project=cfg.get("proj","delsa-sup-contrast"),
                         name=cfg.get("run_name","eval-tsne"),
                         config={"tsne_perplexity": args.tsne_perplexity,
                                 "tsne_max_points": args.tsne_max_points,
                                 "backend": args.tsne_device})
        for p in ["tsne_audio_space_by_spaceid.png",
                  "tsne_text_space_by_spaceid.png",
                  "tsne_joint_space_by_spaceid.png",
                  "tsne_audio_space_by_dirxdist.png",
                  "tsne_text_space_by_dirxdist.png"]:
            run.log({p: wandb.Image(str(outdir/p))})
        run.finish()

if __name__ == "__main__":
    main()
