#!/usr/bin/env python3
# train_sup.py  ― supervised contrastive (source / space) 版
import argparse, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import random

random.seed(42)  # 再現性のため


#ToDO:Supervised Contrasive Learningを読む。https://proceedings.neurips.cc/paper_files/paper/2020/file/d89a66c7c80a29b1bdbab0f2a1a94af8-Paper.pdf?utm_source=chatgpt.com
# ────────────────────────── CLI ──────────────────────────
def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_csv", required=True)
    ap.add_argument("--rir_csv",   required=True)
    ap.add_argument("--audio_base",required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--batch_size",type=int, default=8)
    ap.add_argument("--n_views",   type=int, default=2)
    ap.add_argument("--epochs",    type=int, default=5)
    ap.add_argument("--lr",        type=float, default=1e-4)
    ap.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()








# ───────── Supervised cross-modal contrastive loss ─────────
def sup_contrast(a, b, labels, logit_scale, eps=1e-8):
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)

    logits = torch.matmul(a, b.T) * logit_scale.exp()      # S_ij
    B = logits.size(0)

    # 正例マスク (同ラベル & 対角除外)
    with torch.no_grad():
        pos_mask = labels[:, None].eq(labels[None, :]).fill_diagonal_(False)
        valid = pos_mask.any(1)                            # 正例ゼロ行を除外

    # log-sum-exp 安定化
    max_sim, _ = logits.max(dim=1, keepdim=True)
    exp_sim = torch.exp(logits - max_sim) * (~torch.eye(B, dtype=bool, device=a.device))
    denom = exp_sim.sum(1, keepdim=True) + eps            # Z_i

    log_prob = logits - max_sim - denom.log()             # log p_ij
    mean_log_pos = (log_prob * pos_mask).sum(1) / pos_mask.sum(1).clamp(min=1)

    loss = -mean_log_pos[valid].mean()
    return loss

# ─────────────────────────── Main ─────────────────────────
def main():
    args = parse()
    from dataset.audio_rir_dataset import AudioRIRDataset, collate_fn
    ds = AudioRIRDataset(
        csv_audio=args.audio_csv,
        base_dir=args.audio_base,
        csv_rir=args.rir_csv,
        n_views=args.n_views,
        split=args.split,
    )
    dl = DataLoader(ds, batch_size=args.batch_size,
                    shuffle=True, num_workers=4,
                    collate_fn=collate_fn, pin_memory=True)

    from model.delsa_model import DELSA      # アップロード済みモデル
    model = DELSA(audio_encoder_cfg={}, text_encoder_cfg={}).to(args.device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for ep in range(1, args.epochs+1):
        model.train()
        for step, batch in enumerate(dl, 1):
            # ----------- flatten ------------
            audio_dict = {k: torch.stack([d[k] for d in batch["audio"]]).to(args.device)
                          for k in ("i_act","i_rea","omni_48k")}
            texts  = batch["texts"]
            src_lb = batch["source_id"].reshape(-1).to(args.device)  # [B']
            spc_lb = batch["space_id"].reshape(-1).to(args.device)

            # ----------- forward ------------
            out = model(audio_dict, texts)   # expect 4 embeddings + logit_scale
            a_s  = F.normalize(out["audio_space_emb"],  dim=-1)
            t_s  = F.normalize(out["text_space_emb"],   dim=-1)
            a_src= F.normalize(out["audio_source_emb"], dim=-1)
            t_src= F.normalize(out["text_source_emb"],  dim=-1)

            T = out["logit_scale"].exp()     # 温度

            loss_space  = sup_contrast(a_s,  t_s,  spc_lb, T)
            loss_source = sup_contrast(a_src,t_src,src_lb, T)
            loss = 0.5 * (loss_space + loss_source)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 10 == 0:
                print(f"Epoch {ep} Step {step}/{len(dl)}  "
                      f"space={loss_space:.4f}  src={loss_source:.4f}")

        # -------- checkpoint -------------
        ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True)
        torch.save({"model": model.state_dict(),
                    "opt":   opt.state_dict(),
                    "epoch": ep},
                   ckpt_dir/f"ckpt_sup_ep{ep}.pt")
        print(f"[✓] Saved checkpoint for epoch {ep}")

if __name__ == "__main__":
    main()
