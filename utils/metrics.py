# utils/metrics.py
import torch, torch.nn.functional as F
import wandb

@torch.no_grad()
def cosine_sim(a_emb: torch.Tensor, t_emb: torch.Tensor):
    a = F.normalize(a_emb, dim=-1); t = F.normalize(t_emb, dim=-1)
    return t @ a.T  # [N_text, N_audio]
@torch.no_grad()
def eval_retrieval(a_spa, a_src, t_spa, t_src, src_lb, spa_lb, device, use_wandb=True, epoch=None):
    """
    Retrieval (multi-positive; no pooling)
      - On-task:  SRC (audio_source <-> text_source), SPA (audio_space <-> text_space)
      - Off-task: X-SRC (audio_space <-> text_source; source一致), 
                  X-SPA (audio_source <-> text_space; space一致)
      さらに Chance@K と Excess@K (= R@K - Chance@K) を出す
    """
    ids_src = src_lb
    ids_spa = spa_lb

    # ---- chance の計算（multi-positiveの理論当たり確率） ----
    def _chance_at_k(query_ids, gallery_ids, ks=(1,5,10)):
        Ng = int(gallery_ids.numel())
        # P_i: 各クエリに対してギャラリー中の正解数
        P = torch.stack([(gallery_ids == q).sum() for q in query_ids]).float()  # [Nq]

        out = {}
        for k in ks:
            k = min(k, Ng)
            if k <= 0:
                out[f"Chance@{k}"] = 0.0
                continue
            j = torch.arange(k, dtype=torch.float32)  # [k]
            # prob(no hit) = Π_j (Ng - P - j) / (Ng - j)
            denom_log = torch.sum(torch.log((Ng - j)))  # スカラー
            num_log   = torch.sum(torch.log((Ng - P.unsqueeze(1) - j).clamp(min=1.0)), dim=1)  # [Nq]
            prob_no   = torch.exp(num_log - denom_log).clamp(0.0, 1.0)  # [Nq]
            out[f"Chance@{k}"] = (1.0 - prob_no).mean().item()
        return out

    def _eval_block(prefix, q_emb, g_emb, q_ids, g_ids):
        # T2A（行＝query、列＝gallery）
        S = cosine_sim(q_emb, g_emb)  # [Nq, Ng]
        R = recall_at_k_multi(S, q_ids, g_ids, ks=(1,5,10))
        C = _chance_at_k(q_ids, g_ids, ks=(1,5,10))
        mets = {f"{prefix}/T2A/{k}": v for k, v in R.items()} | \
               {f"{prefix}/T2A/{k}": v for k, v in C.items()}
        for k in (1,5,10):
            mets[f"{prefix}/T2A/Excess@{k}"] = R[f"R@{k}"] - C[f"Chance@{k}"]

        # A2T（転置）
        Rr = recall_at_k_multi(S.T, g_ids, q_ids, ks=(1,5,10))
        Cr = _chance_at_k(g_ids, q_ids, ks=(1,5,10))
        mets |= {f"{prefix}/A2T/{k}": v for k, v in Rr.items()}
        mets |= {f"{prefix}/A2T/{k}": v for k, v in Cr.items()}
        for k in (1,5,10):
            mets[f"{prefix}/A2T/Excess@{k}"] = Rr[f"R@{k}"] - Cr[f"Chance@{k}"]
        return mets

    # ---- On-task ----
    mets = {}
    mets |= _eval_block("SRC", t_src, a_src, ids_src, ids_src)  # text_source -> audio_source
    mets |= _eval_block("SPA", t_spa, a_spa, ids_spa, ids_spa)  # text_space  -> audio_space

    # ---- Off-task（反対の埋め込み）: 低いほど良い → Excess が 0 に近いほど良い ----
    mets |= _eval_block("X-SRC", t_spa, a_spa, ids_src, ids_src)  # text_source -> audio_source（space一致）
    mets |= _eval_block("X-SPA", t_src, a_src, ids_spa, ids_spa)  # text_space  -> audio_space（source一致）

    if use_wandb:
        wandb.log({"epoch": epoch, **mets})
    return mets
@torch.no_grad()
def recall_at_k(sim: torch.Tensor, ids_t: torch.Tensor, ids_a: torch.Tensor, ks=(1,5,10)):
    N = sim.size(0); ranks = []; hits = {k:0 for k in ks}
    for i in range(N):
        pos = (ids_a == ids_t[i])
        if not torch.any(pos): continue
        order = torch.argsort(sim[i], descending=True)
        rank = (pos[order].nonzero(as_tuple=False)[0,0].item() + 1)  # 1-origin
        ranks.append(rank)
        for k in ks:
            if rank <= k: hits[k] += 1
    M = max(1, len(ranks))
    out = {f"R@{k}": hits[k]/M for k in ks}
    out["MedR"] = float(torch.median(torch.tensor(ranks)).item()) if ranks else float("nan")
    return out
@torch.no_grad()
def recall_at_k_multi(S, query_ids, gallery_ids, ks=(1,5,10)):
    same = query_ids[:, None].eq(gallery_ids[None, :])    # [Nq, Ng] 正解マスク
    ranks = S.argsort(dim=1, descending=True)             # [Nq, Ng]

    out = {}
    for k in ks:
        topk = ranks[:, :k]                               # [Nq, k]
        hit = same.gather(1, topk).any(dim=1).float()
        out[f"R@{k}"] = hit.mean().item()

    # 追加で MedR / MnR（最初の正解の順位）
    first = torch.full((S.size(0),), S.size(1)+1, dtype=torch.long, device=S.device)
    for i in range(S.size(0)):
        row = same[i][ranks[i]]
        pos = torch.nonzero(row, as_tuple=False)
        if pos.numel(): first[i] = pos[0,0] + 1
    out["MedR"] = first.median().item(); out["MnR"] = first.float().mean().item()
    return out



# ------- 汎用: ペアワイズ cosine 距離 -------
def _pairwise_cosine_distance(x: torch.Tensor) -> torch.Tensor:
    # x: [N,D]
    x = F.normalize(x, dim=1)
    return 1.0 - (x @ x.T)  # [N,N]

# ------- Invariance Ratio (IR) -------
@torch.no_grad()
def invariance_ratio(emb: torch.Tensor,
                     src_ids: torch.Tensor,
                     spa_ids: torch.Tensor) -> dict:
    """
    emb:     [N,D]  単位は audio_space か audio_source など任意
    src_ids: [N]    source ID
    spa_ids: [N]    space  ID
    戻り値: {'IR_space': float, 'IR_source': float, カウント...}
    """
    N = emb.size(0)
    if N < 2:
        return {"IR_space": float("nan"), "IR_source": float("nan"),
                "num_ss_ds": 0, "num_sd_ss": 0}

    dist = _pairwise_cosine_distance(emb)    # [N,N]
    eye  = torch.eye(N, dtype=torch.bool, device=emb.device)
    src_eq = src_ids[:, None].eq(src_ids[None, :])
    spa_eq = spa_ids[:, None].eq(spa_ids[None, :])
    valid  = ~eye

    # same-space diff-source / same-source diff-space
    same_space_diff_source = spa_eq & (~src_eq) & valid
    same_source_diff_space = src_eq & (~spa_eq) & valid

    d_ss_ds = dist[same_space_diff_source]
    d_sd_ss = dist[same_source_diff_space]
    eps = 1e-8

    # audio_space なら「sourceに不変」で d_sd_ss / d_ss_ds が大きいほど良い
    IR_space  = (d_sd_ss.mean() / (d_ss_ds.mean() + eps)).item() if d_ss_ds.numel() and d_sd_ss.numel() else float("nan")
    # audio_source なら「spaceに不変」で d_ss_ds / d_sd_ss が大きいほど良い
    IR_source = (d_ss_ds.mean() / (d_sd_ss.mean() + eps)).item() if d_ss_ds.numel() and d_sd_ss.numel() else float("nan")

    return {
        "IR_space": IR_space,   # emb を audio_space に当てたときに解釈
        "IR_source": IR_source, # emb を audio_source に当てたときに解釈
        "num_ss_ds": int(same_space_diff_source.sum().item()),
        "num_sd_ss": int(same_source_diff_space.sum().item()),
    }
