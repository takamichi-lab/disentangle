#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_cache.py — 評価用キャッシュ生成スクリプト
入力: エポック番号 or 直接 ckpt パス
出力: cache_dir/embeds_ep{epoch}.pt など（埋め込み4種 + space_id/source_id + rir_meta を1ファイルに保存）

依存:
- model/delsa_model.py : DELSA（4種の埋め込みを出力）:contentReference[oaicite:3]{index=3}
- dataset/precomputed_val_dataset.py : val_precomputed.csv + rir_catalog_val.csv を読み、rir_metaも付与:contentReference[oaicite:4]{index=4}
- dataset/audio_rir_dataset.py : collate_fn（IDsやrir_metaをテンソル/リストに整形）:contentReference[oaicite:5]{index=5}
"""

import argparse, json, os
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.delsa_model import DELSA                                  # :contentReference[oaicite:6]{index=6}
from dataset.precomputed_val_dataset import PrecomputedValDataset     # :contentReference[oaicite:7]{index=7}
from dataset.audio_rir_dataset import collate_fn                      # :contentReference[oaicite:8]{index=8}


def l2norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)

@torch.no_grad()
def run_and_cache(
    config_path: str,
    ckpt_path: str,
    cache_out: Path,
    device: str = "cuda",
    batch_size: int = 64,
    num_workers: int = 4,
):
    # ---- 設定の読み込み（最小限: val_precomp_root / val_index_csv / rir_csv_val）----
    import yaml
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    root = cfg.get("test_precomp_root")
    index_csv = cfg.get("test_index_csv") or (str(Path(root) / "test_precomputed.csv") if root else None)
    rir_csv   = cfg["rir_csv_val"]
    assert index_csv and Path(index_csv).exists(), f"index_csv not found: {index_csv}"
    assert Path(rir_csv).exists(), f"rir_csv_val not found: {rir_csv}"

    # ---- DataLoader（前計算val: i_act/i_rea/omni_48k + caption + rir_meta を返す）----
    ds = PrecomputedValDataset(index_csv=index_csv, rir_meta_csv=rir_csv, root=root)     # :contentReference[oaicite:9]{index=9}
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    # ---- Model & ckpt ----
    device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    model = DELSA(audio_encoder_cfg={}, text_encoder_cfg={}).to(device)                  # 4埋め込み:contentReference[oaicite:10]{index=10}
    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict) and any(k in sd for k in ("model", "state_dict", "model_state_dict")):
        for k in ("model", "state_dict", "model_state_dict"):
            if k in sd:
                sd = sd[k]
                break
    # module. を剥がす
    sd = { (k.replace("module.", "", 1) if k.startswith("module.") else k): v for k, v in sd.items() }
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] ckpt={ckpt_path}  missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    # ---- バッファ ----
    A_SPA, A_SRC, T_SPA, T_SRC = [], [], [], []
    SPA_ID, SRC_ID = [], []
    RIR_META: Dict[str, list] = {}

    for batch in dl:
        if batch is None:
            continue

        # audio を device へ、text は list[str] のまま
        audio = {
            "i_act":   torch.stack([d["i_act"]   for d in batch["audio"]]).to(device),
            "i_rea":   torch.stack([d["i_rea"]   for d in batch["audio"]]).to(device),
            "omni_48k":torch.stack([d["omni_48k"]for d in batch["audio"]]).to(device),
        }
        texts = batch["texts"]

        out = model(audio, texts)  # GPUでforward→4埋め込みを得る:contentReference[oaicite:11]{index=11}

        # 4種の埋め込み（L2正規化して保持）
        A_SPA.append(l2norm(out["audio_space_emb"]).detach().cpu())
        A_SRC.append(l2norm(out["audio_source_emb"]).detach().cpu())
        T_SPA.append(l2norm(out["text_space_emb"]).detach().cpu())
        T_SRC.append(l2norm(out["text_source_emb"]).detach().cpu())

        # ラベル（collate_fnで [B,1] → vstack 済み）:contentReference[oaicite:12]{index=12}
        SPA_ID.append(batch["space_id"].reshape(-1).cpu())
        SRC_ID.append(batch["source_id"].reshape(-1).cpu())

        # rir_meta 全列をテンソル化してくれているので key ごとに append:contentReference[oaicite:13]{index=13}
        meta = batch["rir_meta"]  # dict: key -> tensor/list
        for k, v in meta.items():
            if k not in RIR_META:
                RIR_META[k] = []
            # Tensor は CPU に寄せてから保持
            if isinstance(v, torch.Tensor):
                RIR_META[k].append(v.detach().cpu())
            else:
                # 文字列などは list のまま
                RIR_META[k] += v

    # ---- 結合 ----
    A_SPA = torch.cat(A_SPA, dim=0)  # [N,256]
    A_SRC = torch.cat(A_SRC, dim=0)  # [N,512]
    T_SPA = torch.cat(T_SPA, dim=0)  # [N,256]
    T_SRC = torch.cat(T_SRC, dim=0)  # [N,512]
    SPA_ID = torch.cat(SPA_ID, dim=0).long()  # [N]
    SRC_ID = torch.cat(SRC_ID, dim=0).long()  # [N]

    # rir_meta の各 key も結合（Tensor だけ）
    RIR_META_STACKED: Dict[str, Any] = {}
    for k, chunks in RIR_META.items():
        if len(chunks) == 0:
            continue
        if isinstance(chunks[0], torch.Tensor):
            RIR_META_STACKED[k] = torch.cat(chunks, dim=0)
        else:
            # 文字列などはそのまま
            RIR_META_STACKED[k] = chunks

    # ---- 保存 ----
    cache_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeddings": {
                "audio_space": A_SPA, "audio_source": A_SRC,
                "text_space": T_SPA,  "text_source": T_SRC,
            },
            "labels": {
                "space_id": SPA_ID, "source_id": SRC_ID,
            },
            "rir_meta": RIR_META_STACKED,   # azimuth_deg/elevation_deg/distance/area_m2/fullband_T30_ms/direction_vec 等:contentReference[oaicite:14]{index=14}
            "meta": {
                "config": str(config_path),
                "ckpt_path": str(ckpt_path),
                "index_csv": str(index_csv),
                "rir_csv": str(rir_csv),
            },
        },
        cache_out,
    )
    print(f"[✓] saved cache: {cache_out}  "
          f"(N={A_SPA.size(0)}, dims: a_spa={A_SPA.size(1)}, a_src={A_SRC.size(1)}, t_spa={T_SPA.size(1)}, t_src={T_SRC.size(1)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="config.yaml（val_precomp_root / val_index_csv / rir_csv_val を参照）")
    ap.add_argument("--ckpt_dir", default="Spatial_AudioCaps/takamichi09/checkpoints_delsa_fromELSA",
                    help="ckpt_sup_ep{epoch}.pt が入っているディレクトリ")
    ap.add_argument("--epoch", type=int, help="評価するエポック番号（ckpt_sup_ep{epoch}.pt を使う）")
    ap.add_argument("--ckpt", type=str, default=None, help="ckpt を直接指定（--epoch より優先）")
    ap.add_argument("--out_dir", default="embed_cache", help="出力先ディレクトリ")
    ap.add_argument("--device", default="cuda", help="cuda / cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    if args.ckpt is not None:
        ckpt_path = args.ckpt
        ep_str = Path(ckpt_path).stem
    else:
        assert args.epoch is not None, "--epoch か --ckpt のどちらかを指定してください"
        ckpt_path = str(Path(args.ckpt_dir) / f"ckpt_sup_ep{args.epoch}.pt")
        ep_str = f"ep{args.epoch}"
    assert Path(ckpt_path).exists(), f"ckpt not found: {ckpt_path}"

    out_name = f"embeds_{ep_str}.pt"
    cache_out = Path(args.out_dir) / out_name

    run_and_cache(
        config_path=args.config,
        ckpt_path=ckpt_path,
        cache_out=cache_out,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    import argparse
    main()
