#!/usr/bin/env python3
# scripts/precompute_val.py
# Dry×RIR を事前畳み込みして FOA/IV を保存し、val_precomputed.csv を作成
# 進捗は tqdm の単一バーで表示（総ビュー数 = Audio件数 × n_views）

import argparse
from pathlib import Path
import sys
import random
import yaml
import torch
import torchaudio
import pandas as pd
from tqdm.auto import tqdm

from dataset.audio_rir_dataset import AudioRIRDataset, foa_to_iv, rewrite_caption

# 再現性
random.seed(42)
torch.manual_seed(42)

def _select_device(raw: str | None) -> str:
    if raw is None or raw.lower() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return raw

def load_config(path: str | None = None) -> dict:
    """Load hyper-parameters from a YAML file and fill in sane defaults."""
    # 1) YAML パス決定（CLIの第1引数が .yml/.yaml ならそれを優先）
    if path is None:
        if len(sys.argv) > 1 and sys.argv[1].endswith((".yml", ".yaml")):
            path = sys.argv[1]
        else:
            path = "config.yaml"

    # 2) 読み込み
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        cfg = {}

    # 3) デフォルト値
    defaults = {
        "split": "train",
        "batch_size": 8,
        "n_views": 24,  # 前計算で使う RIR 本数（<=0 なら全部）
        "epochs": 5,
        "lr": 0.0001,
        "device": "auto",
        "wandb": True,
        "proj": "delsa-sup-contrast",
        "run_name": None,
        "audio_csv_train": "AudioCaps_csv/train.csv",
        "rir_csv_train": "RIR_dataset/rir_catalog_train.csv",
        "audio_csv_val": "AudioCaps_csv/val.csv",
        "rir_csv_val": "RIR_dataset/rir_catalog_val.csv",
        "base_dir": "RIR_dataset",
        "audio_base": None,
        "out_dir": "data/val_precomputed",
    }
    for k, v in defaults.items():
        cfg.setdefault(k, v)

    # 4) 後処理
    cfg["device"] = _select_device(cfg["device"])
    return cfg

def main(args: argparse.Namespace):
    cfg = load_config()
    # CLI で上書き
    if args.csv_audio:
        cfg["audio_csv_val"] = args.csv_audio
    if args.csv_rir:
        cfg["rir_csv_val"] = args.csv_rir
    if args.base_dir:
        cfg["audio_base"] = args.base_dir
    cfg["out_dir"] = args.out_dir or cfg.get("out_dir", "data/val_precomputed")

    # n_views（<=0 なら「全部」）
    n_views_cfg = (
        args.n_views
        if args.n_views is not None
        else cfg.get("val_n_views", cfg.get("n_views", 1))
    )
    # Dataset 準備（学習用の挙動は使わないが整合のため値は渡す）
    ds = AudioRIRDataset(
        csv_audio=cfg["audio_csv_val"],
        base_dir=cfg["audio_base"],
        csv_rir=cfg["rir_csv_val"],
        split="val",
        n_views=max(1, n_views_cfg if isinstance(n_views_cfg, int) else 1),
        share_rir=False,
        batch_size=None,
    )

    out_root = Path(cfg["out_dir"])
    (out_root / "foa").mkdir(parents=True, exist_ok=True)
    (out_root / "feat").mkdir(parents=True, exist_ok=True)

    records: list[dict] = []

    # RIR の決定論的順序
    rir_list = sorted(ds.rir_paths)
    n_views = len(rir_list) if (isinstance(n_views_cfg, int) and n_views_cfg <= 0) else min(int(n_views_cfg), len(rir_list))

    torch.set_grad_enabled(False)
    total_views = len(ds) * n_views
    # tqdm開始前に追加
    print(f"[INFO] Audio clips={len(ds)} | RIR candidates={len(rir_list)} | "
      f"n_views(effective)={n_views} | total views={len(ds)*n_views}")
    pbar = tqdm(total=total_views, desc="Precomputing FOA/IV", unit="view")

    with torch.no_grad():
        for idx in range(len(ds)):
            row = ds.audio_df.iloc[idx]
            dry = ds._load_dry(row["audiocap_id"])  # (T,)
            for j, rir_path in enumerate(rir_list[:n_views]):
                # 4ch FOA を合成（48k）
                foa = ds._apply_rir(dry, rir_path)  # (4, T_48k)

                # FOA 48k を保存（16-bit PCM）
                key = f"{row.audiocap_id}_{Path(rir_path).stem}"
                torchaudio.save(
                    out_root / f"foa/{key}.wav",
                    foa,
                    48_000,
                    format="wav",
                    encoding="PCM_S",
                    bits_per_sample=16,
                )

                # FOA→16k へリサンプルして IV 特徴を作成
                foa16 = torchaudio.functional.resample(foa, 48_000, 16_000)  # (4, T_16k)
                i_act, i_rea = foa_to_iv(foa16.unsqueeze(0))  # (1,3,F,T) ×2
                i_act, i_rea = i_act.squeeze(0), i_rea.squeeze(0)  # (3,F,T)

                # 特徴保存
                torch.save({"i_act": i_act, "i_rea": i_rea}, out_root / f"feat/{key}.pt")

                # メタ→キャプション
                meta = ds.rir_meta[rir_path].copy()
                caption = rewrite_caption(row["caption"], meta)

                records.append(
                    {
                        "audiocap_id": row.audiocap_id,
                        "rir_path": rir_path,
                        "foa_path": f"foa/{key}.wav",
                        "feat_path": f"feat/{key}.pt",
                        "caption": caption,
                    }
                )

                # 進捗
                if (j % 4) == 0:
                    pbar.set_postfix({"clip": int(row["audiocap_id"]), "RIR": Path(rir_path).stem})
                pbar.update(1)

    pbar.close()

    # 索引CSV
    df = pd.DataFrame(records)
    df.to_csv(out_root / "val_precomputed.csv", index=False)

    # 簡易サマリ
    print(f"[OK] Wrote {len(df)} views to {out_root.resolve()}")
    print(f" - Audio clips: {len(ds)}")
    print(f" - RIR used per clip: {n_views}")
    print(f" - Output CSV: {out_root / 'val_precomputed.csv'}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="YAML path (default: config.yaml)")
    p.add_argument("--csv_audio", help="Validation audio CSV (e.g., AudioCaps_csv/val.csv)")
    p.add_argument("--base_dir", help="Base directory for audio files")
    p.add_argument("--csv_rir", help="Validation RIR catalog CSV (must include 'rir_path')")
    p.add_argument("--out_dir", help="Output root (default: data/val_precomputed)")
    p.add_argument("--n_views", type=int, help="<=0 なら RIR 全部を使用（デフォルトは config.yaml の n_views）")
    main(p.parse_args())
