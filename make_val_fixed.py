#!/usr/bin/env python3
"""
make_val_fixed.py

毎エポック同一の Validation セットを作るスクリプト（決定論的）
- AudioCaps から N 本を固定抽出
- RIR カタログから M 本を層化・固定抽出
- 両者の直積インデックス pairs_fixed.csv を出力

出力：
  out_dir/
    audio_fixed.csv
    rir_fixed.csv
    pairs_fixed.csv        … audio × RIR の直積（事前計算やオンザフライ評価のインデックスに）
    summary.txt

使い方（例）：
  python make_val_fixed.py \
    --audio_csv AudioCaps_csv/val.csv \
    --rir_csv   RIR_dataset/rir_catalog_val.csv \
    --out_dir   val_fixed_96x96 \
    --audio_n   100 \
    --rir_n     96 \
    --seed      42

その後：
  1) 事前計算する場合：
     python precompute_val.py \
       --csv_audio val_fixed_96x96/audio_fixed.csv \
       --csv_rir   val_fixed_96x96/rir_fixed.csv \
       --out_dir   data/val_precomp_fixed \
       --n_views   0  # (スクリプト側で「0=全部」と解釈するようにしている場合)

  2) オンザフライ評価の場合：pairs_fixed.csv を読む簡易Datasetを使う
"""
import argparse
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd


def _read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")
    except Exception:
        return pd.read_csv(path, encoding="cp932")


def _qbin(series: pd.Series, q: int, name: str):
    """重複binに強いqcutラッパ（重複時はcutへフォールバック）"""
    s = series.astype(float)
    try:
        b = pd.qcut(s, q=q, labels=False, duplicates="drop")
    except ValueError:
        b = pd.cut(s, bins=q, labels=False, include_lowest=True, duplicates="drop")
    return pd.Series(b, index=series.index, name=name).astype("Int64")


def _sector(angle_deg: pd.Series, n_sectors: int = 8, name: str = "az_sector"):
    """[0,360) に正規化して n 分割"""
    a = angle_deg.astype(float).copy()
    a = (a + 360.0) % 360.0
    w = 360.0 / n_sectors
    idx = np.floor(a / w).astype(int)
    idx = np.clip(idx, 0, n_sectors - 1)
    return pd.Series(idx, index=angle_deg.index, name=name).astype("Int64")


def build_strata(rir_df: pd.DataFrame) -> pd.Series:
    """RIRの層キーを作る。列が無い場合は既定値で埋める"""
    df = rir_df.copy()
    get = lambda col, default: df[col] if col in df.columns else pd.Series(default, index=df.index)

    df["t30_bin"]   = _qbin(get("fullband_T30_ms", 0.0), q=3, name="t30_bin")
    df["dist_bin"]  = _qbin(get("source_distance_m", 0.0), q=3, name="dist_bin")
    df["az_sector"] = _sector(get("azimuth_deg", 0.0), n_sectors=8, name="az_sector")
    elev = get("elevation_deg", 0.0).astype(float)
    df["elv_bin"]   = pd.cut(elev, bins=[-91, -15, 15, 91], labels=False, include_lowest=True).astype("Int64")
    if "room_id" not in df.columns:
        df["room_id"] = "room_unk"

    strata_cols = ["room_id", "t30_bin", "dist_bin", "az_sector", "elv_bin"]
    return df[strata_cols].astype(str).agg("|".join, axis=1)


def stratified_sample(df: pd.DataFrame, n: int, seed: int, key_col: str) -> pd.DataFrame:
    """
    層化サンプリングで df から n 行を決定論的に抽出。
    - 層ごとの保有比率に応じて quota を割り当て、端数は大きい残差から配分。
    - 層内は seed でシャッフルして先頭 quota を採用。
    """
    if n <= 0 or n >= len(df):
        # n が範囲外なら全件
        return df.copy().reset_index(drop=True)

    rng = random.Random(seed)
    strata = build_strata(df)
    df2 = df.copy()
    df2["_stratum"] = strata.values

    # 層ごとの比率から quota を決定（floor）
    group_sizes = df2.groupby("_stratum")[key_col].size().rename("size").reset_index()
    group_sizes["prop"] = group_sizes["size"] / group_sizes["size"].sum()
    group_sizes["quota"] = np.floor(group_sizes["prop"] * n).astype(int)

    # 端数を残差の大きい順に配分
    allocated = int(group_sizes["quota"].sum())
    residuals = (group_sizes["prop"] * n) - group_sizes["quota"]
    order = np.argsort(-residuals.values)
    k = n - allocated
    for idx in order[:k]:
        group_sizes.at[idx, "quota"] += 1

    # 各層から quota だけ抽出
    chosen_idx = []
    for _, row in group_sizes.iterrows():
        stratum = row["_stratum"]
        quota = int(row["quota"])
        pool = df2[df2["_stratum"] == stratum]
        if quota <= 0:
            continue
        # 決定論的シャッフル：seed と stratum を用いて固定順序を作る
        local_rng = random.Random((hash(stratum) ^ seed) & 0xFFFFFFFF)
        idxs = list(pool.index)
        local_rng.shuffle(idxs)
        chosen_idx.extend(idxs[:quota])

    chosen = df2.loc[chosen_idx].drop(columns=["_stratum"]).reset_index(drop=True)
    return chosen


# ===== ここから：最小限の Audio 実ファイル存在チェック（引数追加なし） =====
def _infer_split_from_csv(path: str) -> str:
    """audio_csv のファイル名から train/val/test をゆるく推定（既定: val）"""
    name = Path(path).stem.lower()
    if "train" in name:
        return "train"
    if "test" in name:
        return "test"
    return "val"


def _resolve_audio_path_minimal(row: pd.Series, split: str):
    """
    余計な引数なしの最小解決ロジック:
      1) mp3_path 列があればそれを root=AudioCaps_mp3/{split} 基準で解決（絶対パスならそのまま）
      2) ytid があれば ytid.mp3
      3) audiocap_id があれば audiocap_id.mp3
      いずれもなければ None
    """
    root = Path("AudioCaps_mp3") / split

    # 1) mp3_path 列
    if "mp3_path" in row and not pd.isna(row["mp3_path"]):
        s = str(row["mp3_path"]).strip()
        if s:
            rel = Path(s)
            return rel if rel.is_absolute() else (root / rel)

    # 2) ytid
    if "ytid" in row and not pd.isna(row["ytid"]):
        y = str(row["ytid"]).strip()
        if y and y.lower() != "nan":
            return root / f"{y}.mp3"

    # 3) audiocap_id
    if "audiocap_id" in row and not pd.isna(row["audiocap_id"]):
        a = str(row["audiocap_id"]).strip()
        if a and a.lower() != "nan":
            return root / f"{a}.mp3"

    return None
# ===== ここまで：最小限の存在チェック =====


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_csv", required=True)
    ap.add_argument("--rir_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--audio_n", type=int, default=400, help="抽出するAudioの件数（0 or 大きすぎる値なら全件）")
    ap.add_argument("--rir_n", type=int, default=24, help="抽出するRIRの件数（0 or 大きすぎる値なら全件）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    audio_df_all = _read_csv(args.audio_csv)
    rir_df = _read_csv(args.rir_csv)

    if "rir_path" not in rir_df.columns:
        raise SystemExit("RIR CSV に 'rir_path' 列が必要です。")

    # --- 最小限の Audio 実ファイル存在チェック（AudioCaps_mp3/{split} を基準） ---
    split = _infer_split_from_csv(args.audio_csv)
    keep_mask = []
    for _, row in audio_df_all.iterrows():
        p = _resolve_audio_path_minimal(row, split)
        keep_mask.append(p is not None and Path(p).exists())

    kept = int(np.sum(keep_mask))
    missing = len(keep_mask) - kept
    audio_df = audio_df_all.loc[keep_mask].reset_index(drop=True)

    if missing > 0:
        print(f"[INFO] Audio exists filter: kept={kept} / missing={missing}  (root=AudioCaps_mp3/{split})")
    if len(audio_df) == 0:
        raise SystemExit(f"有効なAudioファイルが1件も見つかりませんでした（AudioCaps_mp3/{split} を確認）。")

    # 1) 決定論的 Audio 抽出（存在チェック後に実施）
    if args.audio_n <= 0 or args.audio_n >= len(audio_df):
        audio_fixed = audio_df.copy()
    else:
        audio_fixed = audio_df.sample(n=args.audio_n, random_state=args.seed).sort_index()
    audio_fixed.to_csv(out / "audio_fixed.csv", index=False)

    # 2) 決定論的・層化 RIR 抽出
    if args.rir_n <= 0 or args.rir_n >= len(rir_df):
        rir_fixed = rir_df.copy().reset_index(drop=True)
    else:
        rir_fixed = stratified_sample(rir_df, n=args.rir_n, seed=args.seed, key_col="rir_path")
    rir_fixed.to_csv(out / "rir_fixed.csv", index=False)

    # 3) 直積（Audio × RIR）
    a = audio_fixed[["audiocap_id", "caption"]].copy() if "audiocap_id" in audio_fixed.columns else audio_fixed.copy()
    a["__key"] = 1
    r = rir_fixed[["rir_path"]].copy()
    r["__key"] = 1
    pairs = a.merge(r, on="__key").drop(columns="__key")
    pairs.to_csv(out / "pairs_fixed.csv", index=False)

    # 4) サマリ
    with open(out / "summary.txt", "w", encoding="utf-8") as f:
        # 「Audio total」は CSV の総数を基準に表示（実ファイル存在チェック前）
        f.write(f"Audio total: {len(audio_df_all)}; fixed: {len(audio_fixed)}\n")
        f.write(f"RIR total: {len(rir_df)}; fixed: {len(rir_fixed)}\n")
        f.write(f"Pairs fixed: {len(pairs)}\n")

    print(f"[OK] Wrote fixed validation set to: {out.resolve()}")
    print((out / "summary.txt").read_text())


if __name__ == "__main__":
    main()
