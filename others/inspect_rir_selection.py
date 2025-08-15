#!/usr/bin/env python3
"""
inspect_rir_selection.py

選択されたRIRの一覧と分布を確認するための簡易レポータ。
入力は以下のどちらでも可：
  - make_val_fixed.py が出力した rir_fixed.csv
  - make_val_shards.py が出力した rir_catalog_val.shard*.csv
  - precompute_val.py が出力した val_precomputed.csv（この場合も 'rir_path' 列が必要）

出力：
  out_dir/
    rir_list.csv             … 選択されたRIRの一覧（rir_pathのみ）
    by_room.csv              … room_idごとの件数（存在する場合）
    by_az_sector.csv         … 方位セクタ別の件数（azimuth_degがある場合）
    by_elv_bin.csv           … 仰角ビン別の件数（elevation_degがある場合）
    stats_numeric.csv        … 数値列（T30, distance, area 等）の要約統計
    rir_selection_report.md  … 上記のまとめ

使い方:
  python inspect_rir_selection.py --csv val_fixed_400x24/rir_fixed.csv --out val_fixed_400x24/report
"""
import argparse
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


def _az_sector(az_deg: pd.Series, n: int = 8):
    a = (az_deg.astype(float) + 360.0) % 360.0
    w = 360.0 / n
    return np.clip(np.floor(a / w).astype(int), 0, n - 1)


def _elv_bin(elv_deg: pd.Series):
    e = elv_deg.astype(float)
    # [-90,90] → 3ビン（低/中/高）
    return pd.cut(e, bins=[-91, -15, 15, 91], labels=["low","mid","high"], include_lowest=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    df = _read_csv(args.csv)
    if "rir_path" not in df.columns:
        raise SystemExit("入力CSVに 'rir_path' 列が必要です。")

    # 1) RIR一覧
    rir_list = df[["rir_path"]].drop_duplicates().reset_index(drop=True)
    rir_list.to_csv(out/"rir_list.csv", index=False)

    # 2) room_id 分布
    if "room_id" in df.columns:
        by_room = df.groupby("room_id")["rir_path"].nunique().reset_index(name="count").sort_values("count", ascending=False)
        by_room.to_csv(out/"by_room.csv", index=False)
    else:
        by_room = None

    # 3) 角度分布
    by_az = None; by_elv = None
    if "azimuth_deg" in df.columns:
        sector = _az_sector(df["azimuth_deg"])
        by_az = pd.Series(sector).value_counts().sort_index().rename_axis("az_sector").reset_index(name="count")
        by_az.to_csv(out/"by_az_sector.csv", index=False)
    if "elevation_deg" in df.columns:
        elv = _elv_bin(df["elevation_deg"])
        by_elv = elv.value_counts().sort_index().rename_axis("elv_bin").reset_index(name="count")
        by_elv.to_csv(out/"by_elv_bin.csv", index=False)

    # 4) 数値列の要約
    numeric_cols = [c for c in ["fullband_T30_ms","source_distance_m","area_m2"] if c in df.columns]
    stats = None
    if numeric_cols:
        stats = df[numeric_cols].describe(percentiles=[0.1,0.25,0.5,0.75,0.9]).T
        stats.to_csv(out/"stats_numeric.csv")

    # 5) Markdown レポート
    lines = []
    lines.append(f"# RIR Selection Report\n")
    lines.append(f"- Input CSV: `{Path(args.csv).resolve()}`")
    lines.append(f"- Total unique RIR: {len(rir_list)}")
    lines.append("")
    lines.append("## Preview of RIRs")
    head = "\n".join([f"- {p}" for p in rir_list["rir_path"].head(20).tolist()])
    lines.append(head if len(head) else "(no rows)")
    lines.append("")
    if by_room is not None:
        lines.append("## Count by Room")
        lines.append(by_room.to_markdown(index=False))
        lines.append("")
    if by_az is not None:
        lines.append("## Count by Azimuth Sector (8 sectors)")
        lines.append(by_az.to_markdown(index=False))
        lines.append("")
    if by_elv is not None:
        lines.append("## Count by Elevation Bin (low/mid/high)")
        lines.append(by_elv.to_markdown(index=False))
        lines.append("")
    if stats is not None:
        lines.append("## Numeric Stats (T30, distance, area)")
        lines.append(stats.to_markdown())
        lines.append("")

    (out/"rir_selection_report.md").write_text("\n".join(lines), encoding="utf-8")

    print("[OK] Wrote report to:", out.resolve())
    print((out/"rir_selection_report.md").read_text())


if __name__ == "__main__":
    main()
