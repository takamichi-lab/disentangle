#!/usr/bin/env python3
# check_rir_coverage.py
"""
val split の YAML メタを走査し、5 軸カテゴリ
(dir, elev, dist, size, reverb) の 96 通りがそろっているか検証します。

● 出力
  ✓ All 96 combos are present!        … 完全網羅
  ✗ Missing combos (n=X): [...]       … 欠けているタプルを列挙
  Duplicate keys (n=Y)                … 同一キーが複数回あれば警告
"""
from pathlib import Path
import yaml, collections

# ────────────── マッピング関数 ──────────────
def map_direction(az):
    if -35 <= az <= 35:   return 'front'
    if 55  <= az <= 125:  return 'right'
    if -125<= az <= -55:  return 'left'
    return 'back' if az >= 145 or az <= -145 else ''

def map_elevation(el): return 'up'   if el > 40  else 'down' if el < -40 else ''
def map_distance(d):   return 'near' if d < 1    else 'far'  if d > 2   else ''
def map_size(a):       return 'small' if a < 50  else 'large' if a > 100 else 'mid-sized'
def map_reverb(t30):   return 'acoustically dampened' if t30 < 200 else \
                        'highly reverberant' if t30 > 1000 else ''

# ────────────── 目標 96 タプル ──────────────
DIR  = ['front','right','left','back']
ELEV = ['up','down'];  DIST = ['near','far']
SIZE = ['small','mid-sized','large']
REV  = ['acoustically dampened','highly reverberant']
TARGET = {(d,e,di,s,r) for d in DIR for e in ELEV for di in DIST for s in SIZE for r in REV}

def main(base: Path = Path("RIR_dataset"), split="val"):
    meta_dir = base / "RIR_meta" / split
    if not meta_dir.is_dir():
        raise SystemExit(f"meta dir not found: {meta_dir}")

    combos = collections.Counter()
    for yml_path in meta_dir.glob("*.yml"):
        m = yaml.safe_load(yml_path.read_text())
        key = (
            map_direction(m["azimuth_deg"]),
            map_elevation(m["elevation_deg"]),
            map_distance(m["source_distance_m"]),
            map_size(m["area_m2"]),
            map_reverb(m["fullband_T30_ms"])
        )
        if '' in key:
            print(f"[warn] ambiguous combo skipped: {yml_path.name} → {key}")
            continue
        combos[key] += 1

    # ───── 結果表示 ─────
    missing = TARGET - combos.keys()
    dup     = {k:c for k,c in combos.items() if c > 1}
    if not missing:
        print("✓ All 96 combos are present!")
    else:
        print(f"✗ Missing combos (n={len(missing)}): {sorted(missing)}")
    if dup:
        print(f"Duplicate keys (n={len(dup)}):")
        for k,c in dup.items():
            print(f"  {k} × {c}")

if __name__ == "__main__":
    # 引数を使いたければ argparse を追加してください
    main()
