#!/usr/bin/env python3
# python3 create_csv_for_rir.py --split val (valのRIR_datasetに関するcsvを作成)
import argparse
import csv
import yaml
import math
from pathlib import Path

def center_dist(dims, src):
    cx, cy, cz = [d/2 for d in dims]
    return round(math.dist((cx, cy, cz), src), 3)

def build_catalog(base_dir: Path, split: str):
    data_dir = base_dir / "RIR_data" / split
    meta_dir = base_dir / "RIR_meta" / split
    if not meta_dir.exists():
        print(f"[skip] no meta dir for split '{split}'")
        return

    rows = []
    for yml_path in sorted(meta_dir.glob("*.yml")):
        rid      = yml_path.stem
        wav_path = data_dir / f"{rid}.wav"
        if not wav_path.exists():
            continue

        meta = yaml.safe_load(yml_path.read_text())
        rows.append({
            "rir_path": str(wav_path),
            "dims": meta["dims"],
            "area_m2": meta["area_m2"],
            "alpha": meta["alpha"],
            "fullband_T30_ms": meta["fullband_T30_ms"],
            "source_distance_m": meta["source_distance_m"],
            "azimuth_deg": meta["azimuth_deg"],
            "elevation_deg": meta["elevation_deg"],
            "source_pos_xyz": meta["source_pos_xyz"],
            "distance": center_dist(meta["dims"], meta["source_pos_xyz"]),
            "fs": meta["fs"],
            "room_id": meta["room_id"],
            "split": split,
        })

    if rows:
        out_csv = base_dir / f"rir_catalog_{split}.csv"
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {out_csv} ({len(rows)} rows)")
    else:
        print(f"[empty] no entries for split '{split}'")

def main():
    parser = argparse.ArgumentParser(
        description="Build RIR catalog CSV for one or more splits"
    )
    parser.add_argument(
        "--base", "-b",
        type=Path,
        default=Path("RIR_dataset"),
        help="Base directory containing RIR_data/ and RIR_meta/"
    )
    parser.add_argument(
        "--split", "-s",
        choices=["train", "val", "test", "all"],
        default="all",
        help="Which split to process (default: all)"
    )
    args = parser.parse_args()

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for sp in splits:
        build_catalog(args.base, sp)

if __name__ == "__main__":
    main()
