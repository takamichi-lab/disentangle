# scripts/compute_stats.py
"""実行方法
(.venv) takamichi-lab-pc09@takamichi-lab-pc09:~/DELSA/RIR_dataset$ python3 compute_stats.py --csv rir_catalog_train.csv
Saved → /home/takamichi-lab-pc09/DELSA/RIR_dataset/stats.pt
 area    : mean=145.4749, std=76.0188
 distance: mean=1.9240, std=0.9213
 t30     : mean=1297.8711, std=738.9759
"""
import pandas as pd
import torch
from pathlib import Path

def compute_mean_std(series):
    return float(series.mean()), float(series.std())

def main(csv_path: str, out_path: str = "stats.pt"):
    df = pd.read_csv(csv_path)
    area_mean, area_std = compute_mean_std(df["area_m2"])
    dist_mean, dist_std = compute_mean_std(df["distance"])
    t30_mean, t30_std = compute_mean_std(df["fullband_T30_ms"])

    torch.save(
        {
            "area_m2": {"mean": area_mean, "std": area_std},
            "distance": {"mean": dist_mean, "std": dist_std},
            "fullband_T30_ms": {"mean": t30_mean, "std": t30_std}
        },
        out_path,
    )
    print(f"Saved → {Path(out_path).resolve()}")
    print(f" area    : mean={area_mean:.4f}, std={area_std:.4f}")
    print(f" distance: mean={dist_mean:.4f}, std={dist_std:.4f}")
    print(f" t30     : mean={t30_mean:.4f}, std={t30_std:.4f}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="訓練メタデータ CSV のパス")
    p.add_argument("--out", default="stats.pt")
    args = p.parse_args()
    main(args.csv, args.out)
