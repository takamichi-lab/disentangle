#!/usr/bin/env python3
# check_dataset.py
import argparse, random, torch, torchaudio, pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path

# ------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser("Quick-check for AudioRIRDataset")
    ap.add_argument("--audio_csv",  required=True, help="audio_manifest_train.csv")
    ap.add_argument("--rir_csv",    required=True, help="rir_catalog_train.csv")
    ap.add_argument("--n_views",    type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--audio_base")
    ap.add_argument("--split", choices=["train", "valid", "test"], default="train")
    return ap.parse_args()

# ------------------------------------------------------------
def main():
    args = parse_args()

    # 遅延 import で OK
    from dataset import AudioRIRDataset, collate_fn

    # ---------- Dataset ----------
    ds = AudioRIRDataset(
        csv_audio=args.audio_csv,
        csv_rir  =args.rir_csv,
        n_views  =args.n_views,
        base_dir = args.audio_base,
        split    =args.split
    )
    audio_rows = len(pd.read_csv(args.audio_csv))
    #assert len(ds) == audio_rows, f"len(dataset)={len(ds)} ≠ rows({audio_rows})"
    print(f"[OK] __len__  -> {len(ds)}")

    # ---------- __getitem__ ----------
    sample = ds[random.randrange(len(ds))]
    assert len(sample["waves"]) == args.n_views, "n_views mismatch"
    w = sample["waves"][0]
    assert w.ndim == 2 and w.shape[0] == 4, "waveform should be [4,T]"
    print(f"[OK] __getitem__  -> waves[0] shape {tuple(w.shape)}, text='{sample['texts'][0][:40]}…'")

    # ---------- DataLoader ----------
    dl = DataLoader(ds,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=0,
                    collate_fn=collate_fn)
    batch = next(iter(dl))
    Bp, C, T = batch["waves"].shape          # B' × 4 × T
    expected_Bp = args.batch_size * args.n_views
    assert C == 4, "channel dim should be 4"
    assert Bp == expected_Bp, f"B'={Bp} ≠ batch_size*n_views({expected_Bp})"
    print(f"[OK] DataLoader   -> waves {tuple(batch['waves'].shape)}, "
          f"texts {len(batch['texts'])}, "
          f"src_ids {batch['source_id'].shape}, spa_ids {batch['space_id'].shape}")

    print("\n✨  All quick checks passed!")

# ------------------------------------------------------------
if __name__ == "__main__":
    main()
