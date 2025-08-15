# preview_captions.py

import random
from dataset.audio_rir_dataset import AudioRIRDataset, collate_fn

def main():
    # instantiate exactly as in train
    ds = AudioRIRDataset(
        csv_audio="AudioCaps_csv/train.csv",
        base_dir="AudioCaps_mp3",
        csv_rir="RIR_dataset/rir_catalog_train.csv",
        n_views=1,   # only need one view to inspect
        split="train"
    )

    print(f"Dataset size: {len(ds)} samples\n")
    # sample 5 random indices
    for idx in random.sample(range(len(ds)), 5):
        sample = ds[idx]
        # sample["texts"] is a list of length n_views; we set n_views=1
        rewritten = sample["texts"][0]
        # original caption from the raw dataframe
        orig = ds.audio_df.loc[ds.audio_df.index[idx], "caption"]
        print(f"--- Sample {idx} ---")
        print(f"Original:  {orig}")
        print(f"Rewritten: {rewritten}")
        print()

if __name__ == "__main__":
    main()
