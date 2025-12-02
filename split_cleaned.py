import pandas as pd
import argparse
import math
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Split final_cleaned.csv into train/val/test by participant_id.")
    parser.add_argument("--csv", required=True, help="Path to final_cleaned.csv")
    parser.add_argument("--outdir", required=True, help="Output directory for splits")
    parser.add_argument("--val-size", type=float, default=0.25, help="Validation fraction (default: 0.15)")
    parser.add_argument("--test-size", type=float, default=0.25, help="Test fraction (default: 0.15)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if "participant_id" not in df.columns:
        raise ValueError("participant_id column not found in dataset!")

    participants = sorted(df["participant_id"].dropna().unique().tolist())
    n = len(participants)
    if n < 3:
        raise ValueError(f"Not enough participants ({n}) to split into train/val/test!")

    n_test = math.floor(args.test_size * n)
    n_val = math.floor(args.val_size * n)
    n_train = n - n_val - n_test

    train_parts = participants[:n_train]
    val_parts = participants[n_train:n_train + n_val]
    test_parts = participants[n_train + n_val:]

    train_df = df[df["participant_id"].isin(train_parts)]
    val_df = df[df["participant_id"].isin(val_parts)]
    test_df = df[df["participant_id"].isin(test_parts)]

    train_df.to_csv(outdir / "train.csv", index=False)
    val_df.to_csv(outdir / "val.csv", index=False)
    test_df.to_csv(outdir / "test.csv", index=False)

    print(f"Saved splits to {outdir}")
    print(f"Participants: {n} total → {len(train_parts)} train, {len(val_parts)} val, {len(test_parts)} test")

if __name__ == "__main__":
    main()
