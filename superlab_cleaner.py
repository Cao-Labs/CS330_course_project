import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd

def parse_superlab_file_to_tidy(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    expt_name = None
    for line in lines[:10]:
        if line.startswith(".Experiment Name"):
            parts = line.split("\t")
            if len(parts) > 1:
                expt_name = parts[1].strip()

    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Participant\tParticipant\tSession"):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(f"No table header found in {path.name}")

    tsv = "\n".join(lines[header_idx:])
    df = pd.read_csv(
        pd.io.common.StringIO(tsv),
        sep="\t",
        dtype=str
    )

    df = df[df["Participant"] != "Name"].copy()

    df["__source_file"] = path.name
    df["__expt_name"] = expt_name if expt_name is not None else ""

    return df

def build_cleaned_from_folder(indir: Path) -> pd.DataFrame:

    txt_files = sorted(
        p for p in indir.rglob("*.txt")
        if p.is_file() and not p.name.startswith("._")
    )

    if not txt_files:
        raise FileNotFoundError(f"No .txt files found under {indir}")

    all_dfs = []

    for path in txt_files:
        print(f"[INFO] Reading {path.name}")
        df_raw = parse_superlab_file_to_tidy(path)

        mask_pref = df_raw["Block"].str.contains("pref", case=False, na=False)
        df = df_raw[mask_pref].copy()

        if df.empty:
            print(f"[WARN] No 'pref' block rows in {path.name}; skipping.")
            continue

        def extract_stim(st: str):
            if not isinstance(st, str):
                return pd.NA
            m = re.search(r"\(\s*\d+,\s*([A-Za-z]+)\.wav\)", st)
            if m:
                return m.group(1)
            m2 = re.search(r"([A-Za-z]+)\.wav", st)
            return m2.group(1) if m2 else pd.NA

        df["stimuli_presented"] = df["Trial"].apply(extract_stim)

        df["response_rt"] = pd.to_numeric(df["Error"], errors="coerce")

        df["response_correct"] = df["Correct"].astype(str).str.upper().eq("C")

        df["Group"] = df["Participant"]    
        df["participant_id"] = df["Participant.1"]
        df["expt_id"] = df["__expt_name"]
        df["group_id"] = df["Group"]

        df["trial_template"] = df["Block"]
        df["trial_duration"] = 1500  
        df["Pattern"] = "Devoicing"  
        df["PartBlocks"] = 1
        df["AllBlocks"] = 1

        df["PB"] = df["Reaction"].astype(str).str.upper()
        df["Part"] = 1
        df["Correct"] = df["response_correct"].astype(int)

        df["Cumulative_num"] = pd.to_numeric(df["Cumulative"], errors="coerce")
        df = df.sort_values(["participant_id", "Cumulative_num"])

        df["Trial"] = df.groupby("participant_id").cumcount() + 1
        df["trial_num"] = df["Trial"]

        def invrt_series(s: pd.Series) -> pd.Series:
            mu = s.mean(skipna=True)
            sd = s.std(ddof=0, skipna=True)
            if sd and sd > 0:
                return -(s - mu) / sd
            else:
                return pd.Series([np.nan] * len(s), index=s.index)

        df["invRT"] = df.groupby("participant_id")["response_rt"].transform(invrt_series)

        def classify_final(st):
            if not isinstance(st, str) or not st:
                return pd.NA
            return "Final" if st[-1].lower() in ("p", "b") else "NonFinal"

        df["Final"] = df["stimuli_presented"].apply(classify_final)

        df["response_name"] = df["Reaction"]

        all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError("No usable 'pref' trials found in any file.")

    cleaned = pd.concat(all_dfs, ignore_index=True)

    final_cols = [
        "expt_id", "Group", "group_id", "invRT", "Trial", "participant_id",
        "response_correct", "response_name", "response_rt", "stimuli_presented",
        "trial_template", "trial_duration", "PartBlocks", "AllBlocks", "Pattern",
        "trial_num", "PB", "Final", "Part", "Correct",
    ]

    cleaned = cleaned[final_cols]

    return cleaned


def main():
    ap = argparse.ArgumentParser(
        description="Clean SuperLab logs into final_cleaned.csv with RT, PB, Final, etc."
    )
    ap.add_argument(
        "--indir",
        required=True,
        help="Folder containing the SuperLab .txt logs (e.g. C:\\Users\\Bodhi\\PilotData)"
    )
    ap.add_argument(
        "--outcsv",
        required=True,
        help="Full path to output CSV (e.g. C:\\Users\\Bodhi\\out\\final_cleaned.csv)"
    )

    args = ap.parse_args()

    indir = Path(args.indir)
    outcsv = Path(args.outcsv)

    if not indir.exists() or not indir.is_dir():
        raise FileNotFoundError(f"Input folder does not exist or is not a directory: {indir}")

    outcsv.parent.mkdir(parents=True, exist_ok=True)

    cleaned = build_cleaned_from_folder(indir)

    print(f"[INFO] Final cleaned shape: {cleaned.shape[0]} rows x {cleaned.shape[1]} columns")
    print("[INFO] Rows per participant_id:")
    print(cleaned["participant_id"].value_counts())

    cleaned.to_csv(outcsv, index=False)
    print(f"[OK] Saved cleaned CSV to: {outcsv}")


if __name__ == "__main__":
    main()
