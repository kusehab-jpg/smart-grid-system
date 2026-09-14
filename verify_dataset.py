"""
verify_dataset.py
=================
Run this FIRST to confirm your classData.csv is the correct Kaggle dataset
and to print the exact column structure and class counts.

Usage:
    python verify_dataset.py --data classData.csv
"""
import sys, hashlib, argparse
import numpy as np
import pandas as pd
from pathlib import Path

def main(csv_path):
    if not Path(csv_path).exists():
        print(f"File not found: {csv_path}")
        sys.exit(1)

    # SHA-256 checksum
    sha = hashlib.sha256()
    with open(csv_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    print(f"\n{'='*60}")
    print("DATASET VERIFICATION REPORT")
    print(f"{'='*60}")
    print(f"File       : {csv_path}")
    print(f"SHA-256    : {sha.hexdigest()}")
    print("(Add this checksum to Appendix A of your thesis.)")

    df = pd.read_csv(csv_path)
    print(f"\nShape      : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Columns    : {list(df.columns)}")
    print(f"\nFirst 3 rows:\n{df.head(3).to_string()}")
    print(f"\nData types:\n{df.dtypes.to_string()}")
    print(f"\nMissing values: {df.isnull().sum().sum()} total")

    # Detect GCBA columns and map to fault types
    gcba_map = {(0,0,0,0):("Normal",0),(1,0,0,1):("LG",1),(0,0,1,1):("LL",2),
                (1,0,1,1):("LLG",3),(0,1,1,1):("LLL",4),(1,1,1,1):("LLLG",5)}
    if all(c in df.columns for c in ["G","C","B","A"]):
        print("\nFault class distribution (from G,C,B,A columns):")
        counts = df.groupby(["G","C","B","A"]).size().reset_index(name="count")
        total = 0
        for _, row in counts.iterrows():
            key = (int(row.G),int(row.C),int(row.B),int(row.A))
            label, idx = gcba_map.get(key, (f"Unknown{key}", -1))
            pct = row["count"] / len(df) * 100
            print(f"  {key} → Class {idx} ({label:8s}): {row['count']:,}  ({pct:.1f}%)")
            total += row["count"]
        print(f"  Total: {total:,}")
        print("\n")
    else:
        print("\nNote: G,C,B,A columns not found. Check column names in your CSV.")

    print(f"\n{'='*60}")
    print("THESIS CITATION (APA 7) — add to §3.3:")
    print("  Sathyaprakash, E. (2021). Electrical fault detection and")
    print("  classification [Dataset]. Kaggle.")
    print("  https://www.kaggle.com/datasets/esathyaprakash/")
    print("  electrical-fault-detection-and-classification")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="classData.csv")
    args = parser.parse_args()
    main(args.data)
