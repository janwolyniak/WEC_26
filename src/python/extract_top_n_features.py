import argparse
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = ROOT / "artifacts" / "features"
MODELS_DIR = ROOT / "artifacts" / "models"

def extract(n):
    print(f"Extracting top {n} features...")
    # Load the importance csv from the full model run
    importance_df = pd.read_csv(MODELS_DIR / "shap_importance_final.csv")
    top_n_features = importance_df["feature"].head(n).tolist()

    METADATA_COLS = [
        "player_appearance_id", "player_id", "fixture_id", "date",
        "scored_after", "fold", "partition_role",
        "minute_out", "subbed", "jersey_number",
    ]

    # We need to keep metadata + top n + exposure (if present)
    for split in ["dev", "holdout"]:
        df = pd.read_csv(FEATURES_DIR / f"features_final_{split}.csv")
        cols_to_keep = []
        
        # Keep metadata
        for col in METADATA_COLS:
            if col in df.columns:
                cols_to_keep.append(col)
                
        # Strip prefixes from SHAP feature names to get original column names
        original_top_n = []
        for f in top_n_features:
            if "__" in f:
                original_top_n.append(f.split("__", 1)[1])
            else:
                original_top_n.append(f)
                
        for col in original_top_n:
            if col in df.columns and col not in cols_to_keep:
                cols_to_keep.append(col)
                
        if "exposure" in df.columns and "exposure" not in cols_to_keep:
            cols_to_keep.append("exposure")
            
        df_subset = df[cols_to_keep]
        out_path = FEATURES_DIR / f"features_final_top{n}_{split}.csv"
        df_subset.to_csv(out_path, index=False)
        print(f"Saved {split} top {n} subset with shape {df_subset.shape} to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    args = parser.parse_args()
    extract(args.n)