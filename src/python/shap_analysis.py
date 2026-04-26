import argparse
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import sys

# Ensure src/python is in path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src" / "python"))

# Import classes and functions from probability_calibration
try:
    from probability_calibration import (
        FullyCalibratedModel, PoissonXGBWrapper, 
        SplineCalibrator, BetaCalibrator, IsotonicCalibrator, SigmoidCalibrator,
        to_float, to_category
    )
except ImportError:
    # Fallback definitions if import fails (unlikely given sys.path update)
    pass

# Monkeypatch __main__ to allow joblib to load the model
import __main__
for name in ['FullyCalibratedModel', 'PoissonXGBWrapper', 'SplineCalibrator', 'BetaCalibrator', 'IsotonicCalibrator', 'SigmoidCalibrator', 'to_float', 'to_category']:
    if name in globals():
        setattr(__main__, name, globals()[name])

warnings.filterwarnings("ignore")

# --- Paths ---
FEATURES_DIR = ROOT / "artifacts" / "features"
MODELS_DIR = ROOT / "artifacts" / "models"
REPORTS_DIR = ROOT / "reports" / "figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Constants ---
TARGET = "scored_after"

METADATA_COLS = {
    "player_appearance_id", "player_id", "fixture_id", "date",
    "scored_after", "fold", "partition_role",
    "minute_out", "subbed", "jersey_number"
}
CATEGORICAL_COLS = ["checkpoint", "checkpoint_period", "position", "formation", "player_id"]
BOOLEAN_COLS     = ["is_home"]

def get_numeric_cols(df):
    exclude = METADATA_COLS | set(CATEGORICAL_COLS) | set(BOOLEAN_COLS)
    return [c for c in df.columns if c in df.columns and c not in exclude]

def run_shap(version="final_top20", use_poisson=True):
    print(f"Loading data and model for version: {version} (Poisson: {use_poisson})...")
    
    # 1. Load data
    data_path = FEATURES_DIR / f"features_{version}_dev.csv"
    if not data_path.exists():
        print(f"Error: Data not found at {data_path}.")
        return
        
    dev_df = pd.read_csv(data_path)
    
    # Pre-cast categories for consistency
    actual_cat_cols = [c for c in CATEGORICAL_COLS if c in dev_df.columns]
    for col in actual_cat_cols:
        dev_df[col] = dev_df[col].astype("category")
        
    actual_bool_cols = [c for c in BOOLEAN_COLS if c in dev_df.columns]
    numeric_cols = get_numeric_cols(dev_df)
    feature_cols = numeric_cols + actual_cat_cols + actual_bool_cols
    
    X = dev_df[feature_cols]
    
    # 2. Load calibrated model
    model_path = MODELS_DIR / f"calibrated_xgboost_{version}.pkl"
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}. Please run probability_calibration.py first.")
        return

    calibrated_clf = joblib.load(model_path)
    
    # 3. Extract the fitted pipeline
    pipeline = calibrated_clf.base_estimator
    
    preprocessor = pipeline.named_steps['prep']
    xgb_wrapper = pipeline.named_steps['clf']
    
    # 4. Extract raw XGBoost model
    if hasattr(xgb_wrapper, "model_"):
        xgb_model = xgb_wrapper.model_
    else:
        xgb_model = xgb_wrapper
        
    # 5. Preprocess data
    print("Preprocessing features...")
    X_preprocessed_raw = preprocessor.transform(X)
    
    if isinstance(X_preprocessed_raw, pd.DataFrame):
        X_preprocessed = X_preprocessed_raw.copy()
    else:
        try:
            feature_names = preprocessor.get_feature_names_out()
        except:
            feature_names = [f"f{i}" for i in range(X_preprocessed_raw.shape[1])]
        X_preprocessed = pd.DataFrame(X_preprocessed_raw, columns=feature_names)
    
    if use_poisson:
        exposure_col = "exposure" if "exposure" in X_preprocessed.columns else "remainder__exposure"
        if exposure_col in X_preprocessed.columns:
            X_preprocessed = X_preprocessed.drop(columns=[exposure_col])
    
    # 6. SHAP Analysis
    print("Computing SHAP values (using TreeExplainer)...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_preprocessed)
    
    # 7. Generate Plots
    print("Generating plots...")
    
    # Summary Plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_preprocessed, show=False)
    plt.title(f"SHAP Global Summary - XGBoost ({version})")
    summary_path = REPORTS_DIR / f"shap_summary_xgboost_{version}.png"
    plt.tight_layout()
    plt.savefig(summary_path, dpi=300)
    plt.close()
    
    # Importance Plot (Bar)
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_preprocessed, plot_type="bar", show=False)
    plt.title(f"SHAP Importance ({version})")
    importance_path = REPORTS_DIR / f"shap_importance_xgboost_{version}.png"
    plt.tight_layout()
    plt.savefig(importance_path, dpi=300)
    plt.close()
    
    # 8. Export Importance Data
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    mean_abs_shap = np.abs(sv).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": X_preprocessed.columns,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)
    
    csv_path = MODELS_DIR / f"shap_importance_{version}.csv"
    importance_df.to_csv(csv_path, index=False)
    
    print("\nSHAP Analysis Complete.")
    print(f"- Summary plot: {summary_path}")
    print(f"- Importance plot: {importance_path}")
    print(f"- Importance CSV: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="final_top20")
    parser.add_argument("--poisson", action="store_true", default=True)
    args = parser.parse_args()
    
    run_shap(version=args.version, use_poisson=args.poisson)
