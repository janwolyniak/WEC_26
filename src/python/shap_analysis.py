import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# --- Paths ---
ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = ROOT / "artifacts" / "features"
MODELS_DIR = ROOT / "artifacts" / "models"
REPORTS_DIR = ROOT / "reports" / "figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Constants ---
TARGET = "scored_after"

def to_float(x):
    return x.astype(float)

def to_category(x):
    return x.astype('category')

def get_numeric_cols(df, metadata_cols, categorical_cols, boolean_cols):
    exclude = metadata_cols | set(categorical_cols) | set(boolean_cols)
    return [c for c in df.columns if c not in exclude]

def run_shap():
    print("Loading data and model...")
    # 1. Load data
    dev_df = pd.read_csv(FEATURES_DIR / "features_v3_dev.csv")
    
    # 2. Re-identify feature column buckets (consistent with model_tuning.py)
    METADATA_COLS = {
        "player_appearance_id", "player_id", "fixture_id", "date",
        "scored_after", "fold", "partition_role",
        "minute_out", "subbed", "jersey_number",
    }
    CATEGORICAL_COLS = ["checkpoint", "checkpoint_period", "position", "formation", "player_id"]
    BOOLEAN_COLS     = ["is_home"]
    
    numeric_cols = get_numeric_cols(dev_df, METADATA_COLS, CATEGORICAL_COLS, BOOLEAN_COLS)
    feature_cols = numeric_cols + CATEGORICAL_COLS + BOOLEAN_COLS
    
    X = dev_df[feature_cols]
    
    # 3. Load calibrated model
    model_path = MODELS_DIR / "calibrated_xgboost_v3.pkl"
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}. Please run probability_calibration.py first.")
        return

    calibrated_clf = joblib.load(model_path)
    
    # 4. Extract the fitted pipeline from the first fold
    cal_clf = calibrated_clf.calibrated_classifiers_[0]
    pipeline = cal_clf.estimator
    
    preprocessor = pipeline.named_steps['prep']
    xgb_model = pipeline.named_steps['clf']
    
    # 5. Preprocess data
    print("Preprocessing features...")
    
    print("Transformers in prep:")
    for name, transformer, columns in preprocessor.transformers_:
        print(f"Name: {name}, Transformer: {transformer}, Columns: {len(columns) if hasattr(columns, '__len__') else 'all'}")
    
    X_preprocessed_raw = preprocessor.transform(X)
    print("X_preprocessed_raw type:", type(X_preprocessed_raw))
    print("X_preprocessed_raw shape:", X_preprocessed_raw.shape)
    
    if isinstance(X_preprocessed_raw, pd.DataFrame):
        feature_names = list(X_preprocessed_raw.columns)
        X_preprocessed = X_preprocessed_raw
    else:
        # Fallback
        feature_names = [f"f{i}" for i in range(X_preprocessed_raw.shape[1])]
        X_preprocessed = pd.DataFrame(X_preprocessed_raw, columns=feature_names)
    
    # 6. SHAP Analysis
    print("Computing SHAP values (using TreeExplainer)...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_results = explainer(X_preprocessed)
    
    # 7. Generate Plots
    print("Generating plots...")
    
    # Summary Plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_results, X_preprocessed, show=False)
    plt.title("SHAP Global Summary - XGBoost (Fold 0)")
    summary_path = REPORTS_DIR / "shap_summary_xgboost.png"
    plt.tight_layout()
    plt.savefig(summary_path, dpi=300)
    plt.close()
    
    # Importance Plot (Bar)
    plt.figure(figsize=(12, 10))
    shap.plots.bar(shap_results, max_display=20, show=False)
    plt.title("SHAP Feature Importance (Mean |SHAP value|)")
    importance_path = REPORTS_DIR / "shap_importance_xgboost.png"
    plt.tight_layout()
    plt.savefig(importance_path, dpi=300)
    plt.close()
    
    # 8. Export Importance Data
    mean_abs_shap = np.abs(shap_results.values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)
    
    csv_path = MODELS_DIR / "shap_importance_v3.csv"
    importance_df.to_csv(csv_path, index=False)
    
    print("\nSHAP Analysis Complete.")
    print(f"- Summary plot: {summary_path}")
    print(f"- Importance plot: {importance_path}")
    print(f"- Importance CSV: {csv_path}")

if __name__ == "__main__":
    run_shap()
