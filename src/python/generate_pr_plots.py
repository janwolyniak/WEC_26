"""
Generate Precision-Recall (PR) curves for the tuned finalists (Step 13).
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
import xgboost as xgb

# --- Constants ---
ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = ROOT / "artifacts" / "features"
TUNING_DIR = ROOT / "artifacts" / "tuning"
TARGET = "scored_after"
RANDOM_SEED = 17

METADATA_COLS = {
    "player_appearance_id", "player_id", "fixture_id", "date",
    "scored_after", "fold", "partition_role",
    "minute_out", "subbed", "jersey_number",
}
CATEGORICAL_COLS = ["checkpoint", "checkpoint_period", "position", "formation", "player_id"]
BOOLEAN_COLS = ["is_home"]

def get_numeric_cols(df: pd.DataFrame) -> list[str]:
    exclude = METADATA_COLS | set(CATEGORICAL_COLS) | set(BOOLEAN_COLS)
    return [c for c in df.columns if c not in exclude]

# --- Preprocessors ---
def _cat_tr():
    return Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

def to_float(x):
    return x.astype(float)

def to_category(x):
    return x.astype('category')

def build_lr_pipeline(params, numeric_cols):
    num_tr = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    bool_tr = FunctionTransformer(to_float)
    prep = ColumnTransformer([
        ("num", num_tr, numeric_cols),
        ("cat", _cat_tr(), CATEGORICAL_COLS),
        ("bool", bool_tr, BOOLEAN_COLS),
    ])
    
    # Split penalty_solver
    ps = params.get("penalty_solver", "l2_lbfgs")
    penalty, solver = ps.split("_", 1)
    
    return Pipeline([
        ("prep", prep),
        ("clf", LogisticRegression(
            C=params.get("C", 1.0),
            penalty=penalty,
            solver=solver,
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_SEED
        ))
    ])

def build_xgb_pipeline(params, numeric_cols, spw):
    bool_tr = FunctionTransformer(to_float)
    cat_tr_xgb = FunctionTransformer(to_category)
    prep = ColumnTransformer([
        ("cat", cat_tr_xgb, CATEGORICAL_COLS),
        ("bool", bool_tr, BOOLEAN_COLS),
        ("num", SimpleImputer(strategy="median"), numeric_cols),
    ])
    prep.set_output(transform='pandas')
    return Pipeline([
        ("prep", prep),
        ("clf", xgb.XGBClassifier(
            **params,
            scale_pos_weight=spw,
            eval_metric="logloss",
            enable_categorical=True,
            random_state=RANDOM_SEED
        ))
    ])

def run():
    # 1. Load data and params
    dev_df = pd.read_csv(FEATURES_DIR / "features_v2_dev.csv")
    with open(TUNING_DIR / "best_params.json") as f:
        best_params = json.load(f)
    
    numeric_cols = get_numeric_cols(dev_df)
    feature_cols = numeric_cols + CATEGORICAL_COLS + BOOLEAN_COLS
    folds = sorted(dev_df["fold"].unique())
    
    # 2. Get OOF probabilities
    oof_y = dev_df[TARGET].values
    model_probs = {}
    
    for model_name in ["LogisticRegression", "XGBoost"]:
        print(f"Generating OOF predictions for {model_name}...")
        probs = np.zeros(len(dev_df))
        params = best_params[model_name]["params"]
        
        for fold_id in folds:
            tr_idx = dev_df[dev_df["fold"] != fold_id].index
            va_idx = dev_df[dev_df["fold"] == fold_id].index
            
            X_tr, y_tr = dev_df.loc[tr_idx, feature_cols], dev_df.loc[tr_idx, TARGET]
            X_va = dev_df.loc[va_idx, feature_cols]
            
            if model_name == "LogisticRegression":
                clf = build_lr_pipeline(params, numeric_cols)
            else:
                spw = ((len(y_tr) - y_tr.sum()) / y_tr.sum()) if y_tr.sum() > 0 else 1.0
                clf = build_xgb_pipeline(params, numeric_cols, spw)
            
            clf.fit(X_tr, y_tr)
            probs[va_idx] = clf.predict_proba(X_va)[:, 1]
        
        model_probs[model_name] = probs

    # 3. Plot
    plt.figure(figsize=(10, 7))
    sns.set_style("whitegrid")
    
    # Random baseline
    no_skill = len(dev_df[dev_df[TARGET] == 1]) / len(dev_df)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label=f'Baseline (No Skill): {no_skill:.3f}', color='gray')
    
    colors = {'LogisticRegression': '#1f77b4', 'XGBoost': '#ff7f0e'}
    
    for model_name, probs in model_probs.items():
        precision, recall, _ = precision_recall_curve(oof_y, probs)
        ap = average_precision_score(oof_y, probs)
        plt.plot(recall, precision, label=f'{model_name} (PR AUC = {ap:.3f})', color=colors[model_name], lw=2)
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves (OOF - v2 Features)', fontsize=14)
    plt.legend(frameon=True, fontsize=10)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    
    save_path = TUNING_DIR / "pr_curves_v2.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    run()
