import pandas as pd
import numpy as np
from pathlib import Path
import json

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, average_precision_score, brier_score_loss
import xgboost as xgb

def to_float(x):
    return x.astype(float)

def to_category(x):
    return x.astype('category')

def build_baseline():
    root = Path(__file__).resolve().parents[2]
    data_dir = root / "data"
    splits_dir = data_dir / "splits"
    artifacts_dir = root / "artifacts" / "baseline"
    docs_dir = root / "docs"
    
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(splits_dir / "modeling_row_folds.csv")
    
    # Filter only development
    dev_df = df[df["partition_role"] == "development"].copy()
    
    numeric_features = [col for col in df.columns if col.startswith("last15_") or col.startswith("cumul_")] + ["checkpoint_min", "minute_in", "minute_out"]
    categorical_features = ["checkpoint", "checkpoint_period", "position", "formation", "player_id"]
    boolean_features = ["is_home", "subbed"]
    target = "scored_after"
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    boolean_transformer = Pipeline(steps=[
        ('to_float', FunctionTransformer(to_float)),
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])
    
    preprocessor_lr = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('bool', boolean_transformer, boolean_features)
        ])

    num_imputer = SimpleImputer(strategy='median')
    bool_to_float = FunctionTransformer(to_float)
    cat_to_category = FunctionTransformer(to_category)
    preprocessor_xgb = ColumnTransformer(
        transformers=[
            ('cat', cat_to_category, categorical_features),
            ('bool', bool_to_float, boolean_features),
            ('num', num_imputer, numeric_features),
        ]
    )
    preprocessor_xgb.set_output(transform='pandas')
    
    results = []
    feature_importances = []
    
    for model_name in ["LogisticRegression", "XGBoost"]:
        print(f"Training {model_name}...")
        for fold_idx in sorted(dev_df["fold"].unique()):
            train_mask = dev_df["fold"] != fold_idx
            val_mask = dev_df["fold"] == fold_idx
            
            X_train = dev_df.loc[train_mask, numeric_features + categorical_features + boolean_features]
            y_train = dev_df.loc[train_mask, target]
            
            X_val = dev_df.loc[val_mask, numeric_features + categorical_features + boolean_features]
            y_val = dev_df.loc[val_mask, target]
            
            if model_name == "LogisticRegression":
                clf = Pipeline(steps=[('preprocessor', preprocessor_lr),
                                      ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))])
            else:
                scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum() if y_train.sum() > 0 else 1.0
                clf = Pipeline(steps=[('preprocessor', preprocessor_xgb),
                                      ('classifier', xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric='logloss', enable_categorical=True, random_state=42))])
            
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            y_proba = clf.predict_proba(X_val)[:, 1]
            
            res = {
                "model": model_name,
                "fold": fold_idx,
                "balanced_accuracy": balanced_accuracy_score(y_val, y_pred),
                "roc_auc": roc_auc_score(y_val, y_proba),
                "pr_auc": average_precision_score(y_val, y_proba),
                "brier_score": brier_score_loss(y_val, y_proba)
            }
            results.append(res)
            
            # Extract importances from first fold
            if fold_idx == "fold_0":
                prep = clf.named_steps['preprocessor']
                if model_name == "LogisticRegression":
                    ohe = prep.named_transformers_['cat'].named_steps['onehot']
                    cat_names = list(ohe.get_feature_names_out(categorical_features))
                    # Order: num | cat | bool  (matches preprocessor_lr transformer order)
                    feature_names = numeric_features + cat_names + boolean_features
                    coefs = clf.named_steps['classifier'].coef_[0]
                    for fn, coef in zip(feature_names, coefs):
                        feature_importances.append({"model": model_name, "feature": fn, "importance": coef})
                else:
                    # Order: cat | bool | num  (matches preprocessor_xgb transformer order)
                    cat_names = categorical_features
                    feature_names = cat_names + boolean_features + numeric_features
                    importances = clf.named_steps['classifier'].feature_importances_
                    for fn, imp in zip(feature_names, importances):
                        feature_importances.append({"model": model_name, "feature": fn, "importance": imp})
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(artifacts_dir / "fold_metrics.csv", index=False)
    
    summary_df = results_df.groupby("model").mean(numeric_only=True).reset_index()
    summary_df.to_csv(artifacts_dir / "baseline_metrics.csv", index=False)
    
    fi_df = pd.DataFrame(feature_importances)
    fi_df.to_csv(artifacts_dir / "feature_importances.csv", index=False)
    
    # Write markdown
    md_lines = ["# Baseline Pipeline Metrics\n\nGenerated by `src/python/baseline_pipeline.py`.\n\n## Mean CV Performance\n"]
    md_lines.append("| Model | Balanced Accuracy | ROC AUC | PR AUC | Brier Score |")
    md_lines.append("|---|---:|---:|---:|---:|")
    for _, row in summary_df.iterrows():
        md_lines.append(f"| {row['model']} | {row['balanced_accuracy']:.4f} | {row['roc_auc']:.4f} | {row['pr_auc']:.4f} | {row['brier_score']:.4f} |")
    
    md_lines.append("\n## Top 10 Features (XGBoost)\n")
    xgb_fi = fi_df[fi_df["model"] == "XGBoost"].sort_values("importance", ascending=False).head(10)
    md_lines.append("| Feature | Importance |")
    md_lines.append("|---|---:|")
    for _, row in xgb_fi.iterrows():
        md_lines.append(f"| {row['feature']} | {row['importance']:.4f} |")
        
    md_lines.append("\n## Top 10 Coefficients by absolute value (Logistic Regression)\n")
    lr_fi = fi_df[fi_df["model"] == "LogisticRegression"].copy()
    lr_fi["abs_importance"] = lr_fi["importance"].abs()
    lr_fi = lr_fi.sort_values("abs_importance", ascending=False).head(10)
    md_lines.append("| Feature | Coefficient |")
    md_lines.append("|---|---:|")
    for _, row in lr_fi.iterrows():
        md_lines.append(f"| {row['feature']} | {row['importance']:.4f} |")
        
    (docs_dir / "baseline_metrics.md").write_text("\n".join(md_lines), encoding="utf-8")

if __name__ == "__main__":
    build_baseline()
