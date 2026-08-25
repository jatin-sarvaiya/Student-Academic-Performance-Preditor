"""
train_model.py
==============
Loads preprocessed data, trains and tunes four classifiers via
RandomizedSearchCV, evaluates each on the held-out test set,
selects the best by F1-score, and saves results.

Run directly:
    python src/train_model.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths (all relative to project root)
# ---------------------------------------------------------------------------
ROOT_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR   = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

SEED = 42
CV_FOLDS = 5
N_ITER   = 15          # RandomizedSearchCV iterations per model


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
def load_data():
    """Load the processed train/test splits saved by data_preprocessing.py."""
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()
    y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 2. Model definitions with hyper-parameter grids
# ---------------------------------------------------------------------------
def get_model_configs():
    """
    Returns a dict of { model_name: (estimator, param_grid) }.
    XGBoost is tried first; falls back to GradientBoosting if not installed.
    """
    configs = {}

    # -- Logistic Regression --
    configs["Logistic Regression"] = (
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED),
        {
            "C":       [0.01, 0.1, 0.5, 1, 2, 5, 10],
            "solver":  ["lbfgs", "liblinear"],
            "penalty": ["l2"],
        },
    )

    # -- Decision Tree --
    configs["Decision Tree"] = (
        DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
        {
            "max_depth":        [3, 5, 7, 10, None],
            "min_samples_split":[2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "criterion":        ["gini", "entropy"],
        },
    )

    # -- Random Forest --
    configs["Random Forest"] = (
        RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1),
        {
            "n_estimators":     [100, 200, 300],
            "max_depth":        [5, 10, 15, None],
            "min_samples_split":[2, 5],
            "min_samples_leaf": [1, 2],
            "max_features":     ["sqrt", "log2"],
        },
    )

    # -- XGBoost / GradientBoosting --
    try:
        from xgboost import XGBClassifier
        configs["XGBoost"] = (
            XGBClassifier(
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=SEED,
                n_jobs=-1,
            ),
            {
                "n_estimators": [100, 200, 300],
                "max_depth":    [3, 5, 7],
                "learning_rate":[0.01, 0.05, 0.1, 0.2],
                "subsample":    [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0],
                "scale_pos_weight": [1, 2, 3],     # handles class imbalance
            },
        )
        print("[INFO] XGBoost found — will include XGBoost in training.")
    except ImportError:
        print("[WARN] XGBoost not installed — falling back to GradientBoosting.")
        configs["Gradient Boosting"] = (
            GradientBoostingClassifier(random_state=SEED),
            {
                "n_estimators": [100, 200],
                "max_depth":    [3, 5],
                "learning_rate":[0.05, 0.1, 0.2],
                "subsample":    [0.8, 1.0],
            },
        )

    return configs


# ---------------------------------------------------------------------------
# 3. Train and evaluate each model
# ---------------------------------------------------------------------------
def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Run RandomizedSearchCV for every model and collect test-set metrics."""
    configs = get_model_configs()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    results = []
    best_models = {}

    for name, (estimator, param_grid) in configs.items():
        print(f"\n{'='*60}")
        print(f"  Training: {name}")
        print(f"{'='*60}")

        search = RandomizedSearchCV(
            estimator,
            param_distributions=param_grid,
            n_iter=N_ITER,
            scoring="f1",
            cv=cv,
            random_state=SEED,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)
        best_est = search.best_estimator_

        # --- Predictions ---
        y_pred  = best_est.predict(X_test)
        y_proba = best_est.predict_proba(X_test)[:, 1]

        acc   = accuracy_score(y_test,  y_pred)
        prec  = precision_score(y_test, y_pred, zero_division=0)
        rec   = recall_score(y_test,   y_pred, zero_division=0)
        f1    = f1_score(y_test,       y_pred, zero_division=0)
        auc   = roc_auc_score(y_test,  y_proba)
        cm    = confusion_matrix(y_test, y_pred)

        print(f"  Best params : {search.best_params_}")
        print(f"  CV F1       : {search.best_score_:.4f}")
        print(f"  Test Acc    : {acc:.4f}  Prec: {prec:.4f}  "
              f"Rec: {rec:.4f}  F1: {f1:.4f}  AUC: {auc:.4f}")
        print(f"  Confusion Matrix:\n{cm}")
        print(classification_report(y_test, y_pred, target_names=["Fail", "Pass"]))

        results.append({
            "Model":           name,
            "CV F1":           round(search.best_score_, 4),
            "Test Accuracy":   round(acc,  4),
            "Test Precision":  round(prec, 4),
            "Test Recall":     round(rec,  4),
            "Test F1":         round(f1,   4),
            "Test ROC-AUC":    round(auc,  4),
            "Best Params":     str(search.best_params_),
        })
        best_models[name] = (best_est, f1)

    return results, best_models


# ---------------------------------------------------------------------------
# 4. Select and persist the best model
# ---------------------------------------------------------------------------
def select_and_save_best(results, best_models, X_train):
    """Select the model with highest test F1, package it, save to disk."""
    comparison_df = pd.DataFrame(results).sort_values("Test F1", ascending=False)
    comparison_path = os.path.join(MODELS_DIR, "model_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n[INFO] Model comparison saved -> {comparison_path}")
    print(comparison_df[["Model", "Test Accuracy", "Test F1", "Test ROC-AUC"]].to_string(index=False))

    best_name = comparison_df.iloc[0]["Model"]
    best_estimator, _ = best_models[best_name]
    print(f"\n[INFO] Best model: {best_name}")

    # Feature names from training set
    feature_names = list(X_train.columns)

    model_package = {
        "model":         best_estimator,
        "model_name":    best_name,
        "feature_names": feature_names,
    }
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(model_package, model_path)
    print(f"[INFO] Best model saved -> {model_path}")
    return best_name


# ---------------------------------------------------------------------------
# 5. Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[INFO] Loading preprocessed data ...")
    X_train, X_test, y_train, y_test = load_data()

    # Report class imbalance
    pass_rate = y_train.mean()
    print(f"[INFO] Training class balance -> Pass: {pass_rate*100:.1f}%  "
          f"Fail: {(1-pass_rate)*100:.1f}%")
    if pass_rate < 0.4 or pass_rate > 0.6:
        print("[WARN] Class imbalance detected — models use class_weight='balanced' "
              "or scale_pos_weight to compensate.")

    results, best_models = train_and_evaluate(X_train, X_test, y_train, y_test)
    select_and_save_best(results, best_models, X_train)
    print("\n[INFO] Training complete.")
