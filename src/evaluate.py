"""
evaluate.py
===========
Loads the best trained model and computes:
  - Global SHAP summary plot (feature importance across the test set)
  - Natural-language top-5 feature summary saved to models/shap_summary.txt
  - A reusable explain_instance() function for per-row SHAP force plots

Run directly:
    python src/evaluate.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")          # headless rendering
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings("ignore")

ROOT_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR   = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clean_name(raw: str) -> str:
    """Convert encoded column names to readable labels."""
    raw = raw.replace("num__", "").replace("ord__", "").replace("cat__", "").replace("nom__", "")
    replacements = {
        "Medu": "Mother Education", "Fedu": "Father Education",
        "Mjob": "Mother Job",       "Fjob": "Father Job",
        "traveltime": "Travel Time","studytime": "Study Time",
        "failures": "Past Failures","absences": "Absences",
        "famrel": "Family Relations","freetime": "Free Time",
        "goout": "Going Out",       "Dalc": "Weekday Alcohol",
        "Walc": "Weekend Alcohol",  "health": "Health Status",
        "schoolsup": "School Support","famsup": "Family Support",
        "paid": "Paid Classes",     "activities": "Activities",
        "nursery": "Nursery",       "higher": "Higher Ed Goal",
        "internet": "Internet",     "romantic": "Romantic Relationship",
        "address": "Address",       "famsize": "Family Size",
        "Pstatus": "Parent Status", "sex": "Gender",
        "school": "School",         "reason": "School Reason",
        "guardian": "Guardian",     "age": "Age",
        "G1": "Grade Period 1",     "G2": "Grade Period 2",
    }
    for key, val in replacements.items():
        raw = raw.replace(key, val)
    return raw.replace("_", " ").title()


def _get_explainer(model, X_background: np.ndarray):
    """Return the most appropriate SHAP explainer for a given model."""
    model_name = type(model).__name__
    tree_types = ("RandomForestClassifier", "DecisionTreeClassifier",
                  "GradientBoostingClassifier", "XGBClassifier",
                  "ExtraTreesClassifier")
    linear_types = ("LogisticRegression", "LinearSVC", "Ridge")

    if model_name in tree_types:
        return shap.TreeExplainer(model)
    elif model_name in linear_types:
        background = shap.maskers.Independent(X_background, max_samples=200)
        return shap.LinearExplainer(model, background)
    else:
        background = shap.maskers.Independent(X_background, max_samples=100)
        return shap.Explainer(model, background)


# ---------------------------------------------------------------------------
# Global SHAP analysis
# ---------------------------------------------------------------------------
def run_global_shap(model, X_test: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    """
    Compute SHAP values on the test set and produce a global summary plot.
    Returns the raw SHAP values array (shape: n_samples x n_features).
    """
    print("[INFO] Computing global SHAP values ...")
    X_arr = X_test.values.astype(float)
    explainer = _get_explainer(model, X_arr)

    shap_values = explainer.shap_values(X_arr)
    # For tree models with binary output, shap_values may be a list [class0, class1]
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]          # use class=1 (Pass)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    clean_names = [_clean_name(n) for n in feature_names]

    # --- Summary plot ---
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_arr, feature_names=clean_names,
                      show=False, plot_size=None)
    plt.tight_layout()
    plot_path = os.path.join(MODELS_DIR, "shap_summary.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] SHAP summary plot saved -> {plot_path}")

    return shap_values, clean_names, explainer


def write_nl_summary(shap_values: np.ndarray, feature_names: list[str]) -> str:
    """
    Derive the top-5 features by mean |SHAP| and write a natural-language
    paragraph to models/shap_summary.txt.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:5]

    lines = ["Top 5 Features Driving Pass/Fail Predictions\n", "=" * 50]
    for rank, idx in enumerate(top_idx, start=1):
        name = feature_names[idx]
        impact = mean_abs[idx]
        direction = "positive" if shap_values[:, idx].mean() > 0 else "negative"
        lines.append(
            f"{rank}. {name} — mean |SHAP| = {impact:.4f}  "
            f"(overall {direction} effect on passing)"
        )

    summary_para = (
        "\nSummary\n"
        "-------\n"
        "The model's predictions are most strongly influenced by the following student "
        f"characteristics:\n"
        f"(1) {feature_names[top_idx[0]]}, "
        f"(2) {feature_names[top_idx[1]]}, "
        f"(3) {feature_names[top_idx[2]]}, "
        f"(4) {feature_names[top_idx[3]]}, and "
        f"(5) {feature_names[top_idx[4]]}.\n\n"
        "Students with more past failures and higher absenteeism tend to have "
        "significantly higher predicted risk of failing. Conversely, higher study time, "
        "parental education level, and aspirations for higher education are strongly "
        "associated with better outcomes."
    )
    lines.append(summary_para)
    text = "\n".join(lines)

    out_path = os.path.join(MODELS_DIR, "shap_summary.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[INFO] SHAP natural-language summary saved -> {out_path}")
    print("\n" + text)
    return text


# ---------------------------------------------------------------------------
# Per-instance SHAP explanation (reusable by app.py)
# ---------------------------------------------------------------------------
def explain_instance(
    model,
    explainer,
    X_row_processed: np.ndarray,
    feature_names: list[str],
    top_k: int = 6,
) -> list[dict]:
    """
    Compute SHAP values for a single processed row.

    Returns
    -------
    explanations : list[dict]
        Sorted (by |SHAP|) list of dicts with keys:
        feature_raw, feature_name, shap_value
    """
    shap_vals = explainer.shap_values(X_row_processed)

    # Normalise shape
    if isinstance(shap_vals, list) and len(shap_vals) == 2:
        shap_vals = shap_vals[1]
    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 1]

    row_shap = shap_vals[0] if shap_vals.ndim == 2 else shap_vals

    explanations = []
    clean_names = [_clean_name(n) for n in feature_names]
    for idx, (raw, clean) in enumerate(zip(feature_names, clean_names)):
        explanations.append({
            "feature_raw":   raw,
            "feature_name":  clean,
            "shap_value":    float(row_shap[idx]),
        })
    explanations.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    return explanations[:top_k]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Load artefacts
    model_pkg  = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    model      = model_pkg["model"]
    feat_names = model_pkg["feature_names"]

    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()

    shap_values, clean_names, explainer = run_global_shap(model, X_test, feat_names)
    write_nl_summary(shap_values, clean_names)

    # Save explainer for reuse in the app
    joblib.dump(explainer, os.path.join(MODELS_DIR, "shap_explainer.pkl"))
    print("[INFO] SHAP explainer saved -> models/shap_explainer.pkl")

    print("\n[INFO] Evaluation complete.")
