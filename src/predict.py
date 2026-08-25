"""
predict.py
==========
Provides the StudentPredictor class for inference.

Usage:
    from src.predict import StudentPredictor
    predictor = StudentPredictor()
    result = predictor.predict({"age": 17, "sex": "F", ...})
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

ROOT_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT_DIR, "models")


class StudentPredictor:
    """
    Loads the preprocessing pipeline and best model, then provides
    predict() for raw user input (as a dict or single-row DataFrame).
    """

    def __init__(self, models_dir: str = MODELS_DIR):
        pipeline_path = os.path.join(models_dir, "preprocessing_pipeline.pkl")
        model_path    = os.path.join(models_dir, "best_model.pkl")
        meta_path     = os.path.join(models_dir, "feature_metadata.pkl")
        explainer_path = os.path.join(models_dir, "shap_explainer.pkl")

        for path, label in [
            (pipeline_path, "preprocessing_pipeline.pkl"),
            (model_path,    "best_model.pkl"),
            (meta_path,     "feature_metadata.pkl"),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Required file not found: {label}. "
                    "Please run data_preprocessing.py and train_model.py first."
                )

        self.pipeline      = joblib.load(pipeline_path)
        model_pkg          = joblib.load(model_path)
        self.model         = model_pkg["model"]
        self.model_name    = model_pkg["model_name"]
        self.feature_names = model_pkg["feature_names"]   # post-preprocessing col names

        meta = joblib.load(meta_path)
        self.raw_feature_cols = meta["feature_cols"]      # pre-preprocessing col order
        self.has_grades       = meta["has_grades"]

        # SHAP explainer (optional – only available after evaluate.py)
        self.explainer = None
        if os.path.exists(explainer_path):
            self.explainer = joblib.load(explainer_path)

    # -----------------------------------------------------------------------
    # Input validation
    # -----------------------------------------------------------------------
    _RANGE_CHECKS = {
        "age":      (15, 22),
        "absences": (0, 93),
        "famrel":   (1, 5),
        "freetime": (1, 5),
        "goout":    (1, 5),
        "Dalc":     (1, 5),
        "Walc":     (1, 5),
        "health":   (1, 5),
        "Medu":     (0, 4),
        "Fedu":     (0, 4),
        "traveltime":(1, 4),
        "studytime": (1, 4),
        "failures":  (0, 3),
        "G1":        (0, 20),
        "G2":        (0, 20),
    }

    def _validate(self, raw: dict) -> dict:
        cleaned = dict(raw)
        for col, (lo, hi) in self._RANGE_CHECKS.items():
            if col in cleaned and cleaned[col] is not None:
                val = cleaned[col]
                if not (lo <= float(val) <= hi):
                    warnings.warn(
                        f"Value for '{col}' ({val}) is outside expected range "
                        f"[{lo}, {hi}]. Clipping.", UserWarning
                    )
                    cleaned[col] = max(lo, min(hi, float(val)))
        return cleaned

    # -----------------------------------------------------------------------
    # Main predict method
    # -----------------------------------------------------------------------
    def predict(self, raw_input: dict, include_shap: bool = True, top_k: int = 6) -> dict:
        """
        Parameters
        ----------
        raw_input : dict
            Dictionary of raw feature values exactly matching the training
            schema (before preprocessing). Grade columns G1/G2 are optional
            if the pipeline was trained without them.
        include_shap : bool
            Whether to compute per-instance SHAP explanations.
        top_k : int
            Number of top SHAP factors to return.

        Returns
        -------
        dict with keys:
            prediction_label  : str  ("Pass" or "Fail")
            prediction_class  : int  (1=Pass, 0=Fail)
            pass_probability  : float
            risk_score        : float  (= fail probability, 0-1)
            shap_explanations : list[dict]  (sorted by |SHAP|)
            force_html        : str  (HTML string for SHAP force plot)
        """
        raw_input = self._validate(raw_input)

        # Build single-row DataFrame aligned to training feature order
        # Fill missing optional columns (G1, G2) with median 10
        row = {}
        for col in self.raw_feature_cols:
            if col in raw_input and raw_input[col] is not None:
                row[col] = raw_input[col]
            else:
                row[col] = 10 if col in ("G1", "G2") else None

        df_input = pd.DataFrame([row])

        # Preprocess
        X_proc = self.pipeline.transform(df_input)

        # Predict
        pred_class = int(self.model.predict(X_proc)[0])
        pred_proba = self.model.predict_proba(X_proc)[0]
        pass_prob  = float(pred_proba[1])
        risk_score = float(pred_proba[0])   # fail probability

        result = {
            "prediction_label": "Pass" if pred_class == 1 else "Fail",
            "prediction_class": pred_class,
            "pass_probability": pass_prob,
            "risk_score":       risk_score,
            "shap_explanations": [],
            "shap_plot":        None,
        }

        if include_shap and self.explainer is not None:
            try:
                from src.evaluate import explain_instance
                expls = explain_instance(
                    self.model, self.explainer, X_proc,
                    self.feature_names, top_k=top_k
                )
                result["shap_explanations"] = expls
            except Exception as exc:
                warnings.warn(f"SHAP explanation failed: {exc}")

        return result


# ---------------------------------------------------------------------------
# Quick sanity test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    predictor = StudentPredictor()
    sample = {
        "school": "GP", "sex": "F", "age": 17,
        "address": "U", "famsize": "GT3", "Pstatus": "T",
        "Medu": 4, "Fedu": 3, "Mjob": "services", "Fjob": "other",
        "reason": "reputation", "guardian": "mother",
        "traveltime": 1, "studytime": 3, "failures": 0,
        "schoolsup": "no", "famsup": "yes", "paid": "no",
        "activities": "yes", "nursery": "yes", "higher": "yes",
        "internet": "yes", "romantic": "no",
        "famrel": 5, "freetime": 3, "goout": 2,
        "Dalc": 1, "Walc": 1, "health": 5, "absences": 2,
        "G1": 14, "G2": 15,
    }
    result = predictor.predict(sample, include_shap=False)
    print(f"\n  Prediction : {result['prediction_label']}")
    print(f"  Pass prob  : {result['pass_probability']:.4f}")
    print(f"  Risk score : {result['risk_score']:.4f}")
    print("\n[INFO] predict.py sanity check passed.")
