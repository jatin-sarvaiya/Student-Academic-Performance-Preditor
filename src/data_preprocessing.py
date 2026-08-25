"""
data_preprocessing.py
=====================
Downloads (or synthetically generates) the UCI Student Performance dataset,
validates inputs, builds a full sklearn preprocessing pipeline, performs
stratified train/test split, and saves all artefacts.

Run directly:
    python src/data_preprocessing.py
"""

import os
import sys
import io
import zipfile
import urllib.request
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
PIPELINE_PATH = os.path.join(MODELS_DIR, "preprocessing_pipeline.pkl")
RAW_CSV_PATH = os.path.join(DATA_DIR, "student_data.csv")

# ---------------------------------------------------------------------------
# Column definitions (UCI schema)
# ---------------------------------------------------------------------------
NUMERIC_COLS = ["age", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"]
ORDINAL_COLS = {
    # column  -> ordered categories (low -> high)
    "Medu":      [0, 1, 2, 3, 4],
    "Fedu":      [0, 1, 2, 3, 4],
    "traveltime":[1, 2, 3, 4],
    "studytime": [1, 2, 3, 4],
    "failures":  [0, 1, 2, 3],
}
NOMINAL_COLS = [
    "school", "sex", "address", "famsize", "Pstatus",
    "Mjob", "Fjob", "reason", "guardian",
    "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic",
]
# Optional grade columns – included only when present in data
GRADE_COLS = ["G1", "G2"]
TARGET_COL = "G3"


# ---------------------------------------------------------------------------
# 1. Dataset acquisition
# ---------------------------------------------------------------------------
def download_dataset(data_dir: str = DATA_DIR) -> pd.DataFrame | None:
    """Attempt to download student-mat.csv from UCI repository."""
    url = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
    try:
        print(f"[INFO] Downloading dataset from {url} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            zip_bytes = resp.read()

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as outer:
            # The outer zip contains a nested student.zip
            if "student.zip" in outer.namelist():
                inner_bytes = outer.read("student.zip")
                with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
                    with inner.open("student-mat.csv") as f:
                        df = pd.read_csv(f, sep=";")
            elif "student-mat.csv" in outer.namelist():
                with outer.open("student-mat.csv") as f:
                    df = pd.read_csv(f, sep=";")
            else:
                raise FileNotFoundError("student-mat.csv not found inside zip.")

        print(f"[INFO] Downloaded UCI dataset: {len(df)} rows, {len(df.columns)} columns.")
        return df

    except Exception as exc:
        print(f"[WARN] Download failed: {exc}")
        return None


def generate_synthetic_dataset(n: int = 800, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic dataset that closely mimics the UCI student-mat.csv
    schema – same column names, same categorical values, same numeric ranges
    and realistic inter-feature correlations.
    """
    rng = np.random.default_rng(seed)

    def choice(vals, probs, size=n):
        return rng.choice(vals, size=size, p=probs)

    data = {}
    data["school"]    = choice(["GP","MS"],           [0.78, 0.22])
    data["sex"]       = choice(["F","M"],              [0.53, 0.47])
    data["age"]       = choice(range(15, 23),          [0.19,0.29,0.25,0.17,0.06,0.02,0.01,0.01])
    data["address"]   = choice(["U","R"],              [0.78, 0.22])
    data["famsize"]   = choice(["GT3","LE3"],          [0.71, 0.29])
    data["Pstatus"]   = choice(["T","A"],              [0.89, 0.11])
    data["Medu"]      = choice([0,1,2,3,4],            [0.03,0.15,0.26,0.29,0.27])
    data["Fedu"]      = choice([0,1,2,3,4],            [0.02,0.23,0.30,0.25,0.20])
    data["Mjob"]      = choice(["at_home","health","other","services","teacher"], [0.19,0.09,0.35,0.28,0.09])
    data["Fjob"]      = choice(["at_home","health","other","services","teacher"], [0.06,0.08,0.42,0.28,0.16])
    data["reason"]    = choice(["course","home","other","reputation"],            [0.34,0.27,0.10,0.29])
    data["guardian"]  = choice(["father","mother","other"],                       [0.22,0.71,0.07])
    data["traveltime"]= choice([1,2,3,4], [0.56,0.32,0.09,0.03])
    data["studytime"] = choice([1,2,3,4], [0.23,0.55,0.17,0.05])
    data["failures"]  = choice([0,1,2,3], [0.79,0.13,0.05,0.03])
    data["schoolsup"] = choice(["no","yes"], [0.87,0.13])
    data["famsup"]    = choice(["no","yes"], [0.41,0.59])
    data["paid"]      = choice(["no","yes"], [0.54,0.46])
    data["activities"]= choice(["no","yes"], [0.49,0.51])
    data["nursery"]   = choice(["no","yes"], [0.19,0.81])
    data["higher"]    = choice(["no","yes"], [0.05,0.95])
    data["internet"]  = choice(["no","yes"], [0.17,0.83])
    data["romantic"]  = choice(["no","yes"], [0.67,0.33])
    data["famrel"]    = choice([1,2,3,4,5], [0.03,0.06,0.23,0.45,0.23])
    data["freetime"]  = choice([1,2,3,4,5], [0.04,0.18,0.36,0.31,0.11])
    data["goout"]     = choice([1,2,3,4,5], [0.06,0.26,0.32,0.25,0.11])
    data["Dalc"]      = choice([1,2,3,4,5], [0.68,0.20,0.07,0.03,0.02])
    data["Walc"]      = choice([1,2,3,4,5], [0.38,0.27,0.19,0.10,0.06])
    data["health"]    = choice([1,2,3,4,5], [0.11,0.09,0.22,0.22,0.36])
    data["absences"]  = np.clip(rng.poisson(lam=5.7, size=n), 0, 75).astype(int)

    df = pd.DataFrame(data)

    # Derive correlated grades
    ability = (
        10.5
        + 1.5 * df["studytime"]
        - 2.8 * df["failures"]
        + 0.4 * df["Medu"]
        + 0.3 * df["Fedu"]
        - 0.08 * df["absences"]
        - 0.25 * df["goout"]
        - 0.25 * df["Walc"]
        + np.where(df["higher"] == "yes", 1.0, 0)
        + np.where(df["internet"] == "yes", 0.5, 0)
        + rng.normal(0, 1.8, size=n)
    )
    G1 = np.clip(rng.normal(ability, 1.2), 0, 20).round().astype(int)
    G2 = np.clip(rng.normal(0.6 * G1 + 0.4 * ability, 0.9), 0, 20).round().astype(int)
    G3 = np.clip(rng.normal(0.75 * G2 + 0.25 * ability, 0.9), 0, 20).round().astype(int)
    # Students with very poor G2 often withdraw -> G3 = 0
    G3[G2 < 5] = np.where(rng.random(size=(G2 < 5).sum()) < 0.55, 0, G3[G2 < 5])

    df["G1"] = G1
    df["G2"] = G2
    df["G3"] = G3
    return df


# ---------------------------------------------------------------------------
# 2. Input validation
# ---------------------------------------------------------------------------
def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and coerce known impossible values:
    - absences: clip to [0, 93]
    - age: clip to [15, 22]
    - grade columns G1, G2, G3: clip to [0, 20]
    - ordinal cols: clip to valid range
    """
    if "absences" in df.columns:
        df["absences"] = df["absences"].clip(0, 93)
    if "age" in df.columns:
        df["age"] = df["age"].clip(15, 22)
    for col in ["G1", "G2", "G3"]:
        if col in df.columns:
            df[col] = df[col].clip(0, 20)
    for col, vals in ORDINAL_COLS.items():
        if col in df.columns:
            df[col] = df[col].clip(min(vals), max(vals))
    return df


# ---------------------------------------------------------------------------
# 3. Build preprocessing pipeline
# ---------------------------------------------------------------------------
def build_preprocessor(has_grades: bool = True) -> ColumnTransformer:
    """
    Returns a ColumnTransformer that handles:
    - Numeric columns  -> median imputation -> StandardScaler
    - Ordinal columns  -> most-frequent imputation -> OrdinalEncoder
    - Nominal columns  -> most-frequent imputation -> OneHotEncoder
    - Grade columns (optional) -> median imputation -> StandardScaler
    """
    ordinal_col_list  = list(ORDINAL_COLS.keys())
    ordinal_categories = [ORDINAL_COLS[c] for c in ordinal_col_list]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    ord_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(categories=ordinal_categories, handle_unknown="use_encoded_value", unknown_value=np.nan)),
    ])
    nom_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
    ])

    transformers = [
        ("num", num_pipe, NUMERIC_COLS),
        ("ord", ord_pipe, ordinal_col_list),
        ("nom", nom_pipe, NOMINAL_COLS),
    ]
    if has_grades:
        grade_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])
        transformers.append(("grades", grade_pipe, GRADE_COLS))

    return ColumnTransformer(transformers=transformers, remainder="drop")


# ---------------------------------------------------------------------------
# 4. Main preprocessing workflow
# ---------------------------------------------------------------------------
def run_preprocessing(n_synthetic: int = 800, seed: int = 42):
    """Full preprocessing workflow: load -> validate -> split -> fit pipeline -> save."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # --- Load data ---
    df = download_dataset()
    if df is None:
        print(f"[INFO] Generating synthetic dataset ({n_synthetic} rows) ...")
        df = generate_synthetic_dataset(n=n_synthetic, seed=seed)

    df = validate_dataframe(df)
    df.to_csv(RAW_CSV_PATH, index=False)
    print(f"[INFO] Raw data saved to {RAW_CSV_PATH} ({len(df)} rows).")

    # --- Target variable ---
    y = (df[TARGET_COL] >= 10).astype(int)
    pass_count  = y.sum()
    fail_count  = (y == 0).sum()
    print(f"[INFO] Class distribution -> Pass: {pass_count} ({pass_count/len(y)*100:.1f}%)  "
          f"Fail: {fail_count} ({fail_count/len(y)*100:.1f}%)")

    # --- Features ---
    has_grades = all(c in df.columns for c in GRADE_COLS)
    feature_cols = NUMERIC_COLS + list(ORDINAL_COLS.keys()) + NOMINAL_COLS
    if has_grades:
        feature_cols += GRADE_COLS
    X = df[feature_cols].copy()

    # --- Stratified split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    print(f"[INFO] Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")

    # --- Fit preprocessor ---
    preprocessor = build_preprocessor(has_grades=has_grades)
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, PIPELINE_PATH)
    print(f"[INFO] Preprocessing pipeline saved -> {PIPELINE_PATH}")

    # --- Transform and save processed splits ---
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)

    # Get feature names
    feature_names = _get_feature_names(preprocessor, has_grades)

    X_train_df = pd.DataFrame(X_train_proc, columns=feature_names)
    X_test_df  = pd.DataFrame(X_test_proc,  columns=feature_names)

    X_train_df.to_csv(os.path.join(DATA_DIR, "X_train.csv"), index=False)
    X_test_df.to_csv(os.path.join(DATA_DIR,  "X_test.csv"),  index=False)
    y_train.reset_index(drop=True).to_csv(os.path.join(DATA_DIR, "y_train.csv"), index=False)
    y_test.reset_index(drop=True).to_csv(os.path.join(DATA_DIR,  "y_test.csv"),  index=False)

    # Save metadata
    meta = {
        "feature_cols":    feature_cols,
        "feature_names":   feature_names,
        "has_grades":      has_grades,
        "numeric_cols":    NUMERIC_COLS,
        "ordinal_cols":    list(ORDINAL_COLS.keys()),
        "nominal_cols":    NOMINAL_COLS,
        "grade_cols":      GRADE_COLS if has_grades else [],
        "seed":            seed,
    }
    joblib.dump(meta, os.path.join(MODELS_DIR, "feature_metadata.pkl"))
    print("[INFO] Preprocessing complete. All artefacts saved.")
    return X_train_df, X_test_df, y_train, y_test


def _get_feature_names(preprocessor: ColumnTransformer, has_grades: bool) -> list:
    """Extract ordered feature names from a fitted ColumnTransformer."""
    names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "num":
            names += list(cols)
        elif name == "ord":
            names += list(cols)
        elif name == "nom":
            enc = transformer.named_steps["encoder"]
            names += list(enc.get_feature_names_out(cols))
        elif name == "grades":
            names += list(cols)
    return names


if __name__ == "__main__":
    run_preprocessing()
