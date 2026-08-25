"""
app.py  —  Student Academic Performance Predictor
==================================================
A fully-featured Streamlit demo app.

Launch:
    streamlit run app/app.py
(Run from the project root directory.)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import joblib

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup — make imports work when launched from project root
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

MODELS_DIR = os.path.join(ROOT_DIR, "models")

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — premium dark-mode styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #334155;
}
[data-testid="stSidebar"] .stMarkdown h2 {
    color: #94a3b8; font-size: 0.75rem; text-transform: uppercase;
    letter-spacing: 0.1em; margin-top: 1.4rem; margin-bottom: 0.4rem;
}

/* Main area background */
.main { background: #0f172a; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155; border-radius: 12px;
    padding: 1.2rem 1.5rem; margin-bottom: 0.8rem;
}
.metric-card h1 { font-size: 2.4rem; font-weight: 700; margin: 0; }
.metric-card p  { color: #94a3b8; font-size: 0.9rem; margin: 0; }

/* Badges */
.badge-pass {
    display: inline-block;
    background: linear-gradient(135deg, #22c55e, #16a34a);
    color: white; border-radius: 999px; padding: 0.5rem 1.8rem;
    font-size: 1.5rem; font-weight: 700; letter-spacing: 0.05em;
}
.badge-fail {
    display: inline-block;
    background: linear-gradient(135deg, #ef4444, #b91c1c);
    color: white; border-radius: 999px; padding: 0.5rem 1.8rem;
    font-size: 1.5rem; font-weight: 700; letter-spacing: 0.05em;
}

/* Factor cards */
.factor-pos {
    background: #052e16; border-left: 4px solid #22c55e;
    border-radius: 8px; padding: 0.6rem 1rem; margin-bottom: 0.4rem;
    color: #bbf7d0; font-size: 0.88rem;
}
.factor-neg {
    background: #450a0a; border-left: 4px solid #ef4444;
    border-radius: 8px; padding: 0.6rem 1rem; margin-bottom: 0.4rem;
    color: #fecaca; font-size: 0.88rem;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load model artefacts  (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model …")
def load_artefacts():
    pipeline_path  = os.path.join(MODELS_DIR, "preprocessing_pipeline.pkl")
    model_path     = os.path.join(MODELS_DIR, "best_model.pkl")
    meta_path      = os.path.join(MODELS_DIR, "feature_metadata.pkl")
    explainer_path = os.path.join(MODELS_DIR, "shap_explainer.pkl")

    missing = [p for p in [pipeline_path, model_path, meta_path] if not os.path.exists(p)]
    if missing:
        return None, None, None, None, [os.path.basename(p) for p in missing]

    pipeline   = joblib.load(pipeline_path)
    model_pkg  = joblib.load(model_path)
    meta       = joblib.load(meta_path)
    explainer  = joblib.load(explainer_path) if os.path.exists(explainer_path) else None
    return pipeline, model_pkg, meta, explainer, []


pipeline, model_pkg, meta, explainer, missing_files = load_artefacts()


# ---------------------------------------------------------------------------
# Helper: run inference
# ---------------------------------------------------------------------------
def run_prediction(raw_input: dict):
    """Transform raw input dict → prediction dict."""
    feature_cols = meta["feature_cols"]
    has_grades   = meta["has_grades"]

    row = {}
    for col in feature_cols:
        val = raw_input.get(col)
        if val is None:
            val = 10 if col in ("G1", "G2") else 0
        row[col] = val
    df_input = pd.DataFrame([row])

    X_proc = pipeline.transform(df_input)
    model  = model_pkg["model"]
    feat_names = model_pkg["feature_names"]

    pred_class = int(model.predict(X_proc)[0])
    proba      = model.predict_proba(X_proc)[0]
    pass_prob  = float(proba[1])
    risk_score = float(proba[0])

    shap_explanations = []

    if explainer is not None:
        try:
            shap_vals = explainer.shap_values(X_proc)
            if isinstance(shap_vals, list) and len(shap_vals) == 2:
                shap_vals = shap_vals[1]
            elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                shap_vals = shap_vals[:, :, 1]
            row_shap = shap_vals[0] if shap_vals.ndim == 2 else shap_vals

            from src.evaluate import _clean_name
            for idx, n in enumerate(feat_names):
                shap_explanations.append({
                    "feature_raw":   n,
                    "feature_name":  _clean_name(n),
                    "shap_value":    float(row_shap[idx]),
                })
            shap_explanations.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
            shap_explanations = shap_explanations[:6]
        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")

    return {
        "label":      "Pass" if pred_class == 1 else "Fail",
        "pred_class": pred_class,
        "pass_prob":  pass_prob,
        "risk_score": risk_score,
        "shap":       shap_explanations,
        "X_proc":     X_proc,
    }


# ---------------------------------------------------------------------------
# Default values (matching median student in UCI dataset)
# ---------------------------------------------------------------------------
DEFAULTS = {
    "age": 17, "sex": "Female", "address": "Urban", "famsize": ">3 members",
    "Pstatus": "Living together",
    "Medu": "Secondary Education", "Fedu": "Secondary Education",
    "Mjob": "Other", "Fjob": "Other",
    "famrel": 4, "famsup": True,
    "studytime": "2–5 hours", "failures": 0, "absences": 4,
    "schoolsup": False, "paid": False, "higher": True,
    "internet": True, "activities": False, "nursery": True,
    "freetime": 3, "goout": 3, "Dalc": 1, "Walc": 2, "health": 4,
    "romantic": False, "traveltime": "<15 min",
    "use_g1": False, "G1": 10,
    "use_g2": False, "G2": 10,
}

# Mapping helpers
SEX_MAP       = {"Female": "F", "Male": "M"}
ADDRESS_MAP   = {"Urban": "U", "Rural": "R"}
FAMSIZE_MAP   = {">3 members": "GT3", "≤3 members": "LE3"}
PSTATUS_MAP   = {"Living together": "T", "Separated": "A"}
MEDU_MAP      = {"None": 0, "Primary (4th grade)": 1, "5th–9th grade": 2,
                 "Secondary Education": 3, "Higher Education": 4}
EDUJOB_MAP    = {"Teacher": "teacher", "Health": "health",
                 "Civil services": "services", "At home": "at_home", "Other": "other"}
STUDYTIME_MAP = {"<2 hours": 1, "2–5 hours": 2, "5–10 hours": 3, ">10 hours": 4}
TRAVELTIME_MAP= {"<15 min": 1, "15–30 min": 2, "30–60 min": 3, ">60 min": 4}
YN_MAP        = {True: "yes", False: "no"}


# ---------------------------------------------------------------------------
# Reset to defaults
# ---------------------------------------------------------------------------
def reset_defaults():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


# Initialise session state on first load
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------------------------------------------------------------------
# Sidebar — Input Form
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎓 Student Performance Predictor")
    st.caption("Fill in the student's details, then click **Predict**.")

    if st.button("🔄 Reset to Defaults", use_container_width=True):
        reset_defaults()
        st.rerun()

    st.divider()

    # ── Section A: Demographics ──────────────────────────────────────────
    st.markdown("## A · Demographics")

    age = st.slider("Age", 15, 22, st.session_state["age"], key="age",
        help="Student's age in years (15–22).")
    sex = st.radio("Gender", ["Female", "Male"], index=0 if st.session_state["sex"] == "Female" else 1,
        key="sex", horizontal=True,
        help="Student's gender as recorded in school records.")
    address = st.radio("Home address type", ["Urban", "Rural"],
        index=0 if st.session_state["address"] == "Urban" else 1,
        key="address", horizontal=True,
        help="Whether the student lives in an urban or rural area.")
    famsize = st.radio("Family size", [">3 members", "≤3 members"],
        index=0 if st.session_state["famsize"] == ">3 members" else 1,
        key="famsize", horizontal=True,
        help="Total number of people in the student's household.")
    Pstatus = st.radio("Parents' cohabitation status",
        ["Living together", "Separated"],
        index=0 if st.session_state["Pstatus"] == "Living together" else 1,
        key="Pstatus", horizontal=True,
        help="Whether the student's parents live together or are separated.")

    # ── Section B: Family Background ─────────────────────────────────────
    st.markdown("## B · Family Background")

    edu_options = ["None", "Primary (4th grade)", "5th–9th grade",
                   "Secondary Education", "Higher Education"]
    Medu = st.selectbox("Mother's education level", edu_options,
        index=edu_options.index(st.session_state["Medu"]), key="Medu",
        help="Highest level of education completed by the student's mother.")
    Fedu = st.selectbox("Father's education level", edu_options,
        index=edu_options.index(st.session_state["Fedu"]), key="Fedu",
        help="Highest level of education completed by the student's father.")

    job_options = ["Teacher", "Health", "Civil services", "At home", "Other"]
    Mjob = st.selectbox("Mother's occupation", job_options,
        index=job_options.index(st.session_state["Mjob"]), key="Mjob",
        help="Mother's current or most recent occupation category.")
    Fjob = st.selectbox("Father's occupation", job_options,
        index=job_options.index(st.session_state["Fjob"]), key="Fjob",
        help="Father's current or most recent occupation category.")

    famrel = st.slider("Quality of family relationships", 1, 5,
        st.session_state["famrel"], key="famrel",
        help="1 = Very bad, 5 = Excellent — how good are relationships within the family?")
    famsup = st.checkbox("Family provides educational support",
        value=st.session_state["famsup"], key="famsup",
        help="Does the student's family actively help with studies?")

    # ── Section C: Academic History ───────────────────────────────────────
    st.markdown("## C · Academic History")

    st_options = ["<2 hours", "2–5 hours", "5–10 hours", ">10 hours"]
    studytime = st.selectbox("Weekly study time", st_options,
        index=st_options.index(st.session_state["studytime"]), key="studytime",
        help="How many hours per week does the student study outside school?")

    failures = st.slider("Number of past class failures", 0, 3,
        st.session_state["failures"], key="failures",
        help="How many courses has the student failed in previous years? (0–3+)")

    absences = st.slider("School absences (days)", 0, 93,
        st.session_state["absences"], key="absences",
        help="Total number of school days the student was absent this year.")

    schoolsup = st.checkbox("Receives extra educational support from school",
        value=st.session_state["schoolsup"], key="schoolsup",
        help="Does the school provide additional academic support to this student?")
    paid = st.checkbox("Attends extra paid tutoring classes",
        value=st.session_state["paid"], key="paid",
        help="Does the student attend privately paid extra classes in the subject?")
    higher = st.checkbox("Wants to pursue higher education",
        value=st.session_state["higher"], key="higher",
        help="Does the student aspire to go to university or college?")
    internet = st.checkbox("Has internet access at home",
        value=st.session_state["internet"], key="internet",
        help="Does the student have internet at home for study purposes?")
    activities = st.checkbox("Participates in extracurricular activities",
        value=st.session_state["activities"], key="activities",
        help="Is the student involved in school clubs, sports, or other activities?")
    nursery = st.checkbox("Attended nursery/preschool",
        value=st.session_state["nursery"], key="nursery",
        help="Did the student attend nursery school as a child?")

    # ── Section D: Lifestyle & Social ────────────────────────────────────
    st.markdown("## D · Lifestyle & Social")

    freetime = st.slider("Free time after school", 1, 5,
        st.session_state["freetime"], key="freetime",
        help="1 = Very little, 5 = Very much — how much free time does the student have?")
    goout = st.slider("Going out with friends frequency", 1, 5,
        st.session_state["goout"], key="goout",
        help="1 = Rarely, 5 = Very frequently — how often does the student go out socially?")
    Dalc = st.slider("Workday alcohol consumption", 1, 5,
        st.session_state["Dalc"], key="Dalc",
        help="1 = None/Very low, 5 = Very high — alcohol consumption on school days.")
    Walc = st.slider("Weekend alcohol consumption", 1, 5,
        st.session_state["Walc"], key="Walc",
        help="1 = None/Very low, 5 = Very high — alcohol consumption on weekends.")
    health = st.slider("Current health status", 1, 5,
        st.session_state["health"], key="health",
        help="1 = Very bad, 5 = Very good — student's self-reported health.")
    romantic = st.checkbox("Currently in a romantic relationship",
        value=st.session_state["romantic"], key="romantic",
        help="Is the student currently in a romantic relationship?")

    tt_options = ["<15 min", "15–30 min", "30–60 min", ">60 min"]
    traveltime = st.selectbox("Travel time from home to school", tt_options,
        index=tt_options.index(st.session_state["traveltime"]), key="traveltime",
        help="How long does the student's commute to school take?")

    # ── Section E: Prior Grades (optional) ───────────────────────────────
    st.markdown("## E · Prior Grades (Optional)")
    st.caption("Leave unchecked if grades are unknown — the model will use typical values.")

    use_g1 = st.checkbox("Include 1st period grade (G1)",
        value=st.session_state["use_g1"], key="use_g1",
        help="Check this box if you know the student's first semester grade.")
    if use_g1:
        G1 = st.slider("1st Period Grade (G1)", 0, 20,
            st.session_state["G1"], key="G1",
            help="Grade from the first assessment period (0–20 scale).")
    else:
        G1 = None

    use_g2 = st.checkbox("Include 2nd period grade (G2)",
        value=st.session_state["use_g2"], key="use_g2",
        help="Check this box if you know the student's second semester grade.")
    if use_g2:
        G2 = st.slider("2nd Period Grade (G2)", 0, 20,
            st.session_state["G2"], key="G2",
            help="Grade from the second assessment period (0–20 scale).")
    else:
        G2 = None

    st.divider()
    predict_btn = st.button("🚀 Predict Performance", use_container_width=True, type="primary")


# ---------------------------------------------------------------------------
# Main Panel
# ---------------------------------------------------------------------------
st.markdown("# 🎓 Student Academic Performance Predictor")
st.markdown(
    "Enter a student's details in the sidebar, then click **Predict Performance** "
    "to see whether they are on track to **Pass** or at **risk of Failing**, "
    "along with a personalised AI explanation."
)

# ── Error if models missing ───────────────────────────────────────────────
if missing_files:
    st.error(
        f"⚠️ Model files not found: `{'`, `'.join(missing_files)}`\n\n"
        "Please run the following commands from the project root first:\n"
        "```bash\n"
        "python src/data_preprocessing.py\n"
        "python src/train_model.py\n"
        "python src/evaluate.py\n"
        "```"
    )
    st.stop()

# ── Show placeholder until user predicts ─────────────────────────────────
if not predict_btn:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <p>Prediction</p>
            <h1 style="color:#94a3b8">—</h1>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <p>Pass Probability</p>
            <h1 style="color:#94a3b8">—</h1>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <p>Fail Risk Score</p>
            <h1 style="color:#94a3b8">—</h1>
        </div>""", unsafe_allow_html=True)

    st.info("👈 Fill in the student's details in the sidebar, then click **Predict Performance**.")
    st.stop()

# ── Build raw input dict ──────────────────────────────────────────────────
raw_input = {
    "school":     "GP",                           # default school (most common)
    "sex":        SEX_MAP[sex],
    "age":        age,
    "address":    ADDRESS_MAP[address],
    "famsize":    FAMSIZE_MAP[famsize],
    "Pstatus":    PSTATUS_MAP[Pstatus],
    "Medu":       MEDU_MAP[Medu],
    "Fedu":       MEDU_MAP[Fedu],
    "Mjob":       EDUJOB_MAP[Mjob],
    "Fjob":       EDUJOB_MAP[Fjob],
    "reason":     "course",                        # most common value
    "guardian":   "mother",                        # most common value
    "traveltime": TRAVELTIME_MAP[traveltime],
    "studytime":  STUDYTIME_MAP[studytime],
    "failures":   failures,
    "schoolsup":  YN_MAP[schoolsup],
    "famsup":     YN_MAP[famsup],
    "paid":       YN_MAP[paid],
    "activities": YN_MAP[activities],
    "nursery":    YN_MAP[nursery],
    "higher":     YN_MAP[higher],
    "internet":   YN_MAP[internet],
    "romantic":   YN_MAP[romantic],
    "famrel":     famrel,
    "freetime":   freetime,
    "goout":      goout,
    "Dalc":       Dalc,
    "Walc":       Walc,
    "health":     health,
    "absences":   absences,
    "G1":         G1,
    "G2":         G2,
}

# ── Run prediction ────────────────────────────────────────────────────────
with st.spinner("Running prediction …"):
    try:
        result = run_prediction(raw_input)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

label      = result["label"]
pass_prob  = result["pass_prob"]
risk_score = result["risk_score"]
shap_list  = result["shap"]
X_proc     = result["X_proc"]

# ── Results header ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 📊 Prediction Results")

col1, col2, col3 = st.columns(3)

badge_html = (
    f'<div class="badge-pass">✅ PASS</div>' if label == "Pass"
    else f'<div class="badge-fail">❌ FAIL</div>'
)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <p>Prediction</p>
        {badge_html}
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <p>Pass Probability</p>
        <h1 style="color: {'#22c55e' if pass_prob >= 0.5 else '#ef4444'}">
            {pass_prob*100:.1f}%
        </h1>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <p>Fail Risk Score</p>
        <h1 style="color: {'#ef4444' if risk_score >= 0.5 else '#22c55e'}">
            {risk_score*100:.1f}%
        </h1>
    </div>""", unsafe_allow_html=True)

# ── Risk progress bar ─────────────────────────────────────────────────────
st.markdown("### 🎯 Risk Level")
risk_pct = int(risk_score * 100)
bar_color = "#22c55e" if risk_pct < 30 else "#f59e0b" if risk_pct < 60 else "#ef4444"
risk_label = "Low Risk" if risk_pct < 30 else "Moderate Risk" if risk_pct < 60 else "High Risk"
st.markdown(f"""
<div style="background:#1e293b; border-radius:999px; height:24px; overflow:hidden; margin:0.5rem 0;">
  <div style="width:{risk_pct}%; background:{bar_color}; height:100%; border-radius:999px;
              transition:width 0.5s ease; display:flex; align-items:center; justify-content:center;">
    <span style="color:white; font-size:0.8rem; font-weight:600;">{risk_pct}% — {risk_label}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── SHAP Explanations ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🔍 Why This Prediction? (AI Explanation)")

if shap_list:
    pos_factors = [x for x in shap_list if x["shap_value"] > 0][:3]
    neg_factors = [x for x in shap_list if x["shap_value"] < 0][:3]

    col_pos, col_neg = st.columns(2)

    with col_pos:
        st.markdown("#### ✅ Factors Supporting PASS")
        if pos_factors:
            for f in pos_factors:
                st.markdown(
                    f'<div class="factor-pos">📈 <b>{f["feature_name"]}</b> '
                    f'&nbsp;(+{f["shap_value"]:.3f} SHAP)</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No strong pass-supporting factors found.")

    with col_neg:
        st.markdown("#### ❌ Factors Indicating RISK")
        if neg_factors:
            for f in neg_factors:
                st.markdown(
                    f'<div class="factor-neg">📉 <b>{f["feature_name"]}</b> '
                    f'&nbsp;({f["shap_value"]:.3f} SHAP)</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No strong risk factors found.")

    # Force plot
    if X_proc is not None:
        st.markdown("#### 🧮 SHAP Force Plot (Detailed Feature Impact)")
        st.caption(
            "The force plot shows how each feature pushes the prediction "
            "toward Pass (right/pink) or Fail (left/blue)."
        )
        try:
            import matplotlib.pyplot as plt
            import shap
            from src.evaluate import _clean_name

            expected_value = explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)):
                expected_value = float(expected_value[1] if len(expected_value) == 2 else expected_value[0])
            else:
                expected_value = float(expected_value)

            shap_vals = explainer.shap_values(X_proc)
            if isinstance(shap_vals, list) and len(shap_vals) == 2:
                shap_vals = shap_vals[1]
            elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                shap_vals = shap_vals[:, :, 1]
            row_shap = shap_vals[0] if shap_vals.ndim == 2 else shap_vals

            clean_names = [_clean_name(n) for n in model_pkg["feature_names"]]

            # Generate the plot (do not rely on its return value)
            shap.force_plot(
                expected_value,
                row_shap,
                features=X_proc[0],
                feature_names=clean_names,
                matplotlib=True,
                show=False,
            )

            # Explicitly grab the current active figure
            fig = plt.gcf()
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"Failed to generate SHAP plot: {e}")
else:
    st.info(
        "SHAP explanations are not available yet. "
        "Run `python src/evaluate.py` to generate the SHAP explainer, "
        "then restart the app."
    )

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "🎓 **Student Academic Performance Predictor** | "
    f"Model: `{model_pkg['model_name'] if model_pkg else 'N/A'}` | "
    "Data: UCI Student Performance Dataset (Math)"
)
