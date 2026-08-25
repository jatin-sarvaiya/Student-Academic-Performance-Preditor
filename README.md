# Student Academic Performance Predictor

An end-to-end machine learning project that predicts whether a student will **Pass or Fail** their Math course, using the UCI Student Performance dataset. It also outputs a **risk probability** to enable early academic intervention.

---

## 🗂️ Project Structure

```
student-performance-predictor/
├── data/                         # Raw and processed datasets
│   ├── student_data.csv          # Full dataset (UCI or synthetic)
│   ├── X_train.csv / X_test.csv  # Processed feature splits
│   └── y_train.csv / y_test.csv  # Target label splits
├── notebooks/
│   └── eda.ipynb                 # Exploratory Data Analysis
├── src/
│   ├── data_preprocessing.py     # Data loading, validation, pipeline
│   ├── train_model.py            # Model training & evaluation
│   ├── evaluate.py               # SHAP explainability
│   └── predict.py                # StudentPredictor inference class
├── app/
│   └── app.py                    # Streamlit demo app
├── models/
│   ├── preprocessing_pipeline.pkl
│   ├── best_model.pkl
│   ├── feature_metadata.pkl
│   ├── shap_explainer.pkl
│   ├── shap_summary.png
│   ├── shap_summary.txt
│   └── model_comparison.csv
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

- **Source**: [UCI Machine Learning Repository — Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance)  
- **File used**: `student-mat.csv` (Math class, 395 students, 33 features)  
- **Target**: `G3 >= 10` → **Pass (1)**, otherwise **Fail (0)**  
- **Fallback**: If the download fails, a synthetic dataset of 800 realistic rows is generated automatically, matching the exact schema and distributions of the original.

### Feature Groups
| Group | Features |
|-------|----------|
| Demographics | age, sex, address, famsize, Pstatus |
| Family | Medu, Fedu, Mjob, Fjob, guardian, famrel, famsup |
| Academic | studytime, failures, absences, schoolsup, paid, higher, internet, activities, nursery |
| Lifestyle | freetime, goout, Dalc, Walc, health, romantic, traveltime |
| Prior grades (optional) | G1, G2 |

---

## ⚙️ Setup

### 1. Clone / download the project
```bash
cd student-performance-predictor
```

### 2. (Recommended) Create a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Full Pipeline

### Step 1 — Preprocess Data
```bash
python src/data_preprocessing.py
```
Downloads the UCI dataset (or generates synthetic data), validates inputs,
builds the preprocessing pipeline, saves splits and `models/preprocessing_pipeline.pkl`.

### Step 2 — Train Models
```bash
python src/train_model.py
```
Trains and tunes Logistic Regression, Decision Tree, Random Forest, and XGBoost
via RandomizedSearchCV. Saves `models/best_model.pkl` and `models/model_comparison.csv`.

### Step 3 — Evaluate & SHAP Analysis
```bash
python src/evaluate.py
```
Computes global SHAP values, saves `models/shap_summary.png`, `models/shap_summary.txt`,
and `models/shap_explainer.pkl` (needed for per-prediction explanations in the app).

### Step 4 — Launch the Streamlit App
```bash
streamlit run app/app.py
```
Opens the demo app in your browser at `http://localhost:8501`.

---

## 🧪 Quick Inference Test
```python
from src.predict import StudentPredictor

predictor = StudentPredictor()
result = predictor.predict({
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
})
print(result["prediction_label"], result["pass_probability"])
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: preprocessing_pipeline.pkl` | Run `python src/data_preprocessing.py` first |
| `FileNotFoundError: best_model.pkl` | Run `python src/train_model.py` after preprocessing |
| SHAP explanations not showing in app | Run `python src/evaluate.py` to generate `shap_explainer.pkl` |
| Download fails — no data | Synthetic dataset auto-generated; run preprocessing again |
| `ModuleNotFoundError: xgboost` | Install: `pip install xgboost` or it auto-falls back to GradientBoosting |
| Streamlit crash on first load | Ensure you launch from the project root: `streamlit run app/app.py` |
| Column mismatch errors | Delete `data/X_train.csv` and re-run preprocessing and training |

---

## 📈 Model Performance (example with UCI data)

| Model | Test Accuracy | Test F1 | ROC-AUC |
|-------|--------------|---------|---------|
| Logistic Regression | ~0.72 | ~0.78 | ~0.75 |
| Decision Tree | ~0.70 | ~0.77 | ~0.69 |
| **Random Forest** | **~0.74** | **~0.81** | **~0.78** |
| XGBoost | ~0.73 | ~0.80 | ~0.77 |

*Exact values depend on the dataset version and random seed.*

---

## 👥 Team Ownership
| Module | Owner |
|--------|-------|
| `data_preprocessing.py` + `eda.ipynb` | Data Engineer |
| `train_model.py` + `evaluate.py` | ML Engineer |
| `app/app.py` + `predict.py` | App Developer |

---

*Built with ❤️ using scikit-learn, XGBoost, SHAP, and Streamlit.*
