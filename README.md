> This project was designed, implemented, and deployed by Akshit Sharma.

# 🎓 Student Academic Risk Prediction System

An end-to-end Machine Learning system that identifies students at risk of academic failure using academic, behavioral, and family-background data, so schools can intervene early rather than reactively.

---

## ⚠️ A Note on the Target Variable (please read first)

This project uses the public **UCI Student Performance dataset**, which does **not** contain a literal dropout/withdrawal field. The target used throughout, `at_risk`, is defined as:

```
at_risk = 1 if final grade (G3) < 10 out of 20, else 0
```

This is **risk of academic failure**, used here as a practical, commonly-used proxy for dropout risk in education ML research — a student who is consistently failing is at meaningfully elevated dropout risk in the real world, but this is not the same as observed withdrawal data. This distinction is stated plainly here rather than implied, since claiming "dropout prediction" without this caveat would be a misleading description of what the model actually does.

---

## 🚀 Live Demo

- **Streamlit App:** https://student-dropout-prediction-dqtjjqebhkz3m9jvyafjot.streamlit.app/
- **Flask REST API:** runs locally / can be deployed separately (see [Deployment](#-deployment))
---

## 📌 Problem Statement

Manual monitoring of student performance doesn't scale well, and early identification of at-risk students is difficult, especially across large student populations. The challenge:

- Detect academic-failure risk early
- Minimize missed at-risk students (false negatives are costly — a missed student gets no intervention)
- Avoid over-flagging safe students (false positives waste limited intervention resources)

---

## 🧠 System Design: Two Models for Two Points in the School Year

A single model can't be both "usable on day one" and "maximally accurate" — the information that makes a model accurate (grades) doesn't exist yet at the start of term. So this system ships **two models**, matching two real deployment moments:

| Model | When it's usable | Inputs | Cross-validated ROC-AUC |
|---|---|---|---|
| **Screener** | Day 1 of term, no grades needed | Demographics, study habits, family background, past failures, absences | **~0.71** |
| **Early-Warning** | After the first two grading periods | Everything above **+ G1 and G2** (period grades) | **~0.97** |

Using G1/G2 is **not label leakage** — they are recorded before the final grade G3 that defines the target, so they're a legitimate leading indicator (the same logic as using a mid-semester quiz score to flag risk before finals), not information from the future. It does mean the Early-Warning model can only be used once those grades exist, which is exactly why it's offered as a *second* mode rather than the only model.

---

## 🔧 Enhancements Made (Performance & Correctness Review)

A review of the original single-model version surfaced three real issues, all fixed here:

**1. Deployed threshold didn't match the validated threshold — the project's core claim was silently undone in production.** The original notebook tuned and validated a 0.35 decision threshold, improving recall from 62% to 88% on its test split. But both deployed apps (`backend/app.py`, `streamlit_app/app.py`) hardcoded `THRESHOLD = 0.6` — a value that was **never actually tested** in the notebook. Verifying it directly: at threshold 0.6, real recall was only **46.2%** — worse than doing nothing extra (the default 0.5 threshold), and the exact opposite of the project's stated goal of minimizing missed at-risk students. Fix: `backend/utils/config.py` is now the single source of truth for both thresholds, imported by the training script, the Flask API, and the Streamlit app — the same class of bug cannot happen again, because there is only one place these numbers are defined.

**2. Only 8 of 30 available columns were used, with no cross-validation.** The original model used a single ~79-row test split (small and high-variance for a 395-row dataset) and ignored columns like parental education, alcohol consumption, family relationship quality, and school/family background. Fix: added 5-fold stratified cross-validation for reliable comparisons, expanded to the full available feature set, and compared Logistic Regression, Random Forest, and XGBoost — XGBoost with the expanded feature set won (~0.71 CV AUC vs. ~0.68 for the original 8-feature Logistic Regression).

**3. No path to using period grades (G1/G2) even though they're legitimate, powerful signals.** Adding them (as a *second*, opt-in model, not a replacement) raised cross-validated ROC-AUC from ~0.71 to **~0.97** — by far the single biggest lever available in this dataset, and one the original version never explored.

*(All numbers above are from an actual re-run of training and 5-fold cross-validation on this dataset — see `notebooks/02_model_enhancement.ipynb` for the full analysis.)*

---

## 📈 Results: Before vs. After

| | Original (single 8-feature LR) | Screener (enhanced) | Early-Warning (enhanced) |
|---|---:|---:|---:|
| Features used | 8 | 21 (13 numeric + 8 binary) | 23 (+ G1, G2) |
| Evaluation | single 79-row split | 5-fold CV | 5-fold CV |
| ROC-AUC | ~0.68 | **~0.71** | **~0.97** |
| Deployed threshold | 0.6 (never validated) | 0.35 (validated on OOF predictions) | 0.40 (validated on OOF predictions) |
| Real recall at deployed threshold | **46.2%** (bug) | **~78%** | **~94%** |
| Real precision at deployed threshold | 70.6% | ~43% | ~82% |

The Early-Warning model's near-94% recall *and* ~82% precision means very few at-risk students are missed, without flooding staff with false alarms — a genuinely strong result, made possible mainly by using G1/G2 rather than by better tuning alone. The Screener model is intentionally more modest (moderate ~0.71 AUC) since day-1 behavioral/demographic data has real, honest limits — it's designed as a coarse first-pass flag, not a precise diagnosis, and the app explicitly recommends re-checking with the Early-Warning model once grades are available.

---

## 🎯 Risk Categorization (Production Logic)

Instead of a bare binary output, predictions are grouped into risk buckets for actionable triage:

| Probability Range | Risk Level |
|---|---|
| < 0.4 | Low Risk |
| 0.4 – 0.6 | Medium Risk |
| ≥ 0.6 | High Risk |

The bucket is informational context; the actual At-Risk / Not-At-Risk decision uses the mode-specific validated threshold (0.35 or 0.40) described above, not the bucket boundaries.

---

## 📂 Project Structure

```
student-dropout-prediction/
│
├── backend/
│   ├── app.py                    # Flask REST API (both modes)
│   ├── requirements.txt
│   └── utils/
│       ├── config.py             # Shared feature lists + thresholds (single source of truth)
│       └── preprocess.py         # Shared data loading + feature-row construction
│
├── streamlit_app/
│   └── app.py                    # Streamlit UI (both modes, mode toggle)
│
├── models/
│   ├── screener_model.pkl
│   └── early_warning_model.pkl
│
├── notebooks/
│   ├── 01_data_overview.ipynb           # Original EDA + first model
│   └── 02_model_enhancement.ipynb       # This enhancement pass: CV, expanded features, G1/G2, threshold fix
│
├── data/
│   └── student_data.csv
│
├── train_models.py               # Trains + saves both production models
├── requirements.txt               # Root-level, for Streamlit Cloud deployment
└── README.md
```

---

## ⚙️ Tech Stack

- Python, Pandas, NumPy
- Scikit-learn (preprocessing, cross-validation, Logistic Regression, Random Forest)
- XGBoost (final model for both Screener and Early-Warning)
- Flask (REST API)
- Streamlit (interactive demo UI)

---

## 🔌 API Examples

### Screener mode (no grades required)
**Request** — `POST /predict`
```json
{
  "mode": "screener",
  "studytime": 2,
  "failures": 0,
  "absences": 4,
  "schoolsup": 1,
  "famsup": 1,
  "higher": 1,
  "internet": 1
}
```
**Response**
```json
{
  "mode": "screener",
  "prediction": "Not At Risk",
  "risk_level": "Medium Risk",
  "probability": 0.437,
  "threshold_used": 0.35
}
```

### Early-Warning mode (after 2 grading periods)
**Request** — `POST /predict`
```json
{
  "mode": "early_warning",
  "studytime": 2,
  "failures": 1,
  "absences": 10,
  "G1": 6,
  "G2": 5
}
```
**Response**
```json
{
  "mode": "early_warning",
  "prediction": "At Risk",
  "risk_level": "High Risk",
  "probability": 0.957,
  "threshold_used": 0.4
}
```

Any field not supplied falls back to a dataset-derived median/majority-class default (documented in `backend/utils/preprocess.py`) rather than 0, since fields like age or parental education have valid ranges where 0 is not a real value the model was trained on.

---

## ▶️ How to Run Locally

**Retrain both models** (after any data/code change):
```bash
pip install -r requirements.txt
python train_models.py
```

**Run the Streamlit app:**
```bash
streamlit run streamlit_app/app.py
```

---

## 🧠 What This Project Demonstrates

- Honest framing of a proxy target variable rather than overclaiming what the data supports
- Catching and fixing a real notebook-vs-production threshold mismatch that silently reversed the project's stated goal
- Cross-validated model comparison instead of relying on a single small test split
- Correctly distinguishing legitimate leading-indicator features (G1/G2, known before the outcome) from label leakage
- A shared-config architecture so training and both serving surfaces (API + UI) can't drift apart
- Multi-mode system design for a real deployment constraint (information availability changes over the school year)

---

## 🔮 Future Improvements

- Add SHAP-based explainability for individual predictions (which factors drove a specific student's risk score)
- Add authentication + a small database for a real institutional deployment (currently stateless, single-prediction)
- Explore a third "mid-warning" mode using only G1 (available even earlier than G1+G2)
- Add automated regression tests that fail CI if retraining drops ROC-AUC below a floor
- Validate against a genuine longitudinal dropout dataset if one becomes available, to test how well the G3-based proxy actually correlates with real withdrawal outcomes

---

## 👤 Author

Akshit Sharma
B.Tech | Data Science & Machine Learning
GitHub: https://github.com/akshitsharma009
LinkedIn: https://www.linkedin.com/in/akshit-sharma-7427362a0
