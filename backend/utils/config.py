"""
config.py
---------
Single source of truth for feature lists, categorical values, and decision
thresholds, shared by preprocessing, training, and both the Flask backend
and the Streamlit app -- so training and serving can never drift apart
(which was the root cause of the original bug: the notebook tuned a 0.35
threshold, but both deployed apps hardcoded 0.6).
"""

# ---------------------------------------------------------------
# Two deployable models, for two different points in the school year
# ---------------------------------------------------------------
# SCREENER: usable from day one of term -- no grades required yet.
# EARLY_WARNING: usable once the first two grading periods (G1, G2) are
# available -- far more accurate, since period grades are strong leading
# indicators of the final outcome and are known well before it.

NUMERIC_FEATURES_SCREENER = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
]
NUMERIC_FEATURES_EARLY_WARNING = NUMERIC_FEATURES_SCREENER + ["G1", "G2"]

BINARY_FEATURES = [
    "schoolsup", "famsup", "paid", "activities", "nursery",
    "higher", "internet", "romantic",
]

CATEGORICAL_FEATURES = [
    "school", "sex", "address", "famsize", "Pstatus",
    "Mjob", "Fjob", "reason", "guardian",
]

# Allowed categorical values, used for input validation in the API.
CATEGORICAL_VALUES = {
    "school": ["GP", "MS"],
    "sex": ["F", "M"],
    "address": ["U", "R"],
    "famsize": ["LE3", "GT3"],
    "Pstatus": ["T", "A"],
    "Mjob": ["teacher", "health", "services", "at_home", "other"],
    "Fjob": ["teacher", "health", "services", "at_home", "other"],
    "reason": ["home", "reputation", "course", "other"],
    "guardian": ["mother", "father", "other"],
}

# ---------------------------------------------------------------
# Decision thresholds
# ---------------------------------------------------------------
# Chosen from 5-fold cross-validated out-of-fold predictions (not a single
# ~79-row test split, and not tuned then ignored in production -- these
# are the exact thresholds both apps use).
THRESHOLD_SCREENER = 0.35        # prioritizes recall: ~78% recall / ~43% precision
THRESHOLD_EARLY_WARNING = 0.40   # ~94% recall / ~82% precision

MODEL_PATH_SCREENER = "screener_model.pkl"
MODEL_PATH_EARLY_WARNING = "early_warning_model.pkl"


def get_risk_bucket(probability: float) -> str:
    if probability < 0.4:
        return "Low Risk"
    elif probability < 0.6:
        return "Medium Risk"
    else:
        return "High Risk"
