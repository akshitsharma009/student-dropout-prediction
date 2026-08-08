"""
train_models.py
----------------
Trains and saves BOTH production models on the full dataset:

1. Screener model  -- usable from day one of term, no grades required.
2. Early-Warning model -- usable once the first two grading periods
   (G1, G2) are recorded; substantially more accurate since period
   grades are strong, temporally-prior signals of the final outcome.

Run from the project root:
    python train_models.py
"""

import joblib
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from backend.utils.config import (
    NUMERIC_FEATURES_SCREENER,
    NUMERIC_FEATURES_EARLY_WARNING,
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    MODEL_PATH_SCREENER,
    MODEL_PATH_EARLY_WARNING,
)
from backend.utils.preprocess import load_and_clean, build_preprocessor

df = load_and_clean("data/student_data.csv")
y = df["at_risk"]

scale_pos_weight = (y == 0).sum() / (y == 1).sum()

# Hyperparameters selected via RandomizedSearchCV with 5-fold stratified
# cross-validation, optimizing ROC-AUC (see notebooks/02_model_enhancement.ipynb).

SCREENER_PARAMS = dict(
    n_estimators=150, max_depth=4, learning_rate=0.02,
    subsample=0.7, colsample_bytree=0.7,
)
EARLY_WARNING_PARAMS = dict(
    n_estimators=200, max_depth=2, learning_rate=0.02,
    subsample=0.85, colsample_bytree=0.85,
)


def build_and_train(numeric_features, xgb_params, save_path):
    X = df[numeric_features + BINARY_FEATURES + CATEGORICAL_FEATURES]
    preprocess = build_preprocessor(numeric_features)
    model = XGBClassifier(
        eval_metric="logloss", random_state=42,
        scale_pos_weight=scale_pos_weight, n_jobs=1, **xgb_params,
    )
    clf = Pipeline([("preprocess", preprocess), ("model", model)])
    clf.fit(X, y)
    joblib.dump(clf, f"models/{save_path}")
    print(f"Saved models/{save_path} | trained on {len(X)} rows, "
          f"{len(numeric_features)} numeric + {len(BINARY_FEATURES)} binary + "
          f"{len(CATEGORICAL_FEATURES)} categorical features")
    return clf


print("Training Screener model (no grades)...")
build_and_train(NUMERIC_FEATURES_SCREENER, SCREENER_PARAMS, MODEL_PATH_SCREENER)

print("\nTraining Early-Warning model (with G1 + G2)...")
build_and_train(NUMERIC_FEATURES_EARLY_WARNING, EARLY_WARNING_PARAMS, MODEL_PATH_EARLY_WARNING)

print("\nBoth models trained and saved.")
