"""
preprocess.py
-------------
Shared feature-engineering + pipeline-building logic for the Student
Academic Risk Prediction project. Used by both the training script and
(implicitly, via the saved pipeline objects) the Flask backend and
Streamlit app at inference time.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import BINARY_FEATURES, CATEGORICAL_FEATURES


def load_and_clean(raw_csv_path: str) -> pd.DataFrame:
    """
    Load the raw UCI Student Performance CSV and apply the fixed cleaning
    steps: parse yes/no binary columns to 0/1, and derive the modeling
    target.

    NOTE on the target: this dataset has no true "dropout" field. The
    target used throughout this project, `at_risk`, is defined as
    `G3 < 10` (final-year grade below 10/20, the standard local passing
    mark) -- i.e. risk of academic failure, used here as a proxy for
    dropout risk. This is a reasonable, commonly-used proxy in education
    ML research, but it is NOT literal withdrawal/dropout data, and that
    distinction should always be stated plainly rather than implied.
    """

    df = pd.read_csv(raw_csv_path, sep=";")
    df["at_risk"] = (df["G3"] < 10).astype(int)

    df[BINARY_FEATURES] = df[BINARY_FEATURES].replace({"yes": 1, "no": 0})

    return df


def build_preprocessor(numeric_features):
    """
    Build the ColumnTransformer used inside both the screener and
    early-warning pipelines. Numeric + binary columns are scaled;
    categorical columns are one-hot encoded.
    """

    return ColumnTransformer([
        ("num", StandardScaler(), numeric_features + BINARY_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def build_feature_row(data: dict, numeric_features: list) -> pd.DataFrame:
    """
    Build a single-row DataFrame from a dict of raw feature values (as
    received from the Flask API or the Streamlit form), matching the
    exact column set and dtypes the pipeline's ColumnTransformer expects.

    Any field not supplied by the caller falls back to a sensible,
    dataset-derived default (median for numeric fields, most common
    category for categoricals) rather than 0 -- fields like `age`,
    `Medu`/`Fedu` (parental education), or `health` have valid ranges
    (e.g. 1-5, or 15-22) where 0 is not a real value the model was ever
    trained on. Silently defaulting to 0 previously produced
    out-of-distribution rows and skewed predictions for any caller that
    omitted a field (see the equivalent bug fixed in the fraud-detection
    project's input_schema.py).
    """

    DEFAULT_CATEGORICAL = {
        "school": "GP", "sex": "F", "address": "U", "famsize": "GT3",
        "Pstatus": "T", "Mjob": "other", "Fjob": "other",
        "reason": "course", "guardian": "mother",
    }

    # Dataset medians for numeric fields (see data/student_data.csv).
    DEFAULT_NUMERIC = {
        "age": 17, "Medu": 3, "Fedu": 2, "traveltime": 1, "studytime": 2,
        "failures": 0, "famrel": 4, "freetime": 3, "goout": 3, "Dalc": 1,
        "Walc": 2, "health": 4, "absences": 4,
        # Early-warning only: median period grades in the dataset.
        "G1": 11, "G2": 11,
    }
    # Binary features default to the majority class in the dataset
    # (most students do have internet, aim for higher ed, etc.) rather
    # than always assuming "no".
    DEFAULT_BINARY = {
        "schoolsup": 0, "famsup": 1, "paid": 0, "activities": 1,
        "nursery": 1, "higher": 1, "internet": 1, "romantic": 0,
    }

    row = {}
    for col in numeric_features:
        row[col] = data.get(col, DEFAULT_NUMERIC.get(col, 0))
    for col in BINARY_FEATURES:
        row[col] = data.get(col, DEFAULT_BINARY.get(col, 0))
    for col in CATEGORICAL_FEATURES:
        row[col] = data.get(col, DEFAULT_CATEGORICAL[col])

    return pd.DataFrame([row])
