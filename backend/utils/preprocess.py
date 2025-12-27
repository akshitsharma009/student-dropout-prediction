import numpy as np

# EXACT order used during training
FEATURE_ORDER = [
    "studytime",
    "failures",
    "absences",
    "schoolsup",
    "famsup",
    "paid",
    "higher",
    "internet"
]

# Binary features that must be 0 or 1
BINARY_FEATURES = [
    "schoolsup",
    "famsup",
    "paid",
    "higher",
    "internet"
]

def preprocess_input(data: dict):
    features = []

    for feature in FEATURE_ORDER:
        if feature not in data:
            raise ValueError(f"Missing feature: {feature}")

        value = data[feature]

        # binary safety
        if feature in BINARY_FEATURES:
            if value in [1, "1", True, "yes", "Yes", "YES"]:
                value = 1
            elif value in [0, "0", False, "no", "No", "NO"]:
                value = 0
            else:
                raise ValueError(f"{feature} must be 0 or 1")

        features.append(float(value))

    return np.array(features).reshape(1, -1)
