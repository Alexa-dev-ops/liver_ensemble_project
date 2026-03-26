
import numpy  as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Columns with heavy right skew that benefit from log transform
_LOG_COLS = [
    "Total_Bilirubin",
    "Direct_Bilirubin",
    "Alkaline_Phosphotase",
    "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase",
]

_FEATURE_COLS = [
    "Age",
    "Gender",
    "Total_Bilirubin",
    "Direct_Bilirubin",
    "Alkaline_Phosphotase",
    "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase",
    "Total_Protiens",
    "Albumin",
    "Albumin_and_Globulin_Ratio",
]


def preprocess(df: pd.DataFrame):
    """
    Returns
    -------
    X : pd.DataFrame   — cleaned, scaled feature matrix
    y : pd.Series      — binary target (1 = sick, 0 = healthy)
    """
    data = df.copy()

    # 1 ── Encode gender
    data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})
    # Handle unexpected values
    data["Gender"] = pd.to_numeric(data["Gender"], errors="coerce").fillna(0).astype(int)

    # 2 ── Remap target  (ILPD: 1=patient, 2=healthy → binary 1/0)
    data["Dataset"] = data["Dataset"].map({1: 1, 2: 0})

    # 3 ── Impute missing Albumin_and_Globulin_Ratio with median
    col = "Albumin_and_Globulin_Ratio"
    median_val = data[col].median()
    n_missing  = data[col].isnull().sum()
    data[col] = data[col].fillna(median_val)
    if n_missing:
        print(f"  Imputed {n_missing} missing values in '{col}' with median={median_val:.3f}")

    # 4 ── Log-transform skewed columns  (log1p to handle zeros)
    for c in _LOG_COLS:
        data[c] = np.log1p(data[c])
    print(f"  Log1p transform applied to: {_LOG_COLS}")

    # 5 ── Remove extreme outliers using IQR (cap at 1.5 × IQR fence)
    n_before = len(data)
    for c in _FEATURE_COLS:
        Q1  = data[c].quantile(0.25)
        Q3  = data[c].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 3.0 * IQR
        upper = Q3 + 3.0 * IQR
        data[c] = data[c].clip(lower, upper)
    print(f"  Outlier capping applied (3×IQR fence).  Rows kept: {len(data)} / {n_before}")

    # 6 ── Separate features and target
    X = data[_FEATURE_COLS].copy()
    y = data["Dataset"].copy()

    # 7 ── Final NaN safety net — fill any remaining with column median
    for c in _FEATURE_COLS:
        if X[c].isnull().any():
            X = X.assign(**{c: X[c].fillna(X[c].median())})

    # 8 ── Standardise features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=_FEATURE_COLS,
        index=X.index,
    )

    print(f"  Feature matrix shape : {X_scaled.shape}")
    print(f"  Class counts  —  Sick (1): {(y==1).sum()}  |  Healthy (0): {(y==0).sum()}")
    print(f"  Imbalance ratio       : {(y==1).sum() / (y==0).sum():.2f} : 1")

    return X_scaled, y
