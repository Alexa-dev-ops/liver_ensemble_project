import argparse
import pickle
import numpy  as np
import pandas as pd

MODEL_PATH = "outputs/models/stacking_ensemble.pkl"

FEATURE_COLS = [
    "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
    "Alkaline_Phosphotase", "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase", "Total_Protiens",
    "Albumin", "Albumin_and_Globulin_Ratio",
]

# ── Example patients (representative values in original units) ────────
EXAMPLE_PATIENTS = pd.DataFrame([
    {   # Likely SICK — high bilirubin, low albumin
        "Age":55, "Gender":1,
        "Total_Bilirubin":3.5, "Direct_Bilirubin":1.8,
        "Alkaline_Phosphotase":320, "Alamine_Aminotransferase":80,
        "Aspartate_Aminotransferase":95, "Total_Protiens":5.8,
        "Albumin":2.9, "Albumin_and_Globulin_Ratio":0.75,
    },
    {   # Likely HEALTHY — all values in normal range
        "Age":34, "Gender":0,
        "Total_Bilirubin":0.8, "Direct_Bilirubin":0.2,
        "Alkaline_Phosphotase":95, "Alamine_Aminotransferase":22,
        "Aspartate_Aminotransferase":25, "Total_Protiens":7.1,
        "Albumin":4.2, "Albumin_and_Globulin_Ratio":1.2,
    },
    {   # Borderline — mild enzyme elevation
        "Age":48, "Gender":1,
        "Total_Bilirubin":1.4, "Direct_Bilirubin":0.5,
        "Alkaline_Phosphotase":180, "Alamine_Aminotransferase":55,
        "Aspartate_Aminotransferase":60, "Total_Protiens":6.4,
        "Albumin":3.5, "Albumin_and_Globulin_Ratio":0.95,
    },
])


def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same transforms used during training."""
    import numpy as np
    data = df.copy()
    # Encode gender if string
    if data["Gender"].dtype == object:
        data["Gender"] = data["Gender"].map({"Male":1,"Female":0}).fillna(0).astype(int)
    # Impute missing AGR
    if data["Albumin_and_Globulin_Ratio"].isnull().any():
        data["Albumin_and_Globulin_Ratio"].fillna(
            data["Albumin_and_Globulin_Ratio"].median(), inplace=True
        )
    # Log-transform skewed columns
    for c in ["Total_Bilirubin","Direct_Bilirubin","Alkaline_Phosphotase",
              "Alamine_Aminotransferase","Aspartate_Aminotransferase"]:
        data[c] = np.log1p(data[c])
    return data[FEATURE_COLS]


def predict(patients: pd.DataFrame, model_path: str = MODEL_PATH):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    X     = preprocess_input(patients)
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]

    print("\n  ┌─────┬──────────────────────────────────────┬─────────────┬────────────────┐")
    print("  │  #  │ Patient Summary                      │ Prediction  │ Confidence (%) │")
    print("  ├─────┼──────────────────────────────────────┼─────────────┼────────────────┤")
    for i, (pred, prob) in enumerate(zip(preds, proba)):
        age    = int(patients.iloc[i]["Age"])
        gender = "M" if patients.iloc[i]["Gender"] in [1,"Male"] else "F"
        label  = "⚠  SICK (1)" if pred == 1 else "✓  Healthy (0)"
        conf   = prob * 100 if pred == 1 else (1 - prob) * 100
        summary = f"Age={age}, Gender={gender}"
        print(f"  │  {i+1:<2} │ {summary:<36} │ {label:<11} │ {conf:>13.1f}% │")
    print("  └─────┴──────────────────────────────────────┴─────────────┴────────────────┘")
    print("\n  NOTE: This tool assists clinical decision-making. Always confirm")
    print("        with a licensed healthcare professional.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to CSV of new patient records")
    args = parser.parse_args()

    if args.csv:
        patients = pd.read_csv(args.csv)
        print(f"  Loaded {len(patients)} patient(s) from '{args.csv}'")
    else:
        patients = EXAMPLE_PATIENTS
        print("  Using built-in example patients (no --csv flag provided)")

    predict(patients)
