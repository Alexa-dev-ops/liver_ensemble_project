import pandas as pd
import numpy as np
import sys

COLUMN_NAMES = [
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
    "Dataset",           # 1 = liver patient,  2 = healthy
]


def load_and_inspect(filepath: str) -> pd.DataFrame:
    """
    Load the ILPD CSV, attach column names, print a data quality
    report, and return the raw DataFrame. Raises FileNotFoundError if missing.
    """
    try:
        df = pd.read_csv(filepath, header=None, names=COLUMN_NAMES)
    except FileNotFoundError:
        print(f"  [ERROR] '{filepath}' not found. Please ensure the file exists.")
        sys.exit(1)

    print(f"\n  Shape            : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Columns          : {list(df.columns)}")
    print(f"\n  Class distribution (Dataset column):")
    vc = df["Dataset"].value_counts()
    for val, count in vc.items():
        label = "Liver Patient" if val == 1 else "Healthy"
        print(f"    {val} ({label:<15}) → {count} ({count/len(df)*100:.1f}%)")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        print(f"\n  Missing values:")
        for col, n in missing.items():
            print(f"    {col:<35} {n} missing")
    else:
        print("\n  Missing values   : None")

    print(f"\n  Data types:\n{df.dtypes.to_string()}")
    print(f"\n  Statistical summary:\n{df.describe().to_string()}")
    return df