import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    k_neighbors: int = 5,
):
    """
    Parameters
    ----------
    X_train      : feature matrix (training split only)
    y_train      : target vector  (training split only)
    random_state : reproducibility seed
    k_neighbors  : number of nearest neighbours used by SMOTE

    Returns
    -------
    X_bal : np.ndarray  — balanced feature matrix
    y_bal : np.ndarray  — balanced target vector
    """
    print(f"  Before SMOTE — Class 1 (Sick): {(y_train==1).sum():>4}  "
          f"| Class 0 (Healthy): {(y_train==0).sum():>4}  "
          f"| Total: {len(y_train)}")

    smote = SMOTE(
        sampling_strategy="auto",   # balance to match majority class
        k_neighbors=k_neighbors,
        random_state=random_state,
    )
    X_bal, y_bal = smote.fit_resample(X_train, y_train)

    print(f"  After  SMOTE — Class 1 (Sick): {(y_bal==1).sum():>4}  "
          f"| Class 0 (Healthy): {(y_bal==0).sum():>4}  "
          f"| Total: {len(y_bal)}")
    print(f"  Synthetic samples created: {len(y_bal) - len(y_train)}")

    return X_bal, y_bal
