"""
Architecture
────────────
             ┌──────────────────┐
             │  Balanced        │
             │  Training Data   │
             └────────┬─────────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
    ┌──────────┐ ┌─────────┐ ┌──────────┐
    │  Random  │ │ XGBoost │ │ CatBoost │   ← Level-0  Base Models
    │  Forest  │ │         │ │          │
    └────┬─────┘ └────┬────┘ └────┬─────┘
         │            │           │
         └────────────┼───────────┘
                      │
              ┌───────▼────────┐
              │  Meta-Learner  │            ← Level-1  Stacking
              │  (Logistic     │
              │   Regression)  │
              └───────┬────────┘
                      │
               Final Prediction
"""

import numpy as np
from sklearn.ensemble        import RandomForestClassifier, StackingClassifier
from sklearn.linear_model    import LogisticRegression
from xgboost                 import XGBClassifier
from catboost                import CatBoostClassifier


RF_PARAMS = dict(
    n_estimators      = 300,
    max_depth         = 10,
    min_samples_split = 4,
    min_samples_leaf  = 2,
    class_weight      = "balanced",   # penalise errors on minority class
    n_jobs            = -1,
)

XGB_PARAMS = dict(
    n_estimators       = 300,
    learning_rate      = 0.05,
    max_depth          = 6,
    subsample          = 0.8,
    colsample_bytree   = 0.8,
    scale_pos_weight   = 1,           # adjust if imbalance persists post-SMOTE
    use_label_encoder  = False,
    eval_metric        = "logloss",
    verbosity          = 0,
)

CAT_PARAMS = dict(
    iterations         = 300,
    learning_rate      = 0.05,
    depth              = 6,
    l2_leaf_reg        = 3,
    verbose            = 0,
    allow_writing_files= False,
)

META_PARAMS = dict(
    C              = 1.0,
    max_iter       = 1000,
    solver         = "lbfgs",
    class_weight   = "balanced",
)


def train_base_models(X_train, y_train, random_state: int = 42):
    """
    Train Random Forest, XGBoost, and CatBoost independently
    on the SMOTE-balanced training data.

    Returns
    -------
    rf_model  : trained RandomForestClassifier
    xgb_model : trained XGBClassifier
    cat_model : trained CatBoostClassifier
    """
    # Random Forest
    rf = RandomForestClassifier(**RF_PARAMS, random_state=random_state)
    rf.fit(X_train, y_train)
    print(f"  Random Forest  trained  ({RF_PARAMS['n_estimators']} trees)")

    # XGBoost
    xgb = XGBClassifier(**XGB_PARAMS, random_state=random_state)
    xgb.fit(X_train, y_train)
    print(f"  XGBoost        trained  ({XGB_PARAMS['n_estimators']} estimators)")

    # CatBoost
    cat = CatBoostClassifier(**CAT_PARAMS, random_state=random_state)
    cat.fit(X_train, y_train)
    print(f"  CatBoost       trained  ({CAT_PARAMS['iterations']} iterations)")

    return rf, xgb, cat


def train_stacking_ensemble(
    rf_model, xgb_model, cat_model,
    X_train, y_train,
    random_state: int = 42,
    cv_folds: int = 5,
):
    """
    Build a Stacking Ensemble using the three pre-trained base models
    as level-0 estimators and Logistic Regression as the meta-learner.

    The StackingClassifier uses 5-fold cross-validation to generate
    out-of-fold predictions for training the meta-learner, which
    prevents overfitting.

    Returns
    -------
    stack_model : trained StackingClassifier
    """
    estimators = [
        ("rf",  rf_model),
        ("xgb", xgb_model),
        ("cat", cat_model),
    ]
    meta_learner = LogisticRegression(**META_PARAMS, random_state=random_state)

    stack = StackingClassifier(
        estimators        = estimators,
        final_estimator   = meta_learner,
        cv                = cv_folds,
        stack_method      = "predict_proba",  # use probability outputs, not hard labels
        passthrough       = False,
        n_jobs            = -1,
    )
    stack.fit(X_train, y_train)
    print(f"  Stacking Ensemble trained  (meta-learner: Logistic Regression, "
          f"CV folds: {cv_folds})")

    return stack
