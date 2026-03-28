"""
  Dataset  : Indian Liver Patient Dataset (ILPD) — UCI Repository
  Models   : Random Forest, XGBoost, CatBoost → Stacking Ensemble
  Balancing: SMOTE (Synthetic Minority Over-sampling Technique)
  Metrics  : Accuracy, Recall (Sensitivity), Precision, F1-Score, AUC-ROC

"""
import json
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

from data_loader    import load_and_inspect
from preprocessing  import preprocess
from balancing      import apply_smote
from model_training import train_base_models, train_stacking_ensemble
from evaluation     import evaluate_model, plot_confusion_matrix, plot_roc_curve, plot_feature_importance
from utils          import split_data, save_model, print_section

os.makedirs("outputs/models",  exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

def main():
    print_section("STEP 1: DATA LOADING & INSPECTION")
    df = load_and_inspect("data/Indian Liver Patient Dataset (ILPD).csv")

    print_section("STEP 2: PREPROCESSING")
    X, y = preprocess(df)

    print_section("STEP 3: TRAIN / TEST SPLIT  (80 / 20)")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.20, random_state=42)
    print(f"  Training samples : {X_train.shape[0]}")
    print(f"  Test samples     : {X_test.shape[0]}")

    print_section("STEP 4: SMOTE — BALANCING TRAINING DATA")
    X_train_bal, y_train_bal = apply_smote(X_train, y_train, random_state=42)

    print_section("STEP 5: TRAINING BASE MODELS")
    rf_model, xgb_model, cat_model = train_base_models(
        X_train_bal, y_train_bal, random_state=42
    )

    print_section("STEP 6: TRAINING STACKING ENSEMBLE")
    stack_model = train_stacking_ensemble(
        rf_model, xgb_model, cat_model,
        X_train_bal, y_train_bal, random_state=42
    )

    print_section("STEP 7: EVALUATION ON HELD-OUT TEST SET")
    models = {
        "Random Forest"     : rf_model,
        "XGBoost"           : xgb_model,
        "CatBoost"          : cat_model,
        "Stacking Ensemble" : stack_model,
    }
    results = {}
    for name, model in models.items():
        print(f"\n  ── {name} ──")
        metrics = evaluate_model(model, X_test, y_test, name)
        results[name] = metrics

    print_section("STEP 8: GENERATING VISUALISATIONS")
    plot_confusion_matrix(stack_model, X_test, y_test,
                          title="Stacking Ensemble — Confusion Matrix",
                          save_path="outputs/figures/confusion_matrix.png")

    plot_roc_curve(models, X_test, y_test,
                   save_path="outputs/figures/roc_curve_comparison.png")

    plot_feature_importance(rf_model, X.columns.tolist(),
                            save_path="outputs/figures/feature_importance.png")

    print_section("STEP 9: SAVING MODELS & DATA")
    for name, model in models.items():
        fname = name.lower().replace(" ", "_")
        save_model(model, f"outputs/models/{fname}.pkl")

    # Generate predictions for the ensemble for saving
    y_pred_ensemble = stack_model.predict(X_test)
    y_proba_ensemble = stack_model.predict_proba(X_test)[:, 1]

    # Save data arrays using variables in scope
    np.save("outputs/y_test.npy", y_test)
    np.save("outputs/y_pred_ensemble.npy", y_pred_ensemble)
    np.save("outputs/y_proba_ensemble.npy", y_proba_ensemble)
    np.save("outputs/feature_importances.npy", rf_model.feature_importances_)

    print_section("FINAL SUMMARY")
    print(f"  {'Model':<22} {'Accuracy':>10} {'Recall':>10} {'Precision':>10} {'F1-Score':>10} {'AUC-ROC':>10}")
    print("  " + "-"*72)
    for name, m in results.items():
        print(f"  {name:<22} {m['accuracy']:>10.4f} {m['recall']:>10.4f} "
              f"{m['precision']:>10.4f} {m['f1']:>10.4f} {m['auc']:>10.4f}")

    print("\n  All outputs saved to outputs/\n")

if __name__ == "__main__":
    main()