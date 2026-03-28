import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, classification_report,
)

# Global plot style
plt.rcParams.update({
    "figure.dpi"    : 150,
    "font.family"   : "DejaVu Sans",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
})

PALETTE = {
    "Random Forest"     : "#2196F3",
    "XGBoost"           : "#4CAF50",
    "CatBoost"          : "#FF9800",
    "Stacking Ensemble" : "#9C27B0",
}


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score (y_test, y_pred)
    rec  = recall_score   (y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    f1   = f1_score       (y_test, y_pred)
    auc  = roc_auc_score  (y_test, y_proba)

    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Recall    : {rec:.4f}   ← most critical for medical use")
    print(f"    Precision : {prec:.4f}")
    print(f"    F1-Score  : {f1:.4f}")
    print(f"    AUC-ROC   : {auc:.4f}")
    print(f"\n  Classification Report:\n"
          f"{classification_report(y_test, y_pred, target_names=['Healthy','Sick'])}")

    return dict(accuracy=acc, recall=rec, precision=prec, f1=f1, auc=auc)


def plot_confusion_matrix(model, X_test, y_test,
                          title="Confusion Matrix",
                          save_path="outputs/figures/confusion_matrix.png"):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Predicted\nHealthy", "Predicted\nSick"],
                yticklabels=["Actual\nHealthy",   "Actual\nSick"],
                linewidths=0.5, ax=ax, cbar=False,
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.set_ylabel("Actual Label",    fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)

    # Annotate TN, FP, FN, TP
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.75, labels[i][j],
                    ha="center", va="center",
                    fontsize=9, color="grey")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_roc_curve(models: dict, X_test, y_test,
                   save_path="outputs/figures/roc_curve_comparison.png"):
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        lw  = 3 if name == "Stacking Ensemble" else 1.5
        ls  = "-" if name == "Stacking Ensemble" else "--"
        ax.plot(fpr, tpr, lw=lw, ls=ls,
                color=PALETTE.get(name, "grey"),
                label=f"{name}  (AUC = {auc:.3f})")

    ax.plot([0,1],[0,1], "k:", lw=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_title("ROC Curve — Model Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_feature_importance(rf_model, feature_names: list,
                            save_path="outputs/figures/feature_importance.png"):
    importances = rf_model.feature_importances_
    idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(feature_names)))[::-1]
    bars = ax.barh(
        [feature_names[i] for i in idx[::-1]],
        importances[idx[::-1]],
        color=colors, edgecolor="white", height=0.7
    )
    ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)", fontsize=11)
    ax.set_title("Random Forest — Feature Importance", fontsize=13, fontweight="bold")

    for bar, val in zip(bars, importances[idx[::-1]]):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_metrics_comparison(results: dict,
                            save_path="outputs/figures/metrics_comparison.png"):
    """Bar chart comparing all four models across all five metrics."""
    metrics = ["accuracy", "recall", "precision", "f1", "auc"]
    labels  = ["Accuracy", "Recall", "Precision", "F1-Score", "AUC-ROC"]
    models  = list(results.keys())

    x     = np.arange(len(labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 6))

    for i, model_name in enumerate(models):
        vals = [results[model_name][m] for m in metrics]
        bars = ax.bar(x + i*width, vals, width,
                      label=model_name,
                      color=PALETTE.get(model_name, "grey"),
                      edgecolor="white", alpha=0.9)
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                    f"{b.get_height():.2f}", ha="center",
                    va="bottom", fontsize=7)

    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_title("Model Performance Comparison — All Metrics",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")
