"""
Model comparison: Logistic Regression vs Random Forest.
Evaluates both models via 5-fold cross-validation (mean ± std),
and saves confusion matrices and ROC curves to the figures/ directory.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay

from data_utils import FEATURES, TARGET, load_data, clean_data

FIGURES_DIR = 'figures'
N_FOLDS = 5
SCORING = ['accuracy', 'precision', 'recall', 'f1']

os.makedirs(FIGURES_DIR, exist_ok=True)


# --- Models ---

def build_models():
    lr = Pipeline([
        ('scaler', MinMaxScaler()),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    return [('Logistic Regression', lr), ('Random Forest', rf)]


# --- Cross-validation ---

def run_cross_validation(models, X, y):
    results = []
    for name, model in models:
        cv = cross_validate(model, X, y, cv=N_FOLDS, scoring=SCORING)
        results.append({
            'Model':     name,
            'Accuracy':  f"{cv['test_accuracy'].mean():.3f} ± {cv['test_accuracy'].std():.3f}",
            'Precision': f"{cv['test_precision'].mean():.3f} ± {cv['test_precision'].std():.3f}",
            'Recall':    f"{cv['test_recall'].mean():.3f} ± {cv['test_recall'].std():.3f}",
            'F1-Score':  f"{cv['test_f1'].mean():.3f} ± {cv['test_f1'].std():.3f}",
        })
    return pd.DataFrame(results).set_index('Model')


# --- Plots ---

def plot_confusion_matrices(models, X, y):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Confusion Matrices ({N_FOLDS}-fold CV predictions)', fontsize=14, fontweight='bold')

    for ax, (name, model) in zip(axes, models):
        y_pred = cross_val_predict(model, X, y, cv=N_FOLDS)
        cm = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Breakdown', 'Breakdown'])
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(name, fontsize=12)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'confusion_matrices.png')
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


def plot_roc_curves(models, X, y):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'ROC Curves ({N_FOLDS}-fold CV predictions)', fontsize=14, fontweight='bold')

    for ax, (name, model) in zip(axes, models):
        proba = cross_val_predict(model, X, y, cv=N_FOLDS, method='predict_proba')[:, 1]
        fpr, tpr, _ = roc_curve(y, proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC = {roc_auc:.3f}')
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--', lw=1)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(name, fontsize=12)
        ax.legend(loc='lower right')
        ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'roc_curves.png')
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


# --- Save final models (trained on full dataset) ---

def save_models(models, X, y):
    for name, model in models:
        model.fit(X, y)

    lr_pipeline = dict(models)['Logistic Regression']
    rf = dict(models)['Random Forest']

    joblib.dump(lr_pipeline.named_steps['clf'], 'models/modele_final.pkl')
    joblib.dump(lr_pipeline.named_steps['scaler'], 'models/scaler_final.pkl')
    joblib.dump(rf, 'models/modele_forest.pkl')
    print("Models saved to models/")


# --- Main ---

if __name__ == "__main__":
    print("=" * 60)
    print("  MODEL COMPARISON — Logistic Regression vs Random Forest")
    print(f"  Evaluation: {N_FOLDS}-fold cross-validation")
    print("=" * 60)

    df = load_data()
    df_clean = clean_data(df)

    X = df_clean[FEATURES]
    y = df_clean[TARGET]

    models = build_models()

    results_df = run_cross_validation(models, X, y)
    print("\n--- METRIC COMPARISON (mean ± std) ---")
    print(results_df.to_string())

    plot_confusion_matrices(models, X, y)
    plot_roc_curves(models, X, y)

    save_models(models, X, y)
    print("=" * 60)
