"""
Model comparison: Logistic Regression vs Random Forest.
Trains both models, prints a metric table, and saves confusion matrices
and ROC curves to the figures/ directory.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
)

FEATURES = ['km', 'etat', 'age_vehicule', 'nb_revisions', 'temperature_moteur']
TARGET = 'panne'
FIGURES_DIR = 'figures'
os.makedirs(FIGURES_DIR, exist_ok=True)


# --- Data loading ---

def load_data():
    try:
        df = pd.read_csv('data/donnees_brutes_vab.csv')
        print(f"Dataset loaded: {df.shape[0]} rows.")
        return df
    except FileNotFoundError:
        print("CSV not found. Generating fallback dataset...")
        data_fallback = {
            'id': range(101, 111),
            'km':               [15000, 45000, 12000, 80000, 32000, 65000, 22000, 95000, 40000, 28000],
            'etat':             [2,     1,     2,     0,     1,     0,     2,     0,     1,     2],
            'age_vehicule':     [2,     8,     3,     15,    6,     18,    4,     22,    7,     5],
            'nb_revisions':     [2,     7,     3,     12,    5,     14,    4,     18,    6,     4],
            'temperature_moteur':[75,   88,    72,   108,   85,   112,   73,   115,   90,    76],
            'panne':            [0,     0,     0,     1,     0,     1,     0,     1,     0,     0],
        }
        return pd.DataFrame(data_fallback)


def clean_data(df):
    df = df.drop_duplicates(subset=['id'], keep='first')
    for col in FEATURES:
        df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=[TARGET])
    df.loc[:, 'km'] = df['km'].fillna(df['km'].median())
    df['etat'] = df['etat'].fillna(2)
    df['age_vehicule'] = df['age_vehicule'].fillna(df['age_vehicule'].median())
    df['nb_revisions'] = df['nb_revisions'].fillna(df['nb_revisions'].median())
    df['temperature_moteur'] = df['temperature_moteur'].fillna(df['temperature_moteur'].median())
    df = df[(df['km'] >= 0) & (df['km'] <= 1_000_000)]
    return df


# --- Training ---

def train_models(X_train, y_train):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)

    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)

    return log_reg, rf, scaler


# --- Evaluation ---

def compute_metrics(name, y_true, y_pred):
    return {
        'Model': name,
        'Accuracy':  round(accuracy_score(y_true, y_pred), 4),
        'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'Recall':    round(recall_score(y_true, y_pred, zero_division=0), 4),
        'F1-Score':  round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


# --- Plots ---

def plot_confusion_matrices(y_test, y_pred_lr, y_pred_rf):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')

    for ax, y_pred, title in zip(
        axes,
        [y_pred_lr, y_pred_rf],
        ['Logistic Regression', 'Random Forest']
    ):
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Breakdown', 'Breakdown'])
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(title, fontsize=12)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'confusion_matrices.png')
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


def plot_roc_curves(y_test, proba_lr, proba_rf):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('ROC Curves', fontsize=14, fontweight='bold')

    for ax, proba, title in zip(
        axes,
        [proba_lr, proba_rf],
        ['Logistic Regression', 'Random Forest']
    ):
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC = {roc_auc:.3f}')
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--', lw=1)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title, fontsize=12)
        ax.legend(loc='lower right')
        ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'roc_curves.png')
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


# --- Main ---

if __name__ == "__main__":
    print("=" * 55)
    print("  MODEL COMPARISON — Logistic Regression vs Random Forest")
    print("=" * 55)

    df = load_data()
    df_clean = clean_data(df)

    X = df_clean[FEATURES]
    y = df_clean[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    log_reg, rf, scaler = train_models(X_train, y_train)

    X_test_scaled = scaler.transform(X_test)
    y_pred_lr = log_reg.predict(X_test_scaled)
    y_pred_rf = rf.predict(X_test)
    proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]
    proba_rf = rf.predict_proba(X_test)[:, 1]

    results = [
        compute_metrics('Logistic Regression', y_test, y_pred_lr),
        compute_metrics('Random Forest',       y_test, y_pred_rf),
    ]
    results_df = pd.DataFrame(results).set_index('Model')

    print("\n--- METRIC COMPARISON ---")
    print(results_df.to_string())

    plot_confusion_matrices(y_test, y_pred_lr, y_pred_rf)
    plot_roc_curves(y_test, proba_lr, proba_rf)

    joblib.dump(log_reg, 'models/modele_final.pkl')
    joblib.dump(scaler,  'models/scaler_final.pkl')
    joblib.dump(rf,      'models/modele_forest.pkl')
    print("\nModels saved to models/")
    print("=" * 55)
