import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

FEATURES = ['km', 'etat', 'age_vehicule', 'nb_revisions', 'temperature_moteur']
TARGET = 'panne'


# 1. DATA CLEANING

def remove_duplicates(df):
    return df.drop_duplicates(subset=['id'], keep='first')


def convert_to_numeric(df):
    for col in FEATURES:
        df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')
    return df


def handle_missing_values(df):
    df = df.dropna(subset=[TARGET])
    if df['km'].isnull().all():
        print("WARNING: km column entirely empty, filling with 0")
        df['km'] = df['km'].fillna(0)
    else:
        df.loc[:, 'km'] = df['km'].fillna(df['km'].median())
    df['etat'] = df['etat'].fillna(2)
    df['age_vehicule'] = df['age_vehicule'].fillna(df['age_vehicule'].median())
    df['nb_revisions'] = df['nb_revisions'].fillna(df['nb_revisions'].median())
    df['temperature_moteur'] = df['temperature_moteur'].fillna(df['temperature_moteur'].median())
    return df


def filter_outliers(df):
    return df[(df['km'] >= 0) & (df['km'] <= 1_000_000)]


def clean_data(df):
    df = remove_duplicates(df)
    df = convert_to_numeric(df)
    df = handle_missing_values(df)
    df = filter_outliers(df)
    return df


# 2. MODEL TRAINING

def train_random_forest(df):
    X = df[FEATURES]
    y = df[TARGET]
    # n_estimators=100: 100 trees vote; max_depth=5 limits overfitting
    model_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model_rf.fit(X, y)
    return model_rf


def display_feature_importance(model):
    importances = model.feature_importances_
    print("Feature importances:")
    for name, imp in sorted(zip(FEATURES, importances), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.2%}")


# MAIN ORCHESTRATION

if __name__ == "__main__":

    print("--- RANDOM FOREST PIPELINE STARTED ---")

    def load_data():
        try:
            df_raw = pd.read_csv('data/donnees_brutes_vab.csv')
            print("CSV loaded successfully.")
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
            df_raw = pd.DataFrame(data_fallback)
        return df_raw

    df_raw = load_data()
    df = df_raw.copy()

    df_clean = clean_data(df)

    model_rf = train_random_forest(df_clean)

    display_feature_importance(model_rf)

    joblib.dump(model_rf, 'models/modele_forest.pkl')
    print("Random Forest model saved.")

    print("--- PIPELINE COMPLETE: MODEL SAVED ---")
