import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

from data_utils import FEATURES, TARGET, load_data, clean_data


# 2. FEATURE SCALING

def scale_features(df):
    scaler = MinMaxScaler()
    df[FEATURES] = scaler.fit_transform(df[FEATURES])
    return df, scaler


# 3. MODEL TRAINING

def display_coefficients(model):
    print("Logistic Regression coefficients:")
    coefs = model.coef_[0]
    for name, coef in sorted(zip(FEATURES, coefs), key=lambda x: -abs(x[1])):
        print(f"  {name}: {coef:.2f}")


def train_logistic_regression(df):
    df, scaler = scale_features(df)
    X = df[FEATURES]
    y = df[TARGET]
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model, scaler


# MAIN ORCHESTRATION

if __name__ == "__main__":

    print("--- LOGISTIC REGRESSION PIPELINE STARTED ---")

    df_raw = load_data()
    df = df_raw.copy()

    df_clean = clean_data(df)

    model, scaler = train_logistic_regression(df_clean)

    display_coefficients(model)

    joblib.dump(model, "models/modele_final.pkl")
    joblib.dump(scaler, "models/scaler_final.pkl")

    print("--- PIPELINE COMPLETE: MODEL SAVED ---")
