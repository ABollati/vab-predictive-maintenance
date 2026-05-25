import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.ensemble import RandomForestClassifier
import joblib

from data_utils import FEATURES, TARGET, load_data, clean_data


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

    df_raw = load_data()
    df = df_raw.copy()

    df_clean = clean_data(df)

    model_rf = train_random_forest(df_clean)

    display_feature_importance(model_rf)

    joblib.dump(model_rf, 'models/model_forest.pkl')
    print("Random Forest model saved.")

    print("--- PIPELINE COMPLETE: MODEL SAVED ---")
