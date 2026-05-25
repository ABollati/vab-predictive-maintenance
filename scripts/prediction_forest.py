import pandas as pd
import joblib

FEATURES = ['km', 'condition', 'vehicle_age', 'num_revisions', 'engine_temperature']

model_rf = joblib.load('models/model_forest.pkl')


if __name__ == "__main__":
    print("--- VAB BREAKDOWN PREDICTION — RANDOM FOREST ---")

    km = float(input("Enter current mileage (km): "))
    condition = int(input("Enter engine condition (0=Critical, 1=Fair, 2=Good): "))
    vehicle_age = int(input("Enter vehicle age (years): "))
    num_revisions = int(input("Enter number of past revisions: "))
    engine_temperature = int(input("Enter engine temperature (°C): "))

    new_vab = pd.DataFrame([[km, condition, vehicle_age, num_revisions, engine_temperature]],
                           columns=FEATURES)

    proba_rf = model_rf.predict_proba(new_vab)[0][1]
    prediction = model_rf.predict(new_vab)[0]

    if prediction == 1:
        print(f"ALERT: High breakdown risk ({proba_rf:.2%}). Maintenance required.")
    else:
        print(f"OK: Vehicle operational. Breakdown probability: {proba_rf:.2%}")
