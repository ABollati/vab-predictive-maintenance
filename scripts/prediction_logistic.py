import joblib
import pandas as pd

FEATURES = ['km', 'condition', 'vehicle_age', 'num_revisions', 'engine_temperature']

model = joblib.load('models/model_logistic.pkl')
scaler = joblib.load('models/scaler_logistic.pkl')


def predict_breakdown(km, condition, vehicle_age, num_revisions, engine_temperature):
    data = pd.DataFrame([[km, condition, vehicle_age, num_revisions, engine_temperature]],
                        columns=FEATURES)
    data_scaled = pd.DataFrame(scaler.transform(data), columns=FEATURES)
    prediction = model.predict(data_scaled)
    probability = model.predict_proba(data_scaled)
    return prediction[0], probability[0][1]


if __name__ == "__main__":
    print("--- VAB BREAKDOWN PREDICTION — LOGISTIC REGRESSION ---")

    km = float(input("Enter current mileage (km): "))
    condition = int(input("Enter engine condition (0=Critical, 1=Fair, 2=Good): "))
    vehicle_age = int(input("Enter vehicle age (years): "))
    num_revisions = int(input("Enter number of past revisions: "))
    engine_temperature = int(input("Enter engine temperature (°C): "))

    verdict, score = predict_breakdown(km, condition, vehicle_age, num_revisions, engine_temperature)

    if verdict == 1:
        print(f"ALERT: High breakdown risk ({score:.2%}). Maintenance required.")
    else:
        print(f"OK: Vehicle operational. Confidence: {(1 - score):.2%}")
