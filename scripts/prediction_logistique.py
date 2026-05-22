import joblib
import pandas as pd

FEATURES = ['km', 'etat', 'age_vehicule', 'nb_revisions', 'temperature_moteur']

model = joblib.load('models/modele_final.pkl')
scaler = joblib.load('models/scaler_final.pkl')


def predict_breakdown(km, etat, age_vehicule, nb_revisions, temperature_moteur):
    data = pd.DataFrame([[km, etat, age_vehicule, nb_revisions, temperature_moteur]],
                        columns=FEATURES)
    # Scale with the same scaler used during training
    data_scaled = pd.DataFrame(scaler.transform(data), columns=FEATURES)
    prediction = model.predict(data_scaled)
    probability = model.predict_proba(data_scaled)
    return prediction[0], probability[0][1]


if __name__ == "__main__":
    print("--- VAB BREAKDOWN PREDICTION — LOGISTIC REGRESSION ---")

    km = float(input("Enter current mileage (km): "))
    etat = int(input("Enter engine condition (0=Critical, 1=Fair, 2=Good): "))
    age_vehicule = int(input("Enter vehicle age (years): "))
    nb_revisions = int(input("Enter number of past revisions: "))
    temperature_moteur = int(input("Enter engine temperature (°C): "))

    verdict, score = predict_breakdown(km, etat, age_vehicule, nb_revisions, temperature_moteur)

    if verdict == 1:
        print(f"ALERT: High breakdown risk ({score:.2%}). Maintenance required.")
    else:
        print(f"OK: Vehicle operational. Confidence: {(1 - score):.2%}")
