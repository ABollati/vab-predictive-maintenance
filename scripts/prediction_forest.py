import pandas as pd
import joblib

FEATURES = ['km', 'etat', 'age_vehicule', 'nb_revisions', 'temperature_moteur']

model_rf = joblib.load('models/modele_forest.pkl')


if __name__ == "__main__":
    print("--- VAB BREAKDOWN PREDICTION — RANDOM FOREST ---")

    km = float(input("Enter current mileage (km): "))
    etat = int(input("Enter engine condition (0=Critical, 1=Fair, 2=Good): "))
    age_vehicule = int(input("Enter vehicle age (years): "))
    nb_revisions = int(input("Enter number of past revisions: "))
    temperature_moteur = int(input("Enter engine temperature (°C): "))

    new_vab = pd.DataFrame([[km, etat, age_vehicule, nb_revisions, temperature_moteur]],
                           columns=FEATURES)

    proba_rf = model_rf.predict_proba(new_vab)[0][1]
    prediction = model_rf.predict(new_vab)[0]

    if prediction == 1:
        print(f"ALERT: High breakdown risk ({proba_rf:.2%}). Maintenance required.")
    else:
        print(f"OK: Vehicle operational. Breakdown probability: {proba_rf:.2%}")
