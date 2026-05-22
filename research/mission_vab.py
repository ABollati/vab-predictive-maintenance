import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

# Prototype script: 5-vehicle fleet exploration
data = {
    'id_vehicule': [101, 102, 103, 104, 105],
    'km_compteur': [15000, 45000, 12000, 60000, 32000],
    'last_revision_days': [30, 180, 10, 400, 90],
    'engine_condition': ['Good', 'Fair', 'Good', 'Critical', 'Fair'],
    'on_mission': [1, 0, 1, 0, 1]  # 1 = Yes, 0 = No
}

df = pd.DataFrame(data)

# Isolate vehicles in critical condition
critical_vehicles = df[df['engine_condition'] == 'Critical']

# Encode condition labels to numeric values
condition_map = {'Critical': 0, 'Fair': 1, 'Good': 2}
df['etat_num'] = df['engine_condition'].map(condition_map)

# Feature scaling (MinMax normalization)
scaler = MinMaxScaler()
df[['km_compteur', 'etat_num']] = scaler.fit_transform(df[['km_compteur', 'etat_num']])

# Logistic regression on synthetic training labels
X = df[['km_compteur', 'etat_num']]
y = [0, 0, 0, 1, 0]  # Fictional breakdown labels for prototyping

model = LogisticRegression()
model.fit(X, y)

# Test: predict for a high-mileage critical vehicle
new_vab = pd.DataFrame([[400000, 0]], columns=['km_compteur', 'etat_num'])
new_vab_scaled = pd.DataFrame(scaler.transform(new_vab), columns=new_vab.columns)
prediction = model.predict(new_vab_scaled)
probability = model.predict_proba(new_vab_scaled)

print(f"\nBreakdown probability for test vehicle: {probability[0][1]:.2%}")

coefs = model.coef_[0]
intercept = model.intercept_[0]
print(f"Intercept: {intercept:.2f}")
print(f"Coefficient km_compteur: {coefs[0]:.2f}")
print(f"Coefficient etat_num: {coefs[1]:.2f}")
