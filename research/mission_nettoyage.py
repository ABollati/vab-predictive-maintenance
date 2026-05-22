import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Prototype script: data cleaning and visualization

data = {
    'id': range(1, 11),
    'km': [10000, 12000, np.nan, 5000000, 15000, np.nan, 22000, -500, 18000, 25000],
    'etat': [2, 2, 1, 0, 2, np.nan, 1, 1, 2, 0]
}

df_raw = pd.DataFrame(data)

# --- Cleaning ---

# Replace missing km values with the median
df_raw['km'] = df_raw['km'].fillna(df_raw['km'].median())

# Remove rows with km above 1 000 000 (outliers)
df_raw = df_raw[df_raw['km'] <= 1_000_000]

# Remove rows with negative km
df_raw = df_raw[df_raw['km'] >= 0]

# Replace missing etat values with 2 (Good by default)
df_raw['etat'] = df_raw['etat'].fillna(2)

# --- Visualization ---

# Scatter plot: km vs engine condition
plt.figure(figsize=(10, 6))
plt.scatter(df_raw['km'], df_raw['etat'], color='red', s=100, marker='x')

plt.title("Correlation: Mileage vs Engine Condition")
plt.xlabel("Mileage (km)")
plt.ylabel("Condition (0=Critical, 2=Good)")
plt.yticks([0, 1, 2])
plt.grid(True, linestyle=':', alpha=0.6)

plt.show()
