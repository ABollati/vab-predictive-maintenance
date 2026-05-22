# VAB Predictive Maintenance

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

Predictive maintenance project for *Véhicules de l'Avant Blindé* (VAB) armoured vehicles.  
The goal is to classify whether a vehicle is at risk of breakdown based on operational and mechanical features, using supervised machine learning (Logistic Regression and Random Forest).

Data is synthetic and generated for technical demonstration purposes.

## Features

| Variable | Description |
|---|---|
| `km` | Current odometer reading |
| `etat` | Engine condition: `0` = Critical, `1` = Fair, `2` = Good |
| `age_vehicule` | Vehicle age in years |
| `nb_revisions` | Number of past maintenance revisions |
| `temperature_moteur` | Engine temperature in °C |
| `panne` | **Target** — `1` = Breakdown detected, `0` = OK |

## Project Structure

```
maintenance-predictive-vab/
├── data/
│   └── donnees_brutes_vab.csv      # Synthetic dataset (800 rows)
├── figures/
│   ├── confusion_matrices.png       # Side-by-side confusion matrices
│   └── roc_curves.png               # Side-by-side ROC curves
├── models/
│   ├── modele_final.pkl             # Trained Logistic Regression model
│   ├── scaler_final.pkl             # MinMaxScaler (for Logistic Regression)
│   └── modele_forest.pkl            # Trained Random Forest model
├── notebooks/
│   └── 01_eda_vab.ipynb             # Exploratory Data Analysis
├── research/
│   ├── mission_vab.py               # Prototyping: fleet exploration & LogReg
│   └── mission_nettoyage.py         # Prototyping: data cleaning & visualisation
├── scripts/
│   ├── pipeline_logistique.py       # Training pipeline — Logistic Regression
│   ├── pipeline_forest.py           # Training pipeline — Random Forest
│   ├── prediction_logistique.py     # Live prediction — Logistic Regression
│   ├── prediction_forest.py         # Live prediction — Random Forest
│   └── model_comparison.py          # Side-by-side model evaluation
├── README.md
└── requirements.txt
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ABollati/maintenance-predictive-vab.git
   cd maintenance-predictive-vab
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS / Linux
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies:**  
   Developed under Python 3.11. Using this version or above is recommended for model compatibility.
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Exploratory Data Analysis

Open the EDA notebook to understand the dataset before modelling:
```bash
jupyter notebook notebooks/01_eda_vab.ipynb
```

### 2. Model Comparison

Train both models and generate evaluation figures (saved to `figures/`):
```bash
python scripts/model_comparison.py
```

### 3. Train Models Individually

Re-train with updated data:
```bash
python scripts/pipeline_logistique.py
python scripts/pipeline_forest.py
```

### 4. Live Prediction

Run an interactive breakdown prediction for a specific vehicle:
```bash
python scripts/prediction_logistique.py
# or
python scripts/prediction_forest.py
```
The script will prompt for mileage, engine condition, vehicle age, number of revisions, and engine temperature.

## Model Comparison

Results on a 20% hold-out test set (stratified split, `random_state=42`):

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 0.863 | 0.539 | 0.304 | 0.389 |
| Random Forest | 0.869 | 0.583 | 0.304 | 0.400 |

> **Note:** The dataset has a ~14% positive class rate (realistic class imbalance).  
> Accuracy alone is therefore misleading — Precision, Recall, and F1-Score are the relevant metrics here.  
> Both models show room for improvement, particularly on Recall (detecting actual breakdowns).

**Feature importances (Random Forest):**

| Feature | Importance |
|---|---|
| `temperature_moteur` | 28.8% |
| `km` | 25.7% |
| `etat` | 17.6% |
| `age_vehicule` | 15.3% |
| `nb_revisions` | 12.6% |

## Research Scripts

The `research/` folder contains standalone prototyping scripts used to validate individual building blocks before integration into the final pipelines:
- `mission_vab.py` — fleet exploration, encoding, normalisation, logistic regression prototype
- `mission_nettoyage.py` — data cleaning rules and visualisation prototype
