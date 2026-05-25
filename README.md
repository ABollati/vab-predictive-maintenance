# VAB Predictive Maintenance

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

Predictive maintenance project for *Véhicules de l'Avant Blindé* (VAB) armored vehicles.  
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
├── scripts/
│   ├── data_utils.py                # Shared data loading and cleaning
│   ├── pipeline_logistique.py       # Training pipeline — Logistic Regression
│   ├── pipeline_forest.py           # Training pipeline — Random Forest
│   ├── prediction_logistique.py     # Live prediction — Logistic Regression
│   ├── prediction_forest.py         # Live prediction — Random Forest
│   └── model_comparison.py          # Side-by-side model evaluation (5-fold CV)
├── README.md
└── requirements.txt
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ABollati/vab-predictive-maintenance.git
   cd vab-predictive-maintenance
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

Train each model on the full dataset (no train/test split) to produce the final saved models:
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

Results from 5-fold cross-validation (mean ± std):

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 0.865 ± 0.010 | 0.596 ± 0.138 | 0.217 ± 0.095 | 0.305 ± 0.102 |
| Random Forest | 0.855 ± 0.014 | 0.489 ± 0.113 | 0.217 ± 0.055 | 0.300 ± 0.074 |

> **Note:** The dataset has a ~14% positive class rate (realistic class imbalance).  
> Accuracy alone is therefore misleading — Precision, Recall, and F1-Score are the relevant metrics here.  
> The two models perform similarly; both show room for improvement, particularly on Recall (detecting actual breakdowns).

Feature weights from training on the full dataset (via `pipeline_*.py`):

| Feature | LR Coefficient | RF Importance |
|---|---|---|
| `etat` | -1.99 | 17.6% |
| `temperature_moteur` | +1.40 | 28.8% |
| `age_vehicule` | +1.20 | 15.3% |
| `km` | +0.96 | 25.7% |
| `nb_revisions` | +0.56 | 12.6% |

> LR coefficients are signed (negative = reduces breakdown risk) and scaled (MinMax). RF importances measure mean impurity decrease across trees.

