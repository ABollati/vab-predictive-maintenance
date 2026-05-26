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
| `condition` | Engine condition: `0` = Critical, `1` = Fair, `2` = Good |
| `vehicle_age` | Vehicle age in years |
| `num_revisions` | Number of past maintenance revisions |
| `engine_temperature` | Engine temperature in °C |
| `breakdown` | **Target** — `1` = Breakdown detected, `0` = OK |

## Project Structure

```
maintenance-predictive-vab/
├── data/
│   └── raw_data_vab.csv      # Synthetic dataset (800 rows)
├── figures/
│   ├── confusion_matrices.png       # Side-by-side confusion matrices
│   ├── roc_curves.png               # Side-by-side ROC curves
│   └── pr_curves.png                # Side-by-side Precision-Recall curves
├── models/
│   ├── model_logistic.pkl             # Trained Logistic Regression model
│   ├── scaler_logistic.pkl             # MinMaxScaler (for Logistic Regression)
│   └── model_forest.pkl            # Trained Random Forest model
├── notebooks/
│   └── 01_eda_vab.ipynb             # Exploratory Data Analysis
├── scripts/
│   ├── data_utils.py                # Shared data loading and cleaning
│   ├── pipeline_logistic.py       # Training pipeline — Logistic Regression
│   ├── pipeline_forest.py           # Training pipeline — Random Forest
│   ├── prediction_logistic.py     # Live prediction — Logistic Regression
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

### 2. Model Evaluation

Evaluate both models via 5-fold cross-validation. Prints metric table (mean ± std) and saves confusion matrices, ROC curves, and Precision-Recall curves to `figures/`:
```bash
python scripts/model_comparison.py
```

### 3. Train Final Models

Train each model on the full dataset and save to `models/`. Prints feature weights (LR coefficients and RF importances):
```bash
python scripts/pipeline_logistic.py
python scripts/pipeline_forest.py
```

### 4. Live Prediction

Run an interactive breakdown prediction for a specific vehicle:
```bash
python scripts/prediction_logistic.py
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
> The two models perform similarly; both show room for improvement, particularly on Recall (detecting actual breakdowns). Setting `class_weight='balanced'` on the Logistic Regression would penalise missed breakdowns more heavily and is expected to improve Recall at the cost of Precision.

### Precision-Recall Curves

ROC curves can be misleading on imbalanced datasets: the large number of true negatives inflates the AUC, making models appear stronger than they are. Precision-Recall curves focus exclusively on the minority class (breakdowns), making them more informative in this context.

The baseline (dashed line) represents a random classifier at the positive class rate (~14%). A useful model must stay well above this line. The summary metric is **Average Precision (AP)**, the area under the PR curve.

Feature weights from training on the full dataset (via `pipeline_*.py`):

| Feature | LR Coefficient | RF Importance |
|---|---|---|
| `condition` | -1.99 | 17.6% |
| `engine_temperature` | +1.40 | 28.8% |
| `vehicle_age` | +1.20 | 15.3% |
| `km` | +0.96 | 25.7% |
| `num_revisions` | +0.56 | 12.6% |

> LR coefficients are signed (negative = reduces breakdown risk) and scaled (MinMax). RF importances measure how much each feature contributes to mean impurity decrease in the random forest.

## Limitations

- **Collinearity:** Several feature pairs are correlated by construction (`engine_temperature`/`condition`, `km`/`vehicle_age`, `km`/`num_revisions`, `vehicle_age`/`num_revisions`). For Logistic Regression, this makes individual coefficients less reliable to interpret. For Random Forest, importance is diluted across correlated features — each importance value is likely underestimated.
- **Feature reduction:** Dropping either `condition` or `engine_temperature` (the most collinear pair) could reduce redundancy without significant loss of predictive signal.

