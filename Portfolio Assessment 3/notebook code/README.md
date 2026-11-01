# COS40007 Portfolio Assessment 3 – Advanced ML and Deep Learning (Option 3c: LSTM for Classification)

Classification of household appliance energy use (Low / Medium / High) on the **Appliances Energy Prediction** dataset, continuing Portfolio Assessments 1 and 2.

## Contents

```
.
├── portfolio_assessment_3.ipynb   # main notebook (Parts A–C)
├── data/
│   └── energydata_complete.csv    # dataset (19,735 rows x 29 columns)
├── figures/                       # figures saved by the notebook for the PDF report
├── models/                        # saved models: Random Forest / MLP (.pkl), LSTM (.h5), training histories (.pkl)
├── requirements.txt
└── README.md
```

## Workflow

- **Part A:** Random Forest and MLP Classifier on the tabular features (chronological 70/30 split, StandardScaler, accuracy / weighted F1 / 5-fold time-series CV).
- **Part B:** LSTM on sequences of past readings (MinMaxScaler, sequence windows, 2 stacked LSTM layers with dropout, EarlyStopping and ReduceLROnPlateau).
- **Part C:** master results table, best model and reflection.

## How to reproduce

1. Install Python 3.13 and the dependencies: `pip install -r requirements.txt`
2. Open `portfolio_assessment_3.ipynb` from this folder so the relative paths `data/`, `figures/` and `models/` resolve.
3. Run all cells (about 10–12 minutes on a CPU; the LSTM training and the 4-seed robustness check are the slowest steps). Figures are written to `figures/` and models to `models/`.

## Dataset

- **Source:** Candanedo, L. (2017). *Appliances Energy Prediction* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5VC8G>
- **Paper:** Candanedo, L., Feldheim, V., & Deramaix, D. (2017). Data driven prediction models of energy use of appliances in a low-energy house. *Energy and Buildings, 140*, 81–97.
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0)
