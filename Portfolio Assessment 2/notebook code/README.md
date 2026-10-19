# COS40007 Portfolio Assessment 2 – Preprocessing, Feature Engineering & Model Development

Classification of household appliance energy use (Low / Medium / High) on the **Appliances Energy Prediction** dataset, continuing the EDA of Portfolio Assessment 1.

## Contents

```
.
├── portfolio_assessment_2.ipynb   # main notebook (Tasks 1–10)
├── data/
│   ├── energydata_complete.csv    # dataset (19,735 rows x 29 columns)
│   └── shuffled_modelling_data.csv.gz   # shuffled modelling data written by the notebook (Studio 3)
├── figures/                       # figures saved by the notebook for the PDF report
├── requirements.txt
└── README.md
```

## Workflow

1. Target classes from the PA1 EDA (equal-frequency bins: ≤ 50, 60–80, ≥ 90 Wh)
2. Per-feature scaling (StandardScaler / Yeo-Johnson) and encoding of time features
3. Ten engineered features (occupancy, kitchen activity, aggregates, rates of change)
4. Data shuffled once and saved to `data/shuffled_modelling_data.csv.gz`; five Decision Trees (Studio 3): original raw / original normalised / + engineered / selected normalised / selected raw
5. SVM with 70/30 split and 5-fold CV → kernel selection → GridSearchCV → SelectKBest → PCA → final table and reflections

## How to reproduce

1. Install Python 3.10 or newer and the dependencies: `pip install -r requirements.txt`
2. Open `portfolio_assessment_2.ipynb` from this folder (`notebook code/`) so the relative path `data/...` resolves.
3. Run all cells (about 13 minutes; the grid search is the slowest step). Figures are written to `figures/`.

## Dataset

- **Source:** Candanedo, L. (2017). *Appliances Energy Prediction* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5VC8G>
- **Paper:** Candanedo, L., Feldheim, V., & Deramaix, D. (2017). Data driven prediction models of energy use of appliances in a low-energy house. *Energy and Buildings, 140*, 81–97.
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0)
