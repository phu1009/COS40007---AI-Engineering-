# COS40007 Portfolio Assessment 1 – Exploratory Data Analysis

Exploratory Data Analysis and data preparation of the **Appliances Energy Prediction** dataset (household appliance energy use logged every 10 minutes, with indoor sensor readings and outdoor weather data).

## Repository structure

```
.
├── portfolio_assessment_1_eda.ipynb   # main notebook (Tasks 1–9)
├── data/
│   └── energydata_complete.csv        # dataset (19,735 rows x 29 columns)
├── figures/                           # figures saved by the notebook for the PDF report
├── requirements.txt
└── README.md
```

## How to reproduce

1. Install Python 3.10 or newer, then install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Make sure `data/energydata_complete.csv` is present. If it is missing, download it from the UCI repository (link below) or from the dataset author's repository: <https://github.com/LuisM78/Appliances-energy-prediction-data>.
3. Open the notebook from this folder (`notebook code/`) so the relative path `data/...` resolves:
   
   jupyter notebook portfolio_assessment_1_eda.ipynb
   
4. Run **Kernel → Restart & Run All**. Figures are written to `figures/`.

## Dataset

- **Source:** Candanedo, L. (2017). *Appliances Energy Prediction* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5VC8G>
- **Paper:** Candanedo, L., Feldheim, V., & Deramaix, D. (2017). Data driven prediction models of energy use of appliances in a low-energy house. *Energy and Buildings, 140*.
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0)
