# RavenStack Churn Prediction

Predicting SaaS customer churn for RavenStack using machine learning. Built as part of a data science capstone project at Northwestern University.

## Project Overview
Uses account-level behavioral data (feature usage, support tickets, subscription history) to predict churn probability and segment customers into retention risk tiers.

## Models
- Logistic Regression (baseline)
- Random Forest
- XGBoost
- LightGBM
- Elastic Net

**Best model**: LightGBM (AUC ~0.91 after hyperparameter tuning)

## Structure
```
src/
├── eda/
│   └── 01_eda.R               # Exploratory data analysis & descriptive stats
└── modeling/
    ├── 02_baseline_models.R   # Baseline model comparison (logit, RF, XGBoost)
    ├── 03_extended_analysis.R # Feature engineering stages + hyperparameter tuning
    └── 04_final_comparison.R  # Final model comparison with corrected XGBoost
```

## Data
The scripts expect 5 CSV files (not included in this repo):

| File | Description |
|---|---|
| `ravenstack_accounts.csv` | Account-level metadata |
| `ravenstack_subscriptions.csv` | Subscription plan & billing history |
| `ravenstack_feature_usage.csv` | Per-feature usage events |
| `ravenstack_support_tickets.csv` | Support ticket history |
| `ravenstack_churn_events.csv` | Churn labels |

Update the file path variables at the top of each script to point to your local copies.

## How to Run

**Prerequisites** — install R packages:
```r
install.packages(c(
  "data.table", "dplyr", "tidyr", "lubridate", "ggplot2",
  "stringr", "broom", "ranger", "xgboost", "lightgbm",
  "glmnet", "pROC"
))
```

**Run in order:**
```r
# 1. Exploratory analysis
source("src/eda/01_eda.R")

# 2. Baseline model comparison
source("src/modeling/02_baseline_models.R")

# 3. Feature engineering + hyperparameter tuning
source("src/modeling/03_extended_analysis.R")

# 4. Final model comparison (corrected XGBoost)
source("src/modeling/04_final_comparison.R")
```

Each script outputs results to a `reports/` folder (tables as CSV, plots as PNG).

> **Note**: LightGBM must be installed separately — see the [lightgbm R docs](https://lightgbm.readthedocs.io/en/latest/R/index.html).
> CatBoost is optional — the script will skip it gracefully if not installed.

## Key Features Engineered
- Support ticket recency & volume ratios
- Feature adoption rate per subscription tier
- Account age × usage interaction terms
- Rolling 30/60/90-day activity signals

## Tech Stack
- R (data.table, tidyverse, ranger, xgboost, lightgbm, glmnet, pROC)
