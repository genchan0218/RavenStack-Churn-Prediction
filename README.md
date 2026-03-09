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
  eda/
    01_eda.R               # Exploratory data analysis & descriptive stats
  modeling/
    02_baseline_models.R   # Baseline model comparison (logit, RF, XGBoost)
    03_extended_analysis.R # Feature engineering stages + hyperparameter tuning
    04_final_comparison.R  # Final model comparison with corrected XGBoost
```

## Key Features Engineered
- Support ticket recency & volume ratios
- Feature adoption rate per subscription tier
- Account age × usage interaction terms
- Rolling 30/60/90-day activity signals

## Tech Stack
- R (data.table, tidyverse, ranger, xgboost, lightgbm, glmnet, pROC)
