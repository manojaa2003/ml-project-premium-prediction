# Health Insurance Premium Prediction Project

## Project Overview
This project predicts health insurance premiums for a fictional company "Shield" given a person's attributes.  
During development, extreme errors were found primarily in the young age group (<25). To address this, the solution involved:

1. Segmenting data into **Young** and **Rest**
2. Feature-engineering and multicollinearity checks (VIF)
3. Training **Linear Regression** and **XGBoost** models with hyperparameter tuning
4. Targeted error analysis and extreme-error isolation
5. Incorporating a domain-informed feature **Genetical Risk** for the young cohort, dramatically improving results

---

## Data Preparation & Feature Engineering

- **Preprocessing**: missing-value handling, categorical encoding, scaling selected numeric features
- **Scaler Object**: `scaler_with_cols` saved in `artifacts/` to ensure consistency in the Streamlit app
- **Feature Selection with VIF**:
  - Computed VIF to detect multicollinearity (high VIF > 5–10)
  - Columns with high VIF were removed or combined
  - Reduced redundant information, stabilized Linear Regression coefficients
  - Improved interpretability and model stability

---

## Modeling Strategy

- **Baseline Model**: Linear Regression  
  Evaluated with R², MSE, MAE and residual analysis

- **Boosted Model**: XGBoost (XGBRegressor)  
  - Captures non-linear relationships and interactions  
  - Hyperparameters tuned using `RandomizedSearchCV`  
  - Provided feature importance rankings

- **Model Selection per Segment**: Young vs Rest  
  - Trained and compared models for each segment  
  - Best models saved in `artifacts/`

---

## Error Analysis & Segmentation

- **Residual Analysis**: computed residuals and relative percent error
- **Extreme Error Isolation**: flagged entries where |diff_pct| > 10
- **Segmentation Insight**: most extreme errors were in age <25
- **Feature Addition**: `genetical_risk` added for young cohort

**Result:**  
- Accuracy improved from 73% → 98%  
- Extreme errors dropped from 27% → 2%

---

## Why These Improvements Make Sense

- Young customers lack strong clinical history; risk influenced by inherited/genetic factors  
- `genetical_risk` provided the missing explanatory signal  
- VIF-based pruning stabilized linear models  
- XGBoost captured complex interactions with tuned hyperparameters  

Together, these strategies enabled accurate, interpretable, and segment-aware premium prediction.

---

# 🚀 Live App

Interact with the **Shield Insurance Premium Estimator**:  
👉 [Open App](https://ml-project-premium-prediction-manoj.streamlit.app/)

---

## Notebooks Overview & Recommended Order of Viewing

Here’s the suggested order to explore the notebooks along with what each contains:

1. **[ml_premium_prediction.ipynb](model_training/ml_premium_prediction.ipynb)** – Initial end-to-end model development for premium prediction on the full dataset.  
2. **[data_segmentation.ipynb](model_training/data_segmentation.ipynb)** – Segmentation of data into Young (<25) and Rest cohorts, along with exploratory analysis.  
3. **[ml_premium_prediction_young.ipynb](model_training/ml_premium_prediction_young.ipynb)** – Model training and evaluation specifically for the Young cohort.  
4. **[ml_premium_prediction_rest.ipynb](model_training/ml_premium_prediction_rest.ipynb)** – Model training and evaluation for the Rest of the dataset (age ≥25).  
5. **[ml_premium_prediction_young_with_gr.ipynb](model_training/ml_premium_prediction_young_with_gr.ipynb)** – Young cohort model retrained with the domain-informed **Genetical Risk** feature.  
6. **[ml_premium_prediction_rest_with_gr.ipynb](model_training/ml_premium_prediction_rest_with_gr.ipynb)** – Rest cohort model retrained with the **Genetical Risk** feature for consistency and comparison.  


