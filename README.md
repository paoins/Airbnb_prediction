# Airbnb Price Prediction Pipeline

This project implements an end-to-end machine learning pipeline to predict Airbnb listing prices using data from Istanbul. It integrates listing attributes, calendar availability, and guest reviews to build a robust predictive model.

## Overview

The pipeline processes raw Airbnb data through multiple engineering stages and trains an ensemble of models designed to handle the high variance of short-term rental prices.

---

## Project Architecture
The pipeline is structured into four distinct phases to ensure data integrity and prevent leakage:

1. **Data Ingestion & Cleaning**: Standardizing raw inputs and handling outliers.

2. **Feature Engineering**: Extracting signals from temporal, text, and categorical data.

3. **Model Selection & Hyperparameter Tuning**: Using Optuna and GridSearchCV to find optimal model configurations.

4. **Ensemble Inference**: Combining multiple models to improve generalization.

---

## Features

### Calendar & Demand Signals
- Availability rates over 7, 30, 60, and 90-day windows  
- Booking block detection to estimate demand intensity  
- Minimum night constraints to capture pricing strategies  

### Review Activity & Trust
- Reviews per month to normalize activity across listings  
- Recent review counts (last 90 and 180 days)  
- Days since the most recent review to measure listing freshness  

### Advanced Property Metrics
- Semantic amenity grouping (Luxury, Work, Family, Kitchen, Parking, Outdoor, Safety, Entertainment)  
- Weighted luxury score for premium amenities  
- Ratio-based density metrics (e.g., people per bedroom, beds per bedroom)  

### Location & Text Analysis
- Euclidean distance to Taksim Square as a centrality proxy  
- Neighborhood-relative quality benchmarking  
- TF-IDF features extracted from listing titles and descriptions

---

## Modeling Strategy

The target variable (price) is log-transformed to reduce skewness and limit the influence of extreme luxury listings.

| Model | Purpose | Optimization |
| :--- | :--- | :--- |
| **XGBoost** | Lead Model | Optuna (TPE Sampler) |
| **Random Forest** | Stability | GridSearchCV |
| **Gradient Boosting**| Boosting Baseline | Fixed Hyperparameters |
| **Ridge** | Linear Baseline | Standard Regularization |

**Ensemble Weights**  
50% XGBoost · 25% Gradient Boosting · 20% Random Forest · 5% Ridge


---
## Results

- **RMSE (log-scale)**: 0.403  
- **MAE (price-scale)**: ~296  
- **Cross-Validation**: 5-fold K-Fold  

The final ensemble demonstrates strong generalization despite the high variance inherent in short-term rental pricing.

---

## ⚙️ How to Run
### Requirements
- Python 3.9+
- pandas
- numpy
- scikit-learn
- xgboost
- optuna

### Data Setup
Place the following files in the project root directory:
- `train.csv`
- `test.csv`
- `calendar.csv`
- `reviews.csv`

### Execution
```bash
python final_db_f2d.py
