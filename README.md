# Allstate Claims Severity Prediction

This repo contains the code and my results of the Allstate Claims Severity Kaggle competition (https://www.kaggle.com/competitions/allstate-claims-severity/overview). The goal of this competition is to predict the insurance claim lost using given data of anonymous features.

## Dataset
- 188,318 training samples
- 130 features (116 categorical, 14 continuous)
- Target: claim loss amount (highly skewed)

## Approach
1. Data preprocessing: check missing values, encode categoricals into numericals (for lightgbm usage)
2. Exploratory data analysis: examine target distribution and transform it with log function, remove outliers, examine correlations between continuous features
3. Feature engineering: frequence encoding (categorical), statistical features (continuous)
4. Model training: lightgbm, train/validation split for hyperparameter tuning
5. Model evaluation: plot iteration loss, evaluate predictions with MAE
6. Hyperparameter tuning
7. Final model training and prediction

## Results
Scored MAE = 1172.90425 with late submission