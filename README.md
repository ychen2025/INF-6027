# INF-6027
INF6027 Project: Penalty Shootout Prediction and GLMM Modeling
This repository is part of the coursework for the INF6027 module in MSc Data Science at the University of Sheffield. The goal is to analyze and predict football penalty shootouts using R and statistical/machine learning methods, including Generalized Linear Mixed Models (GLMM), Random Forest, and XGBoost.

Project Structure

INF6027/
├── 01_data_preparation.R           # Data cleaning and preprocessing script
├── 02_glmm_analysis.R              # GLMM modeling script
├── rf.R                            # Random Forest model script
├── xgboost.R                       # XGBoost model script
├── pk_glmm_improved.rds            # Saved GLMM model object
├── results.csv                     # Output or final results
├── *.csv                           # Input or cleaned data files (e.g. shootouts.csv)
├── *.png                           # Visual outputs (e.g. confusion matrix, ROC curves)
├── README.md                       # This file

Model Outputs and Figures
- pk_glmm_improved.rds: fitted GLMM model (load via readRDS())
- glmm_confusion_matrix.png: confusion matrix of GLMM prediction
- glmm_roc_curve.png, glmm_roc_curve_ggplot.png: ROC curve for GLMM
- glmm_fixed_effects_improved.csv: GLMM fixed effect estimates
- confusion_matrix_ranger_XGBoost.png: confusion matrix from XGBoost
- roc_ranger_rf.png: ROC curve for Random Forest
- Other PNGs show various model diagnostics and comparison visuals

How to Run
Required R Packages

install.packages(c("lme4", "ggplot2", "dplyr", "readr", "randomForest", "xgboost", "caret"))

Run Order

1. Run 01_data_preparation.R for cleaning and processing
2. Run 02_glmm_analysis.R to fit the GLMM model
3. Optionally run rf.R and xgboost.R for model comparison
4. View results and figures in the root directory

Data Files (Partial)
File Name

Description

shootouts.csv

Main penalty data

goalscorers.csv

Player scoring info

former_names.csv

Historical team name mapping

results.csv

Final predictions and output
