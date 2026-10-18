# ✈️ Airline Passenger Satisfaction Prediction

> **Kaggle Playground Series (Season 6, Episode 10)**  
> *Predicting passenger satisfaction using advanced gradient boosting and cross-validation techniques.*

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-Top%2010%25-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)

## 🏆 The Result
* **Public Leaderboard Score (ROC-AUC):** `0.95793`
* **Local Cross-Validation (OOF ROC-AUC):** `0.9585`
* **Rank:** Top ~10% of 4,500+ global competitors.

## 📌 Project Overview
This repository contains my solution for the Kaggle Playground Series - Season 6, Episode 10. The objective was to predict whether an airline passenger would be satisfied based on their flight details and demographic data. 

Because the dataset is heavily unbalanced and contains categorical variables, the primary challenge was building a robust pipeline that prevents overfitting while capturing complex feature interactions.

## 🛠️ Technical Approach

To move beyond a basic baseline and achieve a competitive score, I implemented the following pipeline:

### 1. Data Preprocessing & Engineering
* **Missing Values:** Imputed numerical missing values using the **median** to avoid the influence of outliers.
* **Feature Engineering:** Created a new continuous feature `Total_Delay` (Departure Delay + Arrival Delay). This captures the cumulative frustration of the passenger, which proved highly predictive.
* **Categorical Encoding:** Utilized `LabelEncoder` to convert high-cardinality categorical strings into machine-readable integers.

### 2. The "Pro" Modeling Strategy
Instead of a simple `train_test_split`, I used a rigorous **Stratified 5-Fold Cross-Validation** loop. This ensures:
* Every passenger in the training set is used for validation exactly once.
* The class distribution is preserved across all 5 folds.
* The final predictions on the test set are an average of 5 different models, drastically reducing variance and preventing overfitting.

### 3. Hyperparameter Tuning
I selected `HistGradientBoostingClassifier` (Scikit-Learn's optimized version of LightGBM) for its speed and handling of large datasets. I tuned the following parameters:
* `max_iter=600` (Increased to allow deeper learning)
* `learning_rate=0.03` (Slowed down to prevent overshooting the minimum)
* `l2_regularization=1.0` (Added L2 penalty to penalize complex models)

## 📂 Repository Structure
```text
├── README.md                   <- You are here
├── .gitignore                  <- Ignores large raw CSVs
├── notebook.ipynb              <- The main Jupyter Notebook containing the pipeline
├── submission_tuned.csv        <- The final submission file achieving 0.95793
