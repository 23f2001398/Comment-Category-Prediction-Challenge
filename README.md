# Comment Category Prediction Challenge

Solution for the [Comment Category Prediction Challenge](https://www.kaggle.com/competitions/comment-category-prediction-challenge) on Kaggle — a multi-class text classification task that predicts which internal handling category (0–3) a user comment belongs to, using a mix of text, numerical, categorical, and datetime features.

## Problem Overview

Each comment in the dataset is tagged with a `label` (0, 1, 2, or 3) representing an internal moderation/handling category. The goal is to predict this label for unseen comments using the comment text along with metadata like upvotes/downvotes, emoticon flags, and demographic-related indicator columns.

- **Training data:** 198,000 rows × 15 columns
- **Test data:** 102,000 rows × 14 columns
- **Target:** `label` — 4 classes, heavily imbalanced (Label 0 ≈ 57.7%, Label 3 ≈ 2.8%)
- **Evaluation metric:** Weighted / Macro F1-score

## Approach

### 1. Exploratory Data Analysis
- Identified feature types (text, numerical, categorical, datetime) and studied the target distribution.
- Found `race`, `religion`, and `gender` share an identical ~73.45% missing-value rate, suggesting they're populated by the same internal detection system.
- Numerical features (`if_1`, `if_2`, `upvote`, `downvote`) are heavily right-skewed with extreme outliers.
- `if_2` showed the strongest correlation (0.23) with the target among numerical features — confirming text would be the dominant signal.
- Label 3 comments are shorter on average (~194 characters), making length a useful discriminative feature.

### 2. Data Preprocessing
- Filled missing values in `race`, `religion`, and `gender` with `"unknown"`.
- Converted `disability` from boolean to binary.
- Cleaned comment text by removing URLs and extra whitespace.
- Clipped `upvote`/`downvote` at the 99th percentile and applied log transformation to reduce skew.

### 3. Feature Engineering
- Built two TF-IDF vectorizers on comment text:
  - Word-level (6,000 features)
  - Character-level (1,000 features, captures subword patterns)
- Combined TF-IDF output with structured/numerical features → **7,031 total features per sample**.
- Encoded target labels with `LabelEncoder`.

### 4. Model Training
Trained three classifiers on the combined feature matrix:

| Model | Validation F1 (Baseline) |
|---|---|
| XGBoost | 0.9062 |
| Linear SVM | 0.9009 |
| Logistic Regression | 0.8953 |

`class_weight='balanced'` was used for Linear SVM and Logistic Regression to address class imbalance.

### 5. Hyperparameter Tuning
- Used `RandomizedSearchCV` with 3-fold cross-validation, optimizing weighted F1.
- Best configurations selected through multiple experiments.

| Model | Validation F1 (Tuned) |
|---|---|
| **XGBoost** | **0.9102** |
| Linear SVM | 0.9009 |
| Logistic Regression | 0.8957 |

XGBoost was selected as the best-performing individual model, though all three stayed within ~1.5% of each other — indicating the engineered features generalize well across model types.

### 6. Model Evaluation
- Overall accuracy: **0.91**, Macro F1: **0.80** (a fairer measure given class imbalance).
- Label 0 (majority class) achieved the highest F1 (0.96).
- Label 3 (rarest class, 2.76% of data) was hardest to predict — F1 of 0.57, recall of only 0.45.
- Labels 1 and 2 were frequently confused due to overlapping text patterns.

### 7. Final Submission — Weighted Ensemble
All three tuned models were retrained on the full training set and combined into a weighted ensemble:

- XGBoost — 0.60
- Linear SVM — 0.25
- Logistic Regression — 0.15

Since `LinearSVC` has no `predict_proba`, its decision function outputs were converted to probabilities via softmax before blending. Final predictions were generated for all 102,000 test comments.

## Repository Structure

```
.
├── notebook.ipynb        # Full end-to-end notebook (EDA → preprocessing → modeling → submission)
└── README.md
```

## Tech Stack

- Python, Pandas, NumPy
- Matplotlib, Seaborn (EDA/visualization)
- Scikit-learn (TF-IDF, Logistic Regression, LinearSVC, LabelEncoder, RandomizedSearchCV)
- XGBoost

## Results Summary

| Metric | Score |
|---|---|
| Best single model (XGBoost, tuned) | F1 = 0.9102 |
| Overall accuracy | 0.91 |
| Macro F1 | 0.80 |
| Final method | Weighted ensemble (XGBoost + Linear SVM + Logistic Regression) |