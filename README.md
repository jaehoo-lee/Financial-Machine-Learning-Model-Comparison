# Financial Machine Learning Model Comparison

Customer segment classification (A–E) on credit-card customer data, comparing gradient-boosting models under extreme class imbalance.
Team project for Computational Statistics, Hanyang University ERICA (Fall 2025).

## Problem

Each customer belongs to one of five segments (A–E). In the 20% hold-out split (80,000 rows), the segments are distributed as follows:

| Segment | A | B | C | D | E |
|---|---|---|---|---|---|
| Samples | 32 | 5 | 4,253 | 11,642 | 64,068 |

Accuracy is therefore misleading: a model can score about 0.90 while never predicting A or B.
Models were compared on **Macro F1**, with recall on A (VIP) and B (potential VIP) as the business-critical check.

## Data

Eight tables provided for the course: 회원정보 (member) · 신용정보 (credit) · 승인매출정보 (approved sales) · 청구입금정보 (billing and payment) · 잔액정보 (balance) · 채널정보 (channel) · 마케팅정보 (marketing) · 성과정보 (performance).

Duplicate columns shared across tables (such as `기준년월`) were dropped, and the tables were left-joined on `ID` into a 400,000 × 856 table.
Column definitions are in `데이터명세.xlsx`. The raw data is not included in this repository.

## Approach

**Preprocessing**
- Missing values after the left join were treated as *no activity*, not as errors: date columns became presence flags, amount and count columns were filled with 0, and categorical columns with `'missing'`
- Numeric features had heavy L-shaped distributions: extreme values were clipped and features log-transformed
- Categorical features were label-encoded
- One feature from each highly correlated pair was dropped (threshold 0.9, later tuned to 0.94)
- PCA (245 components, 95% variance) was tested and **rejected**: it lowered Macro F1 from 0.6095 to 0.5621 and segment-A recall from 31.25% to 9.3%, so all features were kept

**Modeling**
- Tree-based models for zero-inflated, imbalanced features: XGBoost, LightGBM, CatBoost
- Train/test split 80:20; 5-fold stratified cross-validation on the training set
- Class weights: `N / (5 × n_k)` for segment *k*, capped at 50 to avoid ignoring the large segments
- Hyperparameters: RandomizedSearchCV for the model comparison, Optuna for the final CatBoost

## Results

**Baseline (no class weights).** All three models reached about 0.89 accuracy, but F1 on segment A was near zero.

| | XGBoost | LightGBM | CatBoost |
|---|---|---|---|
| Accuracy | 0.895 | 0.892 | 0.8961 |
| Macro F1 | 0.4446 | 0.456 | 0.4589 |
| F1 (A) | 0 | 0.0755 | 0.0588 |

**Stage 1: class weights and tuning.** Macro F1 rose, but F1 on A and B stayed at 0.35 or below.

| | XGBoost | LightGBM | CatBoost |
|---|---|---|---|
| Accuracy | 0.9235 | 0.9091 | 0.9176 |
| Macro F1 | 0.6239 | 0.5954 | 0.5100 |
| F1 (A) | 0.3500 | 0.3077 | 0.1538 |
| F1 (B) | 0.3333 | 0.3333 | 0.1538 |

**Stage 2: SMOTE on A and B.** A and B were oversampled to 15,000 samples each, using raw rather than log-transformed features. CatBoost led on Macro F1 (CatBoost 0.5757, XGBoost 0.5511, LightGBM 0.53) and was selected for further work.

**Stage 2–3 experiments on CatBoost**

| Experiment | Macro F1 | Accuracy | Recall (A) | Recall (B) | Decision |
|---|---|---|---|---|---|
| SMOTE + raw features | 0.5757 | 0.9059 | 62.5% | 20% | kept |
| Correlation threshold 0.94 | 0.5798 | 0.8972 | 62.5% | 20% | kept |
| Ensemble (CatBoost + XGBoost) | 0.548 | 0.9124 | 46.88% | 0% | withdrawn |
| Deeper trees, more iterations | 0.5273 | 0.9008 | 46.88% | 0% | withdrawn (overfitting) |
| **Optuna (depth 5, learning rate 0.042)** | **0.5926** | **0.8975** | **65.62%** | **20%** | **final** |

The ensemble had the highest accuracy of any model, but it was withdrawn because it lost the rare segments.

**Why not the Stage 1 XGBoost?** It had the highest Macro F1 (0.6239), but its F1 on A and B stayed at 0.35 or below. Because the goal was to find VIP (A) and potential-VIP (B) customers, recall on those segments was prioritized over overall Macro F1 and accuracy.

**Limitation.** Segment A has 32 test samples and segment B has 5, so one sample moves A recall by about 3 points and B recall by 20 points.

## Insight

Segment-level strategy suggested by the model: retain A (churn prevention, premium service), grow B–D toward higher tiers, and serve E at low cost through automated channels.

## Files

| File | Description |
|---|---|
| `train_merged.ipynb` | Table merging and EDA |
| `Preprocessing_first.ipynb` | Preprocessing and feature analysis |
| `xgboost_baseline.ipynb` | XGBoost baseline |
| `test_cat_j.ipynb` | XGBoost / LightGBM / CatBoost comparison |
| `Cat_boost_best.ipynb` | Final CatBoost model |
| `submission.ipynb` | Prediction file generation |
| `데이터명세.xlsx` | Column definitions |

## My contributions

- TODO

## Environment

Python, pandas, scikit-learn, xgboost, lightgbm, catboost, imbalanced-learn, optuna
