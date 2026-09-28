# Financial Machine Learning Model Comparison

신용카드 고객 데이터를 활용해 고객 **Segment(A~E)** 를 분류하고, 여러 부스팅 모델의 성능을 비교한 프로젝트입니다.

## 데이터
`data/` 폴더에 8개 카테고리의 train/test parquet 파일이 있으며, `ID`·`기준년월` 기준으로 병합해 사용합니다.

회원정보 · 신용정보 · 승인매출정보 · 청구입금정보 · 잔액정보 · 채널정보 · 마케팅정보 · 성과정보

컬럼 설명은 `데이터명세.xlsx` 참고.

## 방법
- **전처리**: 데이터 병합, 결측치 처리, Label/Ordinal 인코딩, 통계 검정(ANOVA·Chi²)과 PCA를 통한 피처 선택
- **모델**: XGBoost, LightGBM, CatBoost (RandomizedSearchCV로 하이퍼파라미터 튜닝)
- **불균형 대응**: 클래스 가중치 부여, SMOTE
- **평가 지표**: Macro F1, Accuracy

## 파일 구성
| 파일 | 설명 |
|---|---|
| `xgboost_baseline.ipynb` | XGBoost 베이스라인 |
| `train_merged.ipynb` | 데이터 병합 및 EDA |
| `Preprocessing_first.ipynb` | 전처리 및 피처 분석 |
| `test_cat_j.ipynb` | XGBoost / LightGBM / CatBoost 비교 실험 |
| `Cat_boost_best.ipynb` | 최적 파라미터 CatBoost 단일 모델 |
| `submission.ipynb` | 제출 파일 생성 |

## 실행 환경
Python, pandas, scikit-learn, xgboost, lightgbm, catboost, imbalanced-learn
