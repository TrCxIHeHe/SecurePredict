# Experiment Provenance

Generated 2026-10-08 by `src/data/make_provenance.py`.

**Scope of this document.** Git history for v0.1-v0.9 was squashed into one commit, so
it cannot show the order of tuning, threshold selection and test evaluation. The order is
documented by the numbered reports below, each of which states which split it used.
Milestone commits and tags from v1.0 onward provide real chronological provenance.

## Protocol
| Stage | Report | Data used | Test touched? |
|---|---|---|---|
| Cleaning and split policy | 04_data_cleaning_and_split_policy.txt | full data | no |
| Feature quality | 06_feature_quality_audit.txt | train | no |
| Baselines | 07_baseline_models.txt | fit train, eval validation | no |
| RT-IoT feature ablation | 08_rt_iot_feature_ablation.txt | fit train, eval validation | no |
| XGBoost baselines | 09_xgboost_baselines.txt | fit train, eval validation | no |
| AI4I tuning | 10_ai4i_xgboost_tuning.txt | train only (4-fold CV) | no |
| AI4I threshold study | 11_ai4i_threshold_study.txt | validation | no |
| AI4I final test | 12_ai4i_final_test.txt | fit train+val, eval test once | first use |
| RT-IoT final test | 13_rt_iot2022_final_test.txt | fit train+val, eval test once | first use |
| SHAP | 14_shap_explainability.txt | validation | no |

## Frozen decisions
- AI4I primary threshold: 0.77; early-warning threshold: 0.07
- AI4I fit rows: 8,000; test rows: 2,000; test failures: 68
- RT-IoT2022 threshold: 0.5; fit rows: 98,391; test rows: 24,598; numeric features: 78

## Final model artifacts (SHA-256)
| File | Size (bytes) | SHA-256 |
|---|---|---|
| `models/final/ai4i_xgboost_final.joblib` | 529,795 | `a1f2e08ff0d64508e68ab5d937e055eed3d04146134c4e167b65984c6564c013` |
| `models/final/rt_iot2022_random_forest_final.joblib` | 5,453,138 | `c52bd1a9ca952b113b4e2fe1d9b695b10e3da0514c8c942295d8f29c5ee333d9` |

## Environment used to generate this file
Pickles were originally created with the versions pinned in `requirements.txt`.

| Package | Version |
|---|---|
| python | 3.13.7 |
| numpy | 2.5.3 |
| pandas | 2.3.3 |
| scikit-learn | 1.9.1 |
| xgboost | 3.4.1 |
| shap | 0.52.0 |
| streamlit | 1.65.0 |
| joblib | 1.6.0 |
