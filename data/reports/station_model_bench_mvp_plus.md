# 충전소 horizon 모델 벤치마크 (mvp_plus)

- 데이터: `station_horizon_training_v1.parquet`
- 피처 7개: available_count, total_chargers, known_charger_count, observation_coverage, observation_age_minutes, horizon_minutes, minutes_since_last_change
- train 964,024 / valid 371,020 / test 281,815 · train 양성률 92.45%

정렬 기준은 **PR-AUC(사용불가)** 다. 양성률이 92.5%라 accuracy·AUC 는 잘 안 벌어진다.

| 모델 | 설명 | ROC-AUC | PR-AUC(불가) | Brier | ECE | 불가 recall | 학습(초) |
|---|---|---:|---:|---:|---:|---:|---:|
| `rf` | RandomForest | 0.9459 | 0.7278 | 0.0308 | 0.0052 | 0.649 | 31 |
| `catboost` | CatBoost (기본 캘리브레이션 우수) | 0.9481 | 0.7268 | 0.0308 | 0.0029 | 0.654 | 26 |
| `lgbm` | LightGBM | 0.9470 | 0.7256 | 0.0309 | 0.0036 | 0.653 | 6 |
| `hgb` | 현행 정본 (HistGradientBoosting) | 0.9471 | 0.7234 | 0.0313 | 0.0032 | 0.645 | 8 |
| `xgb` | XGBoost hist | 0.9467 | 0.7231 | 0.0311 | 0.0038 | 0.652 | 11 |
| `logreg` | 로지스틱 회귀 (선형 기준선) | 0.8986 | 0.5065 | 0.0449 | 0.0172 | 0.578 | 1 |
| `base_avail_ratio` | available_count / known_charger_count | 0.7790 | 0.2501 | 0.1389 | 0.1331 | 0.666 | 0 |
| `base_current_state` | 현재 available_count>0 이면 가용 (지속성 규칙) | 0.7766 | 0.2426 | 0.0576 | 0.0279 | 0.000 | 0 |
| `base_always_available` | 항상 사용가능 (다수 클래스) | 0.5000 | 0.0733 | 0.0679 | 0.0022 | 0.000 | 0 |
