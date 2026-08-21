# 충전소 horizon 모델 벤치마크 (full)

- 데이터: `station_horizon_training_v1.parquet`
- 피처 15개: available_count, usable_count, known_charger_count, direct_observed_count, total_chargers, observation_coverage, direct_observation_coverage, observation_age_minutes, available_count_delta_1tick, minutes_since_last_change, unobserved_rate, horizon_minutes, hour, weekday, is_weekend
- train 964,024 / valid 371,020 / test 281,815 · train 양성률 92.45%

정렬 기준은 **PR-AUC(사용불가)** 다. 양성률이 92.5%라 accuracy·AUC 는 잘 안 벌어진다.

| 모델 | 설명 | ROC-AUC | PR-AUC(불가) | Brier | ECE | 불가 recall | 학습(초) |
|---|---|---:|---:|---:|---:|---:|---:|
| `hgb` | 현행 정본 (HistGradientBoosting) | 0.9511 | 0.7499 | 0.0301 | 0.0023 | 0.658 | 9 |
| `lgbm` | LightGBM | 0.9500 | 0.7427 | 0.0300 | 0.0045 | 0.658 | 7 |
| `base_avail_ratio` | available_count / known_charger_count | 0.7790 | 0.2501 | 0.1389 | 0.1331 | 0.666 | 0 |
| `base_current_state` | 현재 available_count>0 이면 가용 (지속성 규칙) | 0.7766 | 0.2426 | 0.0576 | 0.0279 | 0.000 | 0 |
| `base_always_available` | 항상 사용가능 (다수 클래스) | 0.5000 | 0.0733 | 0.0679 | 0.0022 | 0.000 | 0 |
