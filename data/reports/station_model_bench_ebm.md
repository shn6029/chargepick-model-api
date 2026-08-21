# 충전소 horizon 모델 벤치마크 (mvp_plus)

- 데이터: `station_horizon_training_v1.parquet`
- 피처 7개: available_count, total_chargers, known_charger_count, observation_coverage, observation_age_minutes, horizon_minutes, minutes_since_last_change
- train 178,869 / valid 68,840 / test 52,289 · train 양성률 92.50%

정렬 기준은 **PR-AUC(사용불가)** 다. 양성률이 92.5%라 accuracy·AUC 는 잘 안 벌어진다.

| 모델 | 설명 | ROC-AUC | PR-AUC(불가) | Brier | ECE | 불가 recall | 학습(초) |
|---|---|---:|---:|---:|---:|---:|---:|
| `hgb` | 현행 정본 (HistGradientBoosting) | 0.9452 | 0.7210 | 0.0315 | 0.0035 | 0.660 | 2 |
| `ebm` | ExplainableBoostingMachine (설명 가능 GAM) | 0.9394 | 0.6964 | 0.0341 | 0.0051 | 0.597 | 170 |
| `base_avail_ratio` | available_count / known_charger_count | 0.7714 | 0.2468 | 0.1399 | 0.1339 | 0.653 | 0 |
| `base_current_state` | 현재 available_count>0 이면 가용 (지속성 규칙) | 0.7711 | 0.2400 | 0.0586 | 0.0267 | 0.000 | 0 |
| `base_always_available` | 항상 사용가능 (다수 클래스) | 0.5000 | 0.0742 | 0.0687 | 0.0008 | 0.000 | 0 |
