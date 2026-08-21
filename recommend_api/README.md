# recommend_api/

도착 ETA **사용가능 확률** 예측 + FastAPI 추천 서비스 (정본).

**서비스 적용 가능한 1차 검증 모델** (최종 모델 아님).  
기준선: `artifacts/horizon_hgb.joblib`.

랭킹: **후보 제외 → 100점 기본점수 → 접근성 계수** (`scoring.py`).  
모델은 도착 시 사용가능 확률만 예측하고, 추천점수는 백엔드에서 계산합니다.

## 주요 파일

| 파일 | 역할 |
|------|------|
| `main.py` | FastAPI 앱 (`/recommend`, `/health`) |
| `service.py` | 후보 조회·추론·랭킹 오케스트레이션 |
| `model_store.py` | DB JOIN·horizon 라벨·학습·artifact |
| `config.py` | `FEATURE_COLS`, 점수 배점, 임계값 |
| `scoring.py` / `access.py` | 100점 점수 + 접근성·주차 혼잡 배수 |
| `parking.py` | `ev_charger_parking_map` + 실시간 혼잡 조인 |
| `remaining_time.py` | 충전중 잔여시간 blend |
| `soc.py` | 도착 SOC·최소 출력 |
| `train.py` | 학습 엔트리 |
| `holdout_eval.py` / `rolling_eval.py` | 오프라인 평가 |
| `backfill_outcomes.py` / `ops_metrics.py` | 운영 백필·지표 |
| `prediction_log.py` | 추천 로그 |
| [artifacts/](artifacts/) | 모델·평가 정본 산출물 |

본 모델은 API와 CLI의 모델을 단일화하고, 날짜 홀드아웃·Rolling 검증·ETA별 성능·Calibration·Logistic 기준선 비교를 완료한 프로토타입 1차 검증 모델입니다. 실제 서비스 연동은 가능하지만, 장거리 ETA의 성능 저하와 사용불가 탐지 Recall 개선을 위해 **운영 로그 축적 후 장기 성능 검증**이 필요합니다.

라벨: 도착 `stat in {2,3,4,5}`만 (`2→가용`, `3/4/5→불가`; `1`·`9` 제외).  
학습·백필 매칭 tolerance **6분 통일**. 당분간 **피처 추가·자동 재학습 없음**(게이트 충족 시 수동 승격).

## 준비

```bash
cd f:\dev\scheduler
py -m pip install -r recommend_api/requirements.txt
```

`.env`에 DB 접속 정보 (`DB_HOST` 등).

## 학습 (+ 평가)

```bash
py -m recommend_api.train
py -m recommend_api.train --skip-eval
# FEATURE_COLS 불변일 때만 메타 보강
py -m recommend_api.stamp_artifact_meta
# 급속/완속 holdout 분리 평가 (정본 joblib 미갱신)
py -m recommend_api.eval_by_speed
py -m recommend_api.eval_by_speed --coverage-only
# 급속 임계값·충전소 top1 (joblib 미갱신)
py -m recommend_api.analyze_rapid_thresholds
py -m recommend_api.eval_station_top1
# 급속 전용 실험 모델 (experiments/ 만, 서비스 미승격)
py -m recommend_api.train_rapid_experiment
# TMAP 우회계수 피처 효과 (A/B/위약/변형 4-arm, joblib 미갱신)
py -m recommend_api.experiment_detour_eta
# 저메모리 단계 적재 학습 (기본 경로와 같은 모델 · 평가는 date_holdout 까지)
py -m recommend_api.train --staged
py -m recommend_api.train --staged --work-dir D:\tmp\staged   # 중간 산출 보존·재개
# 단계 적재가 기본 경로와 같은 모델을 내는지 검증 (DB 불필요)
py -m recommend_api.test_staged_equivalence
```

`--staged` 는 **RAM 이 부족할 때만** 쓴다. 가용 메모리를 보고 자동 전환하지 않는데,
그러면 다른 프로그램을 켜뒀는지에 따라 산출 모델이 달라져 재현이 안 되기 때문이다.
HGB 의 `_BinMapper` 가 구간 경계를 20만 행 표본으로 정해서 **행 순서가 바뀌면 모델이
달라진다**(실측 최대 |Δp| 0.086). 그래서 단계 경로도 지평 블록 순서를 그대로 재현하고,
중앙값도 표본이 아닌 정확값을 쓴다. 등가는 `test_staged_equivalence` 가 검증한다.

피처: `capped_state_duration` + `is_long_state_duration` + `is_stale_status`(`stat_upd_dt`) + `capped_status_update_age_min`.  
**장시간 동일상태 ≠ 갱신 지연.** `is_invalid_status_update_time`은 품질 플래그(피처 아님).  
현재 수집분에서 장기 stale 0건 → stale **탐지는 구현**, 모델 학습 영향 검증은 표본 축적 후.  
평가 metrics에 `baseline_logistic`(동일 holdout) · `date_holdout.rapid_only` / `slow_only` · `coverage` 포함. 운영 artifact는 통합 HGB만.  
급속 정의: `output_kw >= 50`. 기본 API 추천은 급속만 (`include_slow=false`).  
임계값 분석: `rapid_threshold_analysis.json`. 충전소 top1: `station_top1_metrics.json`.  
`created_at` 인덱스 DDL: `scripts/etl/sql/idx_status_charger_time.sql` (미적용 시 DBA가 수동 실행).

## 정본 · 보류

| 역할 | 정본 | 상태 |
|------|------|------|
| 가용확률 | `horizon_hgb.joblib` (`20260729T020353Z`) | 서비스 적용 |
| 수요 예측 | **B_hybrid** (transfer_eval PASS) | 서비스 기본 |
| 급속 전용 모델 | `experiments/rapid_horizon_hgb.joblib` | **승격 보류** |
| 수요 HGB / 날씨 피처 | — | **후순위 보류** |

shadow warn: 급속 `available_prob < 0.82` → `prediction_log.shadow_warn` 기록 (랭킹 불변).

## 운영 흐름 (관찰 우선 · 승격 수동)

### 지표 확인 순서

1. `backfill_success_rate` / `no_observation` / `label_staleness_min_buckets`
   (백필 라벨은 학습과 같은 LOCF. `labeled_with_current_method` 가 `total_predictions`
   보다 작으면 방식이 섞인 것 → `backfill_outcomes --relabel` 로 맞출 것.
   `overall` 은 오래된 라벨 잡음을 포함하므로 `overall_fresh_label`(나이 60분 이내)과
   같이 볼 것)
2. `top1_station_failure_rate` · `top1_station_failure_rapid` (급속·fresh; matched 축적 후)
3. `shadow_warn_rate` / `shadow_false_warn_rate` / `shadow_miss_rate` (급속 shadow)
4. 사용불가 Recall → ETA 표본 → Calibration/Brier → long/stale 그룹 → Rolling AUC
5. 오프라인: `station_top1_metrics.json` · `rapid_threshold_analysis.json`

### 매일

```bash
py -m recommend_api.backfill_outcomes
py -m recommend_api.ops_metrics
```

### 매주

```bash
py scripts/etl/build_features.py
py -m recommend_api.rolling_eval
py -m recommend_api.data_quality --sample-chargers 500
py -m charge_history overpredict_monitor   # B_hybrid 신규·희소 과대예측
```

### 하지 않음

자동 재학습 · 자동 모델 교체 · 피처 계속 추가 · tolerance 11분 즉시 적용 · 소량 로그로 성능 결론  
급속 전용 모델 서비스 연결 · 수요 HGB 승격 (관찰 후 수동 게이트 통과 시만)

### 재학습·승격

게이트(Rolling AUC > 0.945, 사용불가 Recall > 0.70, Calibration, 장거리 AUC, 스키마, `verify_ops`) 충족 시 **수동**만.

## API

```bash
py -m recommend_api.main
# /health → model_version, feature_schema_hash
# long_state_note / stale_note / is_invalid_status_update_time
```

## CLI

```bash
py scripts/etl/predict_availability.py --eta 15 --lat 35.84217 --lng 128.68043 --top 10
py -m recommend_api.test_scoring
```

## 잔여시간 모델

`REMAINING_MODEL_PATH` → `EVCharger-model-test` joblib (없으면 kW lookup).
