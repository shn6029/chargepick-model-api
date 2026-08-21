# scripts/etl/

DB 적재·파생 피처·부가 데이터 수집 ETL.

## 핵심 파이프라인

| 파일 | 역할 | 주기 |
|------|------|------|
| `run.py` | EV 충전기 상태 API → `ev_charger_status` | 5분 |
| `run_info.py` | 충전기 정적 정보 → `ev_charger_info` | 필요 시 |
| `run_weather.py` | 기상청 ASOS → `weather` | 매일 |
| `build_features.py` | status → `ev_charger_features` (공휴일 파생 포함) | 주간 |
| `build_session_priors.py` | 세션이력 → `ev_charger_session_map` · `ev_charger_session_prior` | 월간 (원본 xlsx 갱신 시) |
| `build_station_timeseries.py` | tick panel → 시계열 피처 · 학습셋 v2 | DA① 팩 갱신 시 |

## 부가 데이터 (로드맵)

| 파일 | 역할 | 소스 |
|------|------|------|
| `run_context.py` | 상권 POI → `ev_charger_context` | 소상공인 상권 API |
| `run_parking.py` | 충전소↔주차장 매핑 → `ev_charger_parking_map` | DB `parking_lot_info` |
| `run_traffic.py` | 교통 스냅샷 → `ev_charger_traffic_snapshot` | 대구 교통 API |

## 기타

| 파일 | 역할 |
|------|------|
| `predict_availability.py` | 추천 CLI (API 없이 근처 충전소 점수) |
| `train_availability.py` | 구형 학습 엔트리 (정본은 `recommend_api.train`) |
| `export_daegu_charge_csv.py` | 대구 충전 데이터 CSV export |
| `sql/idx_status_charger_time.sql` | status 테이블 인덱스 DDL |

## 세션이력 prior

`build_session_priors.py` 는 `charge_history` 세션 parquet(2025-01~2026-06)을
DB 충전기에 붙여 **충전기 × 요일** 수요 prior 를 만든다.
status 는 13일치뿐이라 요일 패턴을 만들 수 없지만 세션 이력은 충전기 × 요일 셀당
중앙값 105건이라 개별 패턴을 그대로 쓸 수 있다. 두 데이터는 기간이 겹치지 않아 누수가 없다.

- 매핑: 충전소명 정규화 일치 + unit 집합 겹침 → 229대 (동명 충전소는 환경부 사업자 우선)
- prior: 대구 전체 24,660대 × 7요일 = 172,620행. 개별 이력 229대, 그룹 prior 포함 1,777대
- 나머지(완속)는 세션 이력 자체가 없어 `occupancy_prior` NULL — HGB 가 NaN 을 그대로 처리
- 계층 shrinkage: 충전기 → 충전소 → 지역×속도군, `w = n/(n+30)` (`SHRINK_K`)

`ev_charger_session_prior` 는 `(stat_id, chger_id, WEEKDAY(created_at))` 로 조인한다.

```bash
py scripts/etl/build_features.py
py scripts/etl/build_session_priors.py            # 매핑 + prior
py scripts/etl/build_session_priors.py --dry-run  # DB 쓰기 없이 요약
py scripts/etl/run_parking.py --map
py scripts/etl/run_context.py --dry-run
py scripts/etl/run_traffic.py --once
```

## 충전소 시계열 피처 (v2 학습셋)

`build_station_timeseries.py` 는 DA① 팩의 `station_tick_panel.parquet`(698만 행 · 5분 grid)에서
회전율·추세·경과시간 피처 8종을 만들어 `station_horizon_training_v1` 에 붙인다.

v1 피처는 전부 현재 시점 스냅샷이라 "얼마나 자주 비는 곳인가", "지금 비는 중인가"가 없었다.
h5·h10 은 현재 상태가 유지되므로 문제없지만(h5↔h10 라벨 일치율 99.7%),
h30 은 h5 와 라벨이 8.3% 어긋나고 그 8.3% 를 맞히려면 변화의 방향과 속도가 필요하다.

측정 효과 (test 8/02~8/04 · HGB 동일 조건):

| | h30 PR-AUC(불가) | h30 사용불가 recall |
|---|---:|---:|
| v1 mvp_plus (7피처) | 0.4627 | 0.229 |
| **v2 (23피처)** | **0.5562** | **0.376** |

모델 교체(LGBM·XGB·CatBoost·RF)로 얻은 것은 0.003 이었다. 피처 쪽이 30배 크다.

**충전소별 사전확률은 넣지 말 것.** `station_hour_prior`/`station_base_rate` 를 넣으면
성능이 반토막 난다(h30 0.4627 → 0.2197). 사실상 `station_id` 를 외우는 인코딩이라
train 기간 패턴을 암기하고 test 기간에 틀린 값을 낸다. 근거는 스크립트의
`REJECTED_FEATURES` 주석 참고.

```bash
py scripts/etl/build_station_timeseries.py
py scripts/analysis/benchmark_station_models.py --featureset v2
```
