# EV Scheduler

대구 충전소 **도착 시 사용가능 확률** 추천 API + 충전이력 수요 분석 저장소.

## 폴더 구조

각 폴더에 `README.md`가 있습니다. 자세한 파일 목록은 해당 README를 보세요.

| 경로 | 역할 |
|------|------|
| [recommend_api/](recommend_api/) | **정본** API · 학습 · 100점 랭킹 · 운영 지표 |
| [recommend_api/artifacts/](recommend_api/artifacts/) | 모델·평가 정본 산출물 |
| [charge_history/](charge_history/) | 세션 이력 수요 예측 · 날씨 피처 |
| [api_contract/](api_contract/) | **폐기됨(2026-08-19).** 스키마 정본은 서버 `GET /openapi.json` |
| [scripts/](scripts/) | 배치·검증 스크립트 |
| [scripts/etl/](scripts/etl/) | 수집·피처 빌드·매핑 ETL |
| [scripts/analysis/](scripts/analysis/) | 탐색용 분석 (one-off) |
| [data/](data/) | 원본·임시 리포트 |
| [docs/](docs/) | 팀 문서·접속 가이드 |
| [archive/](archive/) | 구버전 산출물 |




## 자주 쓰는 명령

공통: 저장소 루트에서 실행합니다.

```bash
cd f:\dev\scheduler
```

---

### 1. `py -m recommend_api.main`

**하는 일:** FastAPI 추천 서버를 띄웁니다. (`POST /api/v1/chargers/recommend`, `GET /health`)

| 구분 | 내용 |
|------|------|
| **쓰는 파일** | `recommend_api/main.py`, `service.py`, `scoring.py`, `access.py`, `soc.py`, `remaining_time.py`, `model_store.py`, `config.py` |
| **입력 데이터** | DB (`ev_charger_status` + `ev_charger_features` + `ev_charger_info`), 학습 모델 `recommend_api/artifacts/horizon_hgb.joblib`, `.env` DB 접속 |
| **결과** | HTTP JSON 응답. 충전소별 `available_prob`(도착 시 사용가능 확률), `recommendation_score`(0~100), `score_breakdown`, `recommendation_label` |
| **표현** | API JSON / Swagger(`http://localhost:8000/docs`). 프론트는 `GET /openapi.json` 스키마 기준 |

---

### 2. `py -m recommend_api.train`

**하는 일:** HistGradientBoosting 가용확률 모델을 **재학습**하고(기본) 홀드아웃·Rolling·Calibration 평가까지 돌린 뒤 정본 산출물을 저장합니다.

| 구분 | 내용 |
|------|------|
| **쓰는 파일** | `recommend_api/train.py` → `model_store.train_and_save`, `holdout_eval.py`, `eval_metrics.py` |
| **입력 데이터** | DB JOIN (`status`+`features`+`info`) → horizon 샘플(ETA 5~60분, ±6분 매칭; `future_stat in {2,3,4,5}`만, `y=(stat==2)`; 1·9 제외) |
| **결과 파일** | `recommend_api/artifacts/horizon_hgb.joblib`(모델+피처 스키마), `horizon_hgb_metrics.json`(지표·버전·임계값) |
| **표현** | 콘솔 요약 + JSON 지표. 해석 문서는 `EVALUATION_horizon_hgb_0727.md`. Calibration PNG는 artifacts에 저장 |

옵션: `--skip-eval`(평가 생략), `--max-rows N`(빠른 테스트).

---

### 3. `py scripts/etl/build_features.py`  
(호환: `py build_features.py`)

**하는 일:** 실시간 status 로그로 **분석/학습용 파생 피처**를 다시 계산해 DB에 UPSERT합니다. (주간 배치)

| 구분 | 내용 |
|------|------|
| **쓰는 파일** | `scripts/etl/build_features.py` |
| **입력 데이터** | DB `ev_charger_status` (상태·시각·충전 시작/종료 시각 등) |
| **결과** | DB 테이블 `ev_charger_features` (`hour`, `avail_ratio_*`, `current_state_duration`, `changes_30m` 등) |
| **표현** | DB 행. 콘솔에 처리 건수 출력. 이후 `train` / 추천 API가 이 테이블을 JOIN해서 사용 |

---

### 4. `py -m charge_history compare_eval`

**하는 일:** 충전이력(세션) 기반 **일별 수요** 모델을 월 단위로 비교 검증합니다. (ETA 가용확률 모델과 **별개**)

| 구분 | 내용 |
|------|------|
| **쓰는 파일** | `charge_history/compare_eval.py` (+ 필요 시 선행: ingest/panel/weather/features) |
| **입력 데이터** | `charge_history/cache/` parquet(세션·패널·날씨·피처). 원본 xlsx는 Downloads 또는 `data/raw/` |
| **비교 모델** | `naive_sw`, `naive_blend`, `ridge_base`, `ridge_seasonal` × expanding / rolling_3 / rolling_6 |
| **결과 파일** | `recommend_api/artifacts/charge_history_compare_eval.json`, `.md` · 요약 `charge_history_weather_eval_summary_MMDD.md` |
| **표현** | Markdown 표(폴드별 MAE, 승패), JSON(상세 수치). 기준 모델은 보통 **naive_blend** |

선행이 비어 있으면:

```bash
py -m charge_history all
# 또는 ingest → panel → weather → features → compare_eval
```

---

### 5. `py -m recommend_api.test_scoring`

**하는 일:** 100점 랭킹 티어·접근성 계수·제외 규칙을 **DB/모델 없이** 단위 검증합니다.

| 구분 | 내용 |
|------|------|
| **쓰는 파일** | `recommend_api/test_scoring.py`, `scoring.py`, `access.py` |
| **입력 데이터** | 없음(하드코딩 케이스) |
| **결과** | 콘솔 `scoring tests OK` (실패 시 AssertionError) |
| **표현** | 터미널 pass/fail. 산출 파일 없음 |

---

## 정본 산출물

| 파일 | 의미 |
|------|------|
| `recommend_api/artifacts/horizon_hgb.joblib` | 서비스 정본 가용확률 모델 |
| `recommend_api/artifacts/horizon_hgb_metrics.json` | 학습·평가 수치·`model_version` |
| `recommend_api/artifacts/EVALUATION_horizon_hgb_0727.md` | 가용확률 평가 해석 문서 |
| `recommend_api/artifacts/charge_history_compare_eval_0727.md` | 수요 모델 비교 표 |

구 `artifacts/availability_model.*` 는 `archive/artifacts_obsolete/` 로 옮겼습니다.