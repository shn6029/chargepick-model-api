# charge_history/

환경부/기후부 **월별 충전 세션 이력** 기반 일별 수요(세션수·kWh) 분석 패키지.  
`recommend_api`의 status 기반 ETA 가용확률 모델과 **별개**입니다.

## 주요 모듈

| 파일 | 역할 |
|------|------|
| `ingest.py` | 월별 xlsx → 세션 parquet |
| `panel.py` | station×day 집계 패널 |
| `weather_asos.py` | ASOS 시간자료 → 일별 날씨 캐시 |
| `features.py` | 캘린더·공휴일·lag·날씨 피처 |
| `compare_eval.py` | naive/ridge 수요 모델 월별 비교 |
| `transfer_eval.py` | 대구↔전국 전이 (B_hybrid) |
| `segment_eval.py` | 세그먼트별 성능 |
| `loro_eval.py` | Leave-One-Region-Out |
| `weather_extreme_eval.py` | 극단 기상 구간 평가 |
| `hgb_demand_eval.py` | HGB 수요 challenger |
| `overpredict_monitor.py` | 신규·희소 과대예측 모니터 |
| `expanding_eval.py` | expanding window 평가 |
| `__main__.py` | `py -m charge_history …` 엔트리 |
| [cache/](cache/) | parquet 캐시 (gitignore) |

```bash
cd f:\dev\scheduler

# 대구 (기본)
py -m charge_history ingest
py -m charge_history panel
py -m charge_history weather         # 기상청 ASOS(대구 143) → 일별 캐시
py -m charge_history features
py -m charge_history compare_eval

# 전국 (data/raw 엑셀, 지역 필터 없음 + 시도별 ASOS 날씨)
py -m charge_history weather --scope nationwide
py -m charge_history features --scope nationwide --force
py -m charge_history compare_eval --scope nationwide

# 전이·세그먼트·LORO·극단기상·HGB challenger
py -m charge_history transfer_eval
py -m charge_history segment_eval
py -m charge_history loro_eval
py -m charge_history weather_extreme_eval
py -m charge_history hgb_demand_eval
```

날씨: 공공데이터포털 `AsosHourlyInfoService` (키: `.env`의 `KMA_SERVICE_KEY`).  
캐시:
- 대구: `charge_history/cache/daegu_asos_{hourly,daily}.parquet` (지점 143)
- 전국: `charge_history/cache/nationwide_asos_{hourly,daily}.parquet` (시도 대표 지점)

원본 xlsx: Downloads(대구 기본) / `data/raw`(전국)  
캐시: `charge_history/cache/{daegu,nationwide}_*.parquet`  
산출:
- 대구/전국 compare: `recommend_api/artifacts/charge_history_compare_eval*.{json,md}`
- 전이: `charge_history_transfer_eval.{json,md}`
- 세그먼트: `charge_history_segment_eval_nationwide.{json,md}`
- LORO: `charge_history_loro_eval.{json,md}`
- 극단기상: `charge_history_weather_extreme_eval.{json,md}`
- HGB 수요: `charge_history_hgb_demand_eval.{json,md}`

## 정본 · 보류

**수요 정본: B_hybrid** (`charge_history_transfer_eval.md` PASS — 대구 new_station/sparse 세그먼트 비교 기준).  
수요 HGB 재실험 · 시도 ASOS 날씨 피처는 **승격 후보에서 제외** (데이터 축적 후 재검토).

## 주간 운영 (B_hybrid 과대예측 모니터)

```bash
py -m charge_history overpredict_monitor   # 신규·희소 과대예측 집계 → artifacts/charge_history_overpredict_monitor_MMDD.md
py -m charge_history segment_eval          # 세그먼트별 성능 재확인
```

- B_hybrid · naive_blend 경로의 **신규·희소** 세그먼트에서 `pred - actual` 양(+) 편향을 시도·속도군별 집계
- prior 보정 후보 목록만 출력 — 코드 적용·HGB 승격 없음

**compare_eval 해석:** `ridge_seasonal`이 여러 윈도우·폴드에서 `naive_blend`를 안정적으로 이기기 전까지 서비스 기준 모델은 Naive 유지.

**전이 해석:** 전체 MAE가 아니라 대구 `new_station`/`sparse`에서 A(대구-only) vs B(하이브리드) vs C(전국)를 본다.

**주의:** 세션 이력 기반 일별 세션수·kWh 예측이며, `recommend_api`의 status 기반 ETA 가용확률과 동일 지표가 아닙니다.
