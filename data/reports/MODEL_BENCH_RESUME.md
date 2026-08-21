# 모델 벤치마크 + h30 개선 — 완료 (2026-08-04 ~ 08-06)

## 요약: 모델이 아니라 피처였다

| | h30 PR-AUC(불가) | h30 recall |
|---|---:|---:|
| v1 (7피처) | 0.4627 | 0.229 |
| **v2 (23피처)** | **0.5535** | **0.379** |
| 모델 교체로 얻은 것 | +0.003 | — |

**v1 → v2 rolling 9 fold 검증: Δ +0.0371 ± 0.0229 · 9승 0패 · 개선 확정.**
정식 편입 완료 — `scripts/etl/build_station_timeseries.py` · `--featureset v2`.

---

# 1부. 모델 비교 (결론: 교체 무의미)

DA① 풀데이터 팩 `station_horizon_training_v1.parquet` (1,616,859행 / 충전소 1,590개)로
**15개 모델 + 규칙 기준선 3개**를 비교했다. 세 단계 전부 완료.

## 결론

**현행 HistGradientBoosting 을 유지한다. 모델을 바꿀 근거가 데이터에 없다.**

1. **고정 분할 비교** — 트리 계열 8개가 PR-AUC(사용불가) 0.70~0.73 안에 전부 뭉쳤다.
2. **rolling fold 대결** (9 fold) — 도전자 4개(LGBM·XGB·CatBoost·RF) **전부 "차이 없음"**.
   HGB 가 평균 1위(0.7433)이고 넷 다 Δ가 음수이며 |Δ평균| < Δ표준편차.
   **fold 간 표준편차 ±0.025 가 모델 간 격차 0.003 보다 8배 크다.**
   날짜를 바꾸는 게 모델을 바꾸는 것보다 8배 큰 영향을 준다.
3. **캘리브레이션** — 트리 계열은 이미 ECE 0.003~0.005 라 isotonic 보정이 고칠 게 없다.
   Brier 는 그대로고 PR-AUC 는 오히려 미세하게 **떨어진다**(isotonic 계단 함수가
   랭킹 해상도를 깎는다). 보정이 유효한 건 logreg 뿐(ECE 0.0172 → 0.0073).

### 진짜 병목은 h30 → 2부에서 해결

### class_weight 계열은 채택 불가

`hgb_balanced` · `xgb_spw` · `lgbm_balanced` 셋 다 같은 패턴:
불가 recall 0.65 → 0.83 을 얻는 대신 **ECE 0.003 → 0.13 (40배 악화)**, Brier 0.031 → 0.069.
확률이 `available_prob` 로 100점 점수에 그대로 들어가는 구조라 쓸 수 없다.
recall 이 필요하면 가중치 말고 **임계값**을 내릴 것 (`holdout_eval.py` 의 임계값 스윕).

### 승패 카운트로 고르면 안 된다

CatBoost 는 9 fold 중 **7승 2패**인데 Δ평균은 −0.0001 이다.
7번은 +0.002씩 찔끔 이기고 fold1 에서 −0.0138 로 크게 져서 순이득이 0.
LGBM 도 fold1 에서 −0.0287 (train 5일뿐일 때 취약). 판정은 승패가 아니라 Δ평균 ± 표준편차로.

---

# 2부. h30 개선 (결론: 시계열 피처 정식 편입)

## 진단 — PR-AUC 절대값으로 horizon 을 비교하면 안 된다

horizon 마다 양성률이 다르다. 무작위 기준선(=음성 비율) 대비 **lift** 로 정규화해야 한다.

| horizon | 양성률 | PR-AUC | lift | 상태변화율 |
|---|---:|---:|---:|---:|
| h5 | 0.918 | 0.799 | 9.8x | 7.5% |
| h10 | 0.919 | 0.799 | 9.8x | 7.5% |
| h15 | 0.931 | 0.605 | 8.7x | 14.5% |
| h30 | 0.943 | 0.463 | **8.1x** | **21.5%** |

붕괴가 아니라 완만한 저하(9.8x → 8.1x)다. 그리고 **h5 와 h10 은 같은 문제**다
(라벨 일치율 99.7% · 82.5% 가 물리적으로 같은 관측). 실질 horizon 은 4개가 아니라 3개다.

핵심: **h5 와 h30 의 라벨은 91.7% 일치한다.** h30 문제 = 나머지 **8.3%** 를 맞히는 문제.

## 원인 — v1 피처에 "변화"가 없었다

v1 의 7피처는 전부 현재 시점 스냅샷이다. "지금 몇 대 비었나"는 있는데
"얼마나 자주 비는 곳인가", "지금 비는 중인가 차는 중인가"가 없다.
h5 는 현재 상태가 유지되니 충분하지만, h30 의 8.3% 는 변화의 방향과 속도를 알아야 맞힌다.

재료는 `station_tick_panel.parquet`(698만 행 · 5분 grid)에 이미 있었는데 안 쓰고 있었다.

## 결과 — 정식 편입 완료

`scripts/etl/build_station_timeseries.py` 가 시계열 피처 8종을 만들어 v2 학습셋을 생성한다.

| horizon | v1 | v2 | Δ | recall v1 → v2 |
|---|---:|---:|---:|---|
| h5 | 0.7989 | 0.8378 | +0.0389 | 0.788 → 0.805 |
| h10 | 0.7992 | 0.8394 | +0.0402 | 0.802 → 0.800 |
| h15 | 0.6051 | 0.6823 | +0.0772 | 0.564 → 0.602 |
| **h30** | **0.4627** | **0.5535** | **+0.0908** | **0.229 → 0.379** |

**rolling 9 fold 검증: Δ +0.0371 ± 0.0229 · 9승 0패 · 판정 "개선 확정".**
단판 이득이 아니라 모든 fold 에서 재현된다.

피처군별 기여 (h30 ablation): 회전율 +0.0920 · 추세 +0.0618 · 경과시간 +0.0358.

## v2 에서 모델 재판정 — 여전히 HGB 유지

| 모델 | PR-AUC(불가) | Δ vs hgb | 승/패 | 판정 |
|---|---:|---:|---:|---|
| catboost | 0.7982 ± 0.0192 | +0.0179 ± 0.0200 | 8/1 | 차이 없음 (경계) |
| **hgb** | 0.7803 ± 0.0249 | — | — | 기준선 |
| lgbm | 0.7576 ± 0.0130 | −0.0227 ± 0.0239 | 1/8 | 차이 없음 |

피처가 좋아지자 모델 간 격차가 커졌다(v1 +0.0001 → v2 +0.0179, 100배).
CatBoost 는 8승 1패에 Δ+0.0179 인데 표준편차가 0.0200 이라 **판정은 아직 "차이 없음"**.
경계선이므로 데이터가 더 쌓이면 재판정할 것. 지금 교체할 근거는 아니다(학습 55초 vs 17초).

## 절대 넣지 말 것 — 충전소별 사전확률

`station_hour_prior` · `station_base_rate` 를 넣으면 **성능이 반토막 난다.**

| | h5 | h30 |
|---|---:|---:|
| baseline | 0.7989 | 0.4627 |
| +priors | 0.6336 | **0.2197** |

충전소 단위 집계라 사실상 `station_id` 를 외우는 고차원 인코딩으로 작동한다.
train 기간 충전소별 가용률을 암기하는데 그 패턴이 주 단위로 바뀌어 test 에서 전부 틀린다.
시간대 prior 가 필요하면 충전소 단위가 아니라 **군집 단위**로 묶을 것.

## 남은 과제 — 서빙 파이프라인

이 시계열 피처는 `available_recon`(복원값) 기반이다. 서빙에서 같은 값을 실시간으로
만들려면 **충전소별 최근 3시간 상태 이력**을 유지해야 한다.
오프라인 이득이 그대로 서빙 이득이 되려면 이 파이프라인이 선행되어야 하고,
이것이 편입 작업의 실제 남은 비용이다.

---

## 결과 파일

| 경로 | 내용 |
|---|---|
| [`station_model_rolling_race.md`](station_model_rolling_race.md) · [`.json`](station_model_rolling_race.json) | **최종 판정** · 9 fold × 5모델, fold별 전체 수치 |
| [`station_model_bench_mvp_plus.md`](station_model_bench_mvp_plus.md) · [`.json`](station_model_bench_mvp_plus.json) | 고정 분할 14모델 + per-horizon + isotonic 보정 전후 |
| [`station_model_bench_ebm.md`](station_model_bench_ebm.md) · [`.json`](station_model_bench_ebm.json) | EBM(30만 표본): PR-AUC 0.696 vs HGB 0.721 · 169초 vs 2초 |
| [`station_model_bench_v2.md`](station_model_bench_v2.md) · [`.json`](station_model_bench_v2.json) | **v2 단판** · per-horizon |
| [`station_model_rolling_race_v2.md`](station_model_rolling_race_v2.md) · [`.json`](station_model_rolling_race_v2.json) | **v2 rolling 9 fold** |
| [`h30_feature_pilot.json`](h30_feature_pilot.json) | 피처군별 ablation |

## 재현

```bash
py scripts/etl/build_station_timeseries.py                     # v2 학습셋 생성 (선행)
py scripts/analysis/benchmark_station_models.py --featureset v2
py scripts/analysis/rolling_model_race.py --featureset v2      # 최종 판정 (~20분)
py scripts/analysis/h30_feature_pilot.py                       # 피처군 ablation
```

## 데이터 함정 (반드시 유지)

피처에 넣으면 안 되는 컬럼 — `benchmark_station_models.LEAKY_COLS` 에 박아 뒀다.

- `label_quality` : `CONFIRMED_POSITIVE` ↔ y=1 **1:1 완전 일치**. 넣으면 AUC 1.0.
  `label_reason` · `label_source` · `label_observed_at` · `label_match_delta_minutes` 도 같은 부류.
- `target_known_chargers` · `target_total_chargers` · `target_observation_coverage`
  : 도착 시점(t+h) 관측값이라 예측 시점엔 알 수 없다.

분할은 파일의 `split` 컬럼을 그대로 쓸 것 (train 7/22–7/29 · valid 7/30–8/01 · test 8/02–8/04).
랜덤 분할하면 같은 충전소의 인접 tick 이 train/test 로 갈려 점수가 부풀려진다.

양성률 92.5% 라 accuracy·ROC-AUC 는 안 벌어진다.
판단은 **PR-AUC(사용불가) · 사용불가 recall · Brier · ECE**.

## 환경 함정 (고쳐 둠)

1. **sklearn 1.9 에서 `CalibratedClassifierCV(cv="prefit")` 제거됨**
   → `CalibratedClassifierCV(FrozenEstimator(model))`.
2. **joblib 이 임시폴더 경로를 ascii 로 인코딩** → 윈도우 temp 가 `C:\Users\상현\...` 라
   한글에서 `UnicodeEncodeError`. EBM·RF 등이 죽는다. 두 스크립트가
   `JOBLIB_TEMP_FOLDER` 를 `.joblib_tmp/` (ascii) 로 돌려놓는다.
3. 콘솔 한글 깨짐 → `PYTHONIOENCODING=utf-8`.
4. 추가 설치: `lightgbm` · `catboost` · `interpret`.
