from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
SCHEDULER_ROOT = ROOT.parent

load_dotenv(SCHEDULER_ROOT / ".env")
load_dotenv(ROOT / ".env")

# 추천 API 인증 키. 백엔드가 별도 서버에서 돌아 8000 포트를 인터넷에 노출해야 하므로
# IP 화이트리스트만으로는 부족하다(백엔드 IP 가 바뀌거나 화이트리스트가 느슨해지면 그대로 뚫린다).
# 비어 있으면 인증을 걸지 않는다 — 로컬 개발·기존 호출자 호환용이다.
# **운영 서버에서는 반드시 .env 에 설정할 것.** /health 는 모니터링용이라 항상 열어 둔다.
RECOMMEND_API_KEY = os.getenv("RECOMMEND_API_KEY", "")

MODEL_PATH = ROOT / "artifacts" / "horizon_hgb.joblib"
METRICS_PATH = ROOT / "artifacts" / "horizon_hgb_metrics.json"
ARTIFACTS_DIR = ROOT / "artifacts"

# 잔여시간 모델: 환경변수 또는 형제 프로젝트 기본 경로
_DEFAULT_REMAINING = (
    SCHEDULER_ROOT.parent / "EVCharger-model-test" / "models" / "daegu_remaining_time_hgb.joblib"
)
REMAINING_MODEL_PATH = Path(
    os.getenv("REMAINING_MODEL_PATH", str(_DEFAULT_REMAINING))
)

# 추천 후보로 쓸 실시간 상태 (그 외는 하드 사용불가 → 제외)
# 2=대기, 3=충전중
USABLE_STATS = {2, 3}

# 기상 미연동 시 기본값 (대구 ASOS 근사)
WEATHER_DEFAULTS = {
    "ta": 20.0,
    "rn": 0.0,
    "hm": 50.0,
    "ws": 1.0,
    "dsnw": 0.0,
    "dc10Tca": 0.0,
}


def db_config() -> dict:
    """DB 접속 설정. 값은 전부 환경변수에서 온다.

    폴백값을 두지 않는다. 예전에는 접속정보가 기본값으로 박혀 있어서 `.env` 없이
    스크립트를 돌리면 **의도치 않게 운영 DB에 조용히 붙었다.** 지금은 비어 있으면
    연결 단계에서 실패한다 — 조용히 잘못된 곳에 붙는 것보다 낫다.

    컨테이너에서는 compose 의 `env_file: .env` 가 전부 채운다(api 서비스만
    DB_USER/DB_PASSWORD 를 최소 권한 계정으로 덮어쓴다).
    """
    return {
        "host": os.getenv("DB_HOST", ""),
        "port": int(os.getenv("DB_PORT", "3306")),
        "user": os.getenv("DB_USER", ""),
        "password": os.getenv("DB_PASSWORD", ""),
        "database": os.getenv("DB_NAME", ""),
        "charset": "utf8mb4",
    }


HORIZONS = [5, 10, 15, 20, 30, 45, 60]
# 5분 수집 주기 여유 — 학습 라벨·운영 백필 tolerance 통일
MATCH_TOLERANCE_MIN = 6

# 라벨링 방식 (2026-08-03 도입)
# ----------------------------
# "asof_nearest": t+h 에서 ±MATCH_TOLERANCE_MIN 안에 물리적 관측 행이 있어야 라벨.
# "locf"        : t+h 이전 마지막 관측 상태를 그대로 라벨로 쓴다.
#
# run.py 의 delta 수집은 **상태가 바뀐 충전기만** 보고한다. 즉 관측이 없다는 건
# 상태가 그대로였다는 뜻이므로, t+h 의 상태는 "t+h 이전 마지막 관측"이다.
# nearest 매칭은 이 사실을 못 쓰고 행이 없으면 표본을 버리는데, 하필 stat=3
# (충전중)은 세션이 끝날 때까지 행이 안 생겨서 h30 매칭률이 9.1%까지 떨어졌다.
# 이건 수집 주기가 아니라 라벨링 방식의 문제였다.
LABEL_METHOD = "locf"

# LOCF 로 상태를 끌고 갈 수 있는 최대 나이(분). 이보다 오래된 관측만 있으면 표본을 버린다.
# 근거: 2026-07-31 전량 스냅샷을 정답지로 delta LOCF 복원 정확도를 실측했다(18,618대).
#
#     상태 나이 30분~8시간   정확도 97.7%   가용률 편차 -0.5%p
#     상태 나이  8~12시간    정확도 91.6%   가용률 편차 -7.3%p
#     상태 나이 12~24시간    정확도 73.2%   가용률 편차 -23.9%p   <-- 붕괴
#
# 8시간까지 평평하다가 그 뒤로 무너진다(delta 가 충전 종료 3->2 이벤트를 놓쳐
# 충전기가 '충전중'에 갇히는 것이 주원인). 그래서 상한을 8시간으로 잡는다.
# run_status_snapshot.py 가 6시간 주기로 전량을 앵커링하므로 정상 운영 시
# 실제 staleness 는 6시간을 넘지 않는다.
LABEL_MAX_STALENESS_MIN = 480

# 도착 시점 정답으로 쓸 미래 상태 (1=통신이상, 9=상태미확인은 학습 제외)
LABELABLE_FUTURE_STATS = frozenset({2, 3, 4, 5})
AVAILABLE_FUTURE_STAT = 2

# long-state / true stale / duration cap
LONG_STATE_DURATION_MIN = 360  # 동일 상태 유지(분)
STALE_UPDATE_AGE_MIN = 360  # stat_upd_dt 기준 갱신 지연(분)
DURATION_CAP_MIN = 360
STATUS_UPDATE_AGE_CAP_MIN = 360
# ETA 신뢰도 구간은 service._confidence_level (≤15 high · 16–20 medium · ≥21 low)
# 이 단독 소유한다. 과거 2단계(45분 경계) 상수는 미사용이라 제거했다.
# 백필: 5분 수집 주기 여유
#
# 2026-08-05 부터 운영 백필의 **기본 라벨 규칙은 LABEL_METHOD(=locf)** 다
# (prediction_log.backfill_outcomes). 이 상수는 두 곳에만 남는다.
#   1) settle margin — target_at 직후는 수집이 아직 안 들어왔을 수 있으므로
#      NOW()-6분 이전 target 만 백필한다(수집 지연을 '상태 유지'로 오해 방지).
#   2) 구 방식 `--method asof_nearest` 의 매칭 tolerance(비교·재현용).
# 구 방식이 기본이던 시절 백필률은 34.8%(430/1,236)였고, 남은 표본도 "그 시각에
# 상태가 움직인 충전기"로 치우쳐 오프라인 지표와 눈금이 달랐다.
OUTCOME_MATCH_TOLERANCE_MIN = 6

# 급속 shadow warn (랭킹·제외 로직 불변, 로그·집계만)
# 정책은 "사용불가 Recall >= 0.70 을 만드는 최소 thr".
#
# rapid_threshold_analysis.json (2026-08-18 모델 20260818T022438Z 로 재산출,
# 급속 홀드아웃 n=1,887,302 · 사용불가율 **8.13%** · 테스트 08-13~08-18):
#     thr 0.44   R 0.604  P 0.810  warn  6.06%
#     thr 0.61   R 0.705  P 0.744  warn  7.71%   <-- 정책 달성점
#     thr 0.82   R 0.802  P 0.613  warn 10.63%
#     (현행 0.58 -> R 0.687  P 0.759  warn 7.36%  = 0.70 미달)
#
# 0.58 -> 0.61. 0.57 -> 0.58 때와 같은 패턴으로, 현행값이 목표 R 을 아슬하게 못 넘긴다.
# 재학습할 때마다 이 곡선을 다시 뽑을 것 — 모델이 바뀌면 확률 분포가 미세하게 이동한다.
#
# **이번엔 임계값보다 기저율이 더 움직였다.** 급속 사용불가율이 10.48% -> 8.13% 다.
# 임계값 곡선은 기저율에 직접 반응하므로, 다음 재산출에서 불가율이 10%대로 돌아오면
# 이 값도 되돌아온다고 보는 게 맞다. 절대값을 외우지 말고 곡선을 다시 뽑을 것.
#
# 재산출 스크립트가 바뀌었다: `scripts/analysis/rapid_thresholds_staged.py`.
# 구 `analyze_rapid_thresholds.py` / `scripts/analysis/eta_thresholds.py` 는 둘 다
# `build_horizon_dataset(df, HORIZONS)` 로 7지평을 한 번에 전개하는데, base 행이
# 2,106,246 (14,498,475 샘플)까지 늘어 15.6GB 머신에서 더 이상 돌지 않는다. 새 경로는
# `train --staged --work-dir DIR` 이 남긴 parquet parts + memmap 을 재사용하고,
# 날짜 8:2 분할·홀드아웃 하이퍼파라미터가 같아 metrics 의 date_holdout 과 같은 눈금이다.
#
# **단일 임계값으로 h30 을 덮으려 하지 말 것.** 같은 실행의 ETA별 결과:
#     ETA 15m  thr 0.26 에서 R 0.70 (P 0.843)
#     ETA 20m  thr 0.42 에서 R 0.70 (P 0.771)
#     ETA 30m  thr 0.70 이 필요하고 그때 P 0.609 로 무너진다
# 전 구간 공통 thr 을 h30 기준으로 올리면 단거리에서 오경고가 폭증한다.
#
# 구값 0.82(2026-07-29)는 무효다. 그 실행은 analyze_rapid_thresholds 가 label_df 없이
# build_horizon_dataset 을 불러 snapshot 앵커가 빠진 라벨이었고, 급속 사용불가율이
# 7.40% 로 실제(10.3~10.5%)보다 낮게 잡혀 같은 R 에 훨씬 높은 thr 이 필요해 보였다.
#
# 공통 단일값의 참고치(2026-08-18 기준 0.61). 서빙은 아래 ETA별 곡선을 쓰므로 이 값을
# 직접 판정에 쓰지 말 것 — 문서·비교용으로만 남긴다(`service.py` 는
# `rapid_shadow_warn_threshold()` 만 호출한다).
RAPID_SHADOW_WARN_THRESHOLD_COMMON = 0.61

# ETA별 shadow warn 임계값 (2026-08-07 도입)
# ---------------------------------------
# 공통 단일값은 **방향이 반대로 틀려 있었다.** 같은 홀드아웃(급속 n=1,239,846,
# 사용불가율 10.54%, 테스트 08-04~07)에서 thr 0.58 을 ETA별로 뜯어보면:
#
#     h5   R 0.937  P 0.919  warn 12.4%      <- 잘 맞히는 구간에서 가장 많이 발화
#     h15  R 0.832  P 0.799  warn 11.9%
#     h30  R 0.629  P 0.653  warn  9.7%
#     h45  R 0.353  P 0.623  warn  5.0%      <- 못 맞히는 구간에서 거의 발화 안 함
#     h60  R 0.302  P 0.646  warn  3.9%
#
# 장거리는 확률이 중앙으로 눌려 0.58 밑으로 내려오는 행 자체가 적다. 그래서 정책을
# "ETA별로 발화 예산(warn <= 12%) 안에서 사용불가 Recall 최대화"로 바꿨다.
# 급속 전체 기준 비교 (eta_thresholds.py, data/reports/_eta_thresholds_20260807.json):
#
#     공통 0.58        R 0.7038  P 0.7789  warn  9.52%
#     ETA별 R>=0.70    R 0.7134  P 0.6093  warn 12.34%   <- 정밀도만 버림. 기각
#     ETA별 warn<=12%  R 0.7751  P 0.6861  warn 11.90%   <- 채택
#
# 특히 h45/h60 이 R 0.35/0.30 -> 0.61/0.57 로 오른다. 대가는 그 구간 정밀도
# 0.45/0.41 이지만 기저 불가율이 8.4~8.8% 라 여전히 5배 리프트이고, 서빙은 이미
# ETA>30 을 "참고 수준"으로 표시한다(_horizon_note).
#
# **요청 ETA 는 연속값이다** — 실제 로그에 1·2·4·12·13·16·21·25·27·31·36·44분이 온다.
# 격자 조회가 아니라 선형 보간 + 양끝 clamp 로 쓸 것(rapid_shadow_warn_threshold).
# 재학습하면 이 곡선도 같이 다시 뽑아야 한다
# (scripts/analysis/rapid_thresholds_staged.py — 구 eta_thresholds.py 는 OOM).
#
# 2026-08-18 재산출: **곡선을 바꾸지 않았다.** 모델 20260818T022438Z, 급속 홀드아웃
# n=1,887,302 · 사용불가율 8.13% · 테스트 08-13~08-18 에서 정책 비교:
#
#     A  공통 0.61              R 0.6871  P 0.7585  warn  7.36%
#     A2 현행 ETA별 곡선(아래)   R 0.7609  P 0.6674  warn  9.27%   <-- 정책 둘 다 충족
#     B  ETA별 R>=0.70          R 0.7086  P 0.5671  warn 10.16%
#     C  ETA별 warn<=12%        R 0.8205  P 0.5682  warn 11.74%
#
# 현행 곡선이 새 모델에서도 R 0.7609(>=0.70) · warn 9.27%(<=12%) 로 정책을 만족한다.
# 08-07 채택 당시(R 0.7751 · warn 11.90%)와 거의 같은 자리다. C 를 다시 적용하면
# 임계값이 통째로 뛰는데(h5 0.27->0.96 · h15 0.59->0.91 · h30 0.71->0.85),
# 이는 성능 변화가 아니라 **기저율 변화**다 — warn 예산 12% 를 기저 불가율로 나눈
# 여유가 12/10.54=1.14배에서 12/8.13=1.48배로 늘어 더 헐겁게 발화해도 예산에 들어온다.
# 근거가 기저율이 유난히 낮은 6일 창 하나뿐이고, 불가율이 10%대로 돌아오면 thr 0.96 인
# h5 는 예산을 훌쩍 넘긴다. C 로 가면 Recall +0.0596 에 정밀도 -0.0992 를 내주는
# 거래인데 그 대가를 6일치로 결정할 근거가 없다.
# 바꾸려면 워크포워드로 확인할 것 — 짧은 창의 효과 크기는 2~5배 낙관된 전례가 있다.
RAPID_SHADOW_WARN_THRESHOLD_BY_ETA: dict[int, float] = {
    5: 0.27,
    10: 0.50,
    15: 0.59,
    20: 0.63,
    30: 0.71,
    45: 0.77,
    60: 0.78,
}


def rapid_shadow_warn_threshold(eta_minutes: float) -> float:
    """ETA(분) -> 급속 shadow warn 임계값. 곡선 사이는 선형 보간, 양끝은 clamp.

    5분 미만 요청(로그상 1·2·4분)은 h5 값을, 60분 초과는 h60 값을 그대로 쓴다.
    곡선이 단조 증가라 보간값도 항상 이웃 두 점 사이에 있다.
    """
    grid = sorted(RAPID_SHADOW_WARN_THRESHOLD_BY_ETA)
    eta = float(eta_minutes)
    if eta <= grid[0]:
        return RAPID_SHADOW_WARN_THRESHOLD_BY_ETA[grid[0]]
    if eta >= grid[-1]:
        return RAPID_SHADOW_WARN_THRESHOLD_BY_ETA[grid[-1]]
    for lo, hi in zip(grid, grid[1:]):
        if lo <= eta <= hi:
            t_lo = RAPID_SHADOW_WARN_THRESHOLD_BY_ETA[lo]
            t_hi = RAPID_SHADOW_WARN_THRESHOLD_BY_ETA[hi]
            ratio = (eta - lo) / (hi - lo)
            return round(t_lo + (t_hi - t_lo) * ratio, 4)
    return RAPID_SHADOW_WARN_THRESHOLD_BY_ETA[grid[-1]]
SHADOW_WARN_ENABLED = True

# 충전중(stat=3): HGB 가용확률 × 잔여시간(ETA 전 비울 점수) blend
#
# 2026-08-12: 0.25/0.75 → 1.0/0.0. 잔여시간 성분을 점수에서 뺀다.
#   (이력: 0.4/0.6 → 0.25/0.75 → 1.0/0.0)
#
# 잔여시간 성분이 HGB 를 망가뜨리고 있었다. 워크포워드 6폴드(수집 시작 2026-07-22
# 부터 08-12, 폴드마다 HGB 재학습, 폴드당 평가 3일, 급속 stat=3 약 5만행/폴드):
#
#   폴드평균 Brier   w=0(HGB단독)   w=0.25   w=0.5   w=0.75(구값)   w=1.0
#     ETA10             0.1093      0.1109   0.1145   0.1200        0.1274
#     ETA20             0.1814      0.1859   0.1997   0.2226        0.2549
#     ETA30             0.2149      0.2275   0.2656   0.3294        0.4188
#
#   18셀(6폴드 x 3ETA) 중 17셀에서 최적 w=0, 나머지 하나도 0.1 이었다.
#   AUC 도 같은 방향(ETA30 폴드평균 0.7059 → 0.6789).
#
# 원인은 모델이 아니라 free_by_eta_score 다. 점추정 잔여를 ±10분 램프로 확률에
# 매핑하는데 ML 예측이 늘 10~25분이라 ETA30 에서 거의 전부 1.0 으로 포화한다.
# 실측 조건부 잔여는 경과 30분에서 중앙 15분 / 평균 71분 — 점추정으로 압축이
# 불가능한 분포다. 그 결과 ETA30 에서 평균예측 0.79 vs 실제 0.47 이 됐다.
# HGB 단독은 같은 구간에서 0.472~0.478 vs 실제 0.470~0.484 로 이미 맞는다.
#
# **성분 자체를 지우지는 않았다.** pred_remaining_min · free_by_eta_score 는
# 응답에 계속 나간다(응답 스키마에 있고 백엔드가 이미 읽는다). 값이 틀린 게
# 아니라 점수에 실린 비중이 틀렸던 것이라, 표시용으로는 그대로 쓸 수 있다.
#
# 다시 올릴 거라면: 조건부 종료확률(충전소 x 경과구간 해저드)로 성분을 바꾼 뒤
# w=0.1~0.3, ETA>=20 한정이 최적이었다. 다만 이득이 Brier −0.0007 로 미미하다.
# 재현: py -m recommend_api.experiment_stat3_walkforward
# 상세: data/reports/세션종료해저드_20260812.md
HGB_BLEND_WEIGHT = 1.0
REMAINING_BLEND_WEIGHT = 0.0

# 3단계 추천점수 (제외 → 100점 기본 → 접근성 계수)
VERY_STALE_EXCLUDE_MIN = 60  # status_update_age_min ≥ 이 값이면 하드 제외
PROXY_SPEED_M_PER_MIN = 500.0  # 직선거리 → ETA 근사 (≈30km/h)
SOC_HARD_EXCLUDE_PCT = 5.0
RADIUS_EXPAND_STEP_KM = 1.0
RADIUS_EXPAND_MAX_EXTRA_KM = 2.0
RADIUS_EXPAND_ABS_CAP_KM = 5.0
SCORE_EXPAND_THRESHOLD = 50.0  # 최고점 미만이면 반경 확장

# 기본점수 배점
SCORE_AVAIL_MAX = 50.0
SCORE_ROUTE_TIME_MAX = 15.0
SCORE_ROUTE_DIST_MAX = 5.0
SCORE_BATTERY_MAX = 10.0  # 일단 남겨둬도 됨
SCORE_CHARGER_COUNT_MAX = 15.0
SCORE_SPEED_MAX = 10.0
SCORE_FRESHNESS_MAX = 5.0

# horizon_hgb.joblib (20260818T022438Z) base_feature_cols 와 동일해야 함.
# FEATURE_COLS(25개)는 그대로다. 20260818 재학습에서 열이 104 -> 105 로 늘었는데
# 늘어난 건 범주 더미 `kind_UNK` 하나뿐이다(ev_charger_info.kind 결측 충전기 4대 —
# 대구시 동부소방서/북부소방서/대구시설공단, 전부 PI 7kW). 사업자·충전기타입 더미는
# 증감이 없다. feature_schema_hash 는 f2519f602e9500f6 -> 09cd9d3ee91bbcf5 로 바뀐다.
# 공휴일·상권 파생은 DB/ETL에만 쌓고, 재학습·승격 전까지 모델 입력에 넣지 않음.
#
# is_operating_at_arrival 은 여기에 넣지 말 것 — 실험 결과 효과 없음.
#   experiment_use_time.py 재실행 (2026-08-03, 677만 샘플, 10일/3일 홀드아웃):
#     AUC -0.0000 / unavail-recall +0.0003 / PR-AUC +0.0001 / 야간 AUC +0.0002
#   ETA 5~60분 전 구간에서 |차이| <= 0.0002. 해가 되지도 득이 되지도 않는 순수 0.
#
#   구 수치(AUC +0.0006 / unavail-recall -0.0063 / PR-AUC -0.0023, 2026-07-31)는
#   무효다. 그 실행은 --max-rows 기본값 400000 으로 사업자 61곳 중 21곳이 빠진
#   표본이었고, label_df 없이 돌려 snapshot 앵커까지 빠져 있었다. 당시 보였던
#   unavail-recall -0.0063 은 피처 효과가 아니라 그 설정의 잡음이었다.
#
#   이유: 라벨 문제를 피처로 고칠 수 없다. 문 닫힌 충전기가 t·t+H 모두 stat=2 면
#   라벨은 y=1 이라, 피처가 정답과 모순되어 모델이 무시하는 게 최적이 된다.
#   도착 시점 운영 여부는 '예측 대상'이 아니라 '이미 아는 사실'이므로
#   USABLE_STATS / VERY_STALE_EXCLUDE_MIN 같은 **하드 필터 계층**에 속한다.
#   (계산 자체는 학습·서빙 프레임에 남겨둠 — 필터·진단용)
#
# 세션이력 요일 prior(occupancy_prior 등)도 아직 넣지 말 것 — 2026-08-03 기각.
#   experiment_session_prior.py (662만 샘플, 10일/3일 홀드아웃, 급속 부분집합):
#     AUC +0.0018 / unavail-recall -0.0055 / PR-AUC -0.0164 / Brier +0.0009
#   AUC 는 올랐지만 사용불가 탐지(PR-AUC·recall)와 캘리브레이션이 함께 나빠진다.
#   랭킹만 미세하게 좋아지고 "이 충전기는 찼을 것"을 더 못 잡는 교환이라 손해다.
#   ETA 별로는 h5 +0.0000 → h60 +0.0020 으로 단조 증가 — 신호 자체는 실재한다.
#   기각 사유는 신호 부재가 아니라 **status 가 13일치뿐**이라는 것이다. prior 는
#   (충전기, 요일) 고정값이라 사실상 충전기 ID 로 작동하는데, 학습 10일/평가 3일
#   에서는 요일 패턴이 아니라 충전기별 기저율을 외우게 된다(Brier 악화가 그 증거).
#   status 가 4주 이상 쌓여 홀드아웃이 충전기 ID 와 요일 패턴을 분리할 수 있게 되면
#   재실험할 것. prior 테이블 자체는 ev_charger_session_prior 에 계속 적재된다.
FEATURE_COLS = [
    "eta_minutes",
    "arrival_hour",
    "arrival_weekday",
    "hour",
    "minute_slot",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "capped_state_duration",
    "is_long_state_duration",
    "is_stale_status",
    "capped_status_update_age_min",
    "changes_30m",
    "avail_ratio_15m",
    "avail_ratio_30m",
    "avail_ratio_60m",
    "time_since_available",
    "time_since_charge_started",
    "time_since_charge_ended",
    "stat",
    "output_kw",
    "is_fast",
    "parking_free_yn",
    "limit_yn_flag",
    "traffic_yn_flag",
]
