# data/reports/

분석 스크립트·배치가 남기는 **임시** 로그·중간 리포트.

## `*_20260818.json` — 정본 지표 사본 (추적 대상)

`recommend_api/artifacts/` 는 `.gitignore` 라 저장소에 안 따라온다. 그런데
`docs/portfolio/` 의 성능 수치가 그 아티팩트를 근거로 인용하고 있어서, **읽는 사람이
검증할 수 없는 숫자**가 되는 문제가 있었다. 그래서 지표 JSON만 여기에 사본을 둔다.

모델 `20260818T022438Z` (`build_method: staged_per_horizon` · `input_dtype: float64`)
아티팩트에서 **복사만** 했고 내용은 수정하지 않았다.

| 파일 | 내용 | 인용처 |
|---|---|---|
| `horizon_hgb_metrics_20260818.json` | 학습 메타 · `date_holdout` · `calibration` · `rolling`(8폴드 요약·ETA별) | README 성능 표 |
| `walkforward_20260818.json` | 확장창 8폴드 원자료(폴드별 `n_train`·`overall`·`per_eta`·`rapid_only`) | `CASE_05` 8절 |
| `station_breakdown_20260818.json` | 충전소 세그먼트 분해 | — |
| `rapid_threshold_analysis_20260818.json` | 급속 shadow warn 임계값 곡선 | `config.py` 주석 |

**모델(`*.joblib`)은 계속 제외한다.** 지표만 추적한다.
재학습 시 이 사본을 갱신할지는 선택이다 — 갱신하면 파일명의 날짜도 같이 바꿀 것.

| 예시 파일 | 출처 |
|-----------|------|
| `_status_heatmap.*` | `scripts/analysis/plot_status_heatmap.py` |
| `_nationwide_*.log` | 전국 weather/features/compare 실행 로그 |
| `_transfer_eval*.log` | 전이 평가 로그 |
| `_segment_eval.log` | 세그먼트 평가 로그 |
| `_loro_eval.log` | LORO 평가 로그 |
| `_*_report.txt` | 요금·출력 구간 점검 |

해석이 끝난 정본 문서는 `recommend_api/artifacts/`의 `.md`/`.json`을 보세요.
