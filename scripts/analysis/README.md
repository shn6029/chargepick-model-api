# scripts/analysis/

탐색·시각화용 **일회성** 분석 스크립트. 서비스 파이프라인에 포함되지 않습니다.

| 파일 | 내용 |
|------|------|
| `analyze_horizon.py` | ETA horizon별 가용 패턴 |
| `analyze_models.py` | 모델 비교 스케치 |
| `analyze_station.py` | 충전소별 상태 분석 |
| `analyze_three_tables.py` | status/features/info JOIN 점검 |
| `analyze_with_weather.py` | 날씨 결합 탐색 |
| `plot_status_heatmap.py` | 상태 히트맵 → `data/reports/` |
| `build_price_table.py` | 요금 테이블 정리 |
| `check_data_quality.py` | 데이터 품질 점검 |
| `parking_eval.py` | 주차장 결합 — 커버리지·신선도 (매칭 pkl 생성) |
| `parking_signal.py` | 주차장 결합 — 시간대 제거 후 잔차 상관 |
| `parking_utility.py` | 주차장 결합 — UI 표시용 가치 평가 |
| `_check_busi_price_match.py` | 사업자·요금 매칭 |
| `_check_power_class.py` | 출력 구간 분류 |

결과 파일은 보통 `data/reports/`에 둡니다.
