# scripts/

스케줄·배치·분석용 실행 스크립트 모음. 서비스 코드(`recommend_api/`)와 분리된 **운영/실험 도구**입니다.

## 하위 폴더

| 경로 | 내용 |
|------|------|
| [etl/](etl/) | 데이터 수집·피처 빌드·CLI 예측 |
| [analysis/](analysis/) | 일회성 탐색·시각화 분석 |

### etl/ 수집기 (원격 Docker 배포 대상)

| 파일 | 주기 | 역할 |
|------|------|------|
| `run.py` | 5분 | 변경분 상태 수집 (getChargerStatus, ~500대 상한) |
| `run_status_snapshot.py` | 수동 1회 / (선택) 10분 스케줄 | **전량 스냅샷** 1회 실행 후 종료 (getChargerInfo) |
| `prune_status.py` | 일 1회 | 21일 롤링 보존 |
| `run_info.py` | 일 1회 | 충전소 마스터 |
| `run_weather.py` | 일 1회 | ASOS 기상 (nightly_backfill.ps1 경유) |

배포 절차와 주의사항은 [docs/배포_체크리스트.md](../docs/배포_체크리스트.md) 참고.
`run.py` 에는 배포 시 DB를 손상시키는 필드 버그가 있었으므로(수정 완료)
구 이미지 교체 전 반드시 확인하세요.

## 루트에 있는 스크립트

| 파일 | 주기 | 역할 |
|------|------|------|
| `incremental_features.ps1` | **5분** | 증분 피처 빌드 — **서빙 필수 의존** |
| `nightly_backfill.ps1` | 일 1회 | outcome 백필 + ops 지표 |
| `weekly_eval.ps1` | 주 1회 | 전체 피처 재빌드 + rolling eval |
| `nightly_eval.ps1` | (deprecated) | 위 두 스크립트로 위임 |
| `verify_ops_flow.py` | 수동 | 운영 흐름(백필·지표) 점검 |

> `incremental_features.ps1` 이 5분 주기로 돌지 않으면 추천 API가 **0건**을 반환합니다.
> `recommend_api/service.py` 가 최신 상태 행에 `ev_charger_features` 를 INNER JOIN 하면서
> 60분 이내 신선도(`config.VERY_STALE_EXCLUDE_MIN`)를 요구하기 때문입니다.
> 현재 지연 상태는 `GET /health` 의 `feature_lag_min` / `feature_lag_ok` 로 확인합니다.

저장소 루트에서 실행합니다.

```bash
cd f:\dev\scheduler
py scripts/etl/build_features.py --lookback-hours 6   # 증분 (5분 주기용)
py scripts/etl/build_features.py                      # 전체 재빌드
py scripts/verify_ops_flow.py
```
