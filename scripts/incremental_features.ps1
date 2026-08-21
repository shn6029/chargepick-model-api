# Every 10 min: incremental feature build (serving dependency)
#
# recommend_api/service.py 는 ev_charger_features 를 INNER JOIN 하고
# config.VERY_STALE_EXCLUDE_MIN(60분) 이내 신선도를 요구한다.
# 이게 멈추면 시간이 지나면서 추천 후보가 0건이 된다.
#
# Register in Windows Task Scheduler (every 10 minutes), e.g.:
#   Program:  powershell.exe
#   Args:     -File F:\dev\scheduler\scripts\incremental_features.ps1
#   Start in: F:\dev\scheduler
#   Trigger:  Daily, repeat every 10 minutes for a duration of 1 day
#
# === 이 작업을 서버(ev-aws)에서 돌리지 말 것 ===
# DB 가 있는 인스턴스는 vCPU 2개짜리 Lightsail 버스터블이다. 지속 CPU 20% 를
# 넘기면 버스트 크레딧이 소진되고 인스턴스 전체가 스로틀되어 MySQL·sshd 까지
# 응답 불능이 된다(2026-07-31 실제 장애). pandas 연산은 반드시 서버 밖에서.
# 자세한 내용: docs/배포_체크리스트.md §0-A
#
# lookback 6시간 근거: 최장 롤링 윈도우가 avail_ratio_60m(60분)이고
# current_state_duration / status_update_age 가 config 상 360분에서 포화하므로
# 6시간 이력이면 저장 구간의 피처가 전체 재빌드와 일치한다(실측 0.00% 불일치).
# 앵커 horizon 은 기본 1일 — 7일로 늘리면 서버 쿼리가 2.3초에서 14.7초로 뛴다.
# 전체 재빌드는 scripts/weekly_eval.ps1 이 주 1회 수행한다(증분 누락 보정).

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$env:PYTHONUNBUFFERED = "1"

Write-Host "[$(Get-Date -Format o)] build_features (incremental, lookback 6h)"
py -u scripts/etl/build_features.py --lookback-hours 6

Write-Host "[$(Get-Date -Format o)] incremental_features done"
