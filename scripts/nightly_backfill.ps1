# Nightly: prediction outcome backfill + ops metrics
# Register in Windows Task Scheduler (daily), e.g.:
#   Program: powershell.exe
#   Args:    -File F:\dev\scheduler\scripts\nightly_backfill.ps1
#   Start in: F:\dev\scheduler

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$env:PYTHONUNBUFFERED = "1"

# 기상(ASOS) 전일 자료. 현재 FEATURE_COLS 에 날씨는 없으므로 모델 입력이 아니라
# 데이터 축적 목적이다(추후 피처 편입 검토용). 전일 자료는 12시 이후 안정화되므로
# 이 배치는 정오 이후에 도는 것을 전제로 한다.
Write-Host "[$(Get-Date -Format o)] run_weather (--once)"
py -u scripts/etl/run_weather.py --once

Write-Host "[$(Get-Date -Format o)] backfill_outcomes"
py -u -m recommend_api.backfill_outcomes

Write-Host "[$(Get-Date -Format o)] ops_metrics"
py -u -m recommend_api.ops_metrics

Write-Host "[$(Get-Date -Format o)] nightly_backfill done"
