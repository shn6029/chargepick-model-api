# Weekly: FULL feature rebuild + rolling eval
# Register in Windows Task Scheduler (weekly), e.g.:
#   Program: powershell.exe
#   Args:    -File F:\dev\scheduler\scripts\weekly_eval.ps1
#   Start in: F:\dev\scheduler
#
# 역할 분담: 여기서는 --lookback-hours 없이 전체 재빌드만 한다(증분 누락 보정용).
# 서빙이 의존하는 최신 피처는 scripts/incremental_features.ps1 이 5분 주기로 채운다.
# 이 스크립트만으로는 서빙 신선도 요건(60분)을 만족시킬 수 없다.

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$env:PYTHONUNBUFFERED = "1"

Write-Host "[$(Get-Date -Format o)] build_features"
py -u scripts/etl/build_features.py

Write-Host "[$(Get-Date -Format o)] rolling_eval"
py -u -m recommend_api.rolling_eval

Write-Host "[$(Get-Date -Format o)] weekly_eval done"
