# DEPRECATED aggregator — prefer split scripts:
#   scripts/nightly_backfill.ps1  (daily: backfill_outcomes → ops_metrics)
#   scripts/weekly_eval.ps1       (weekly: build_features → rolling_eval)
#
# This file delegates to both for backward compatibility.
# Prefer registering the two scripts separately in Task Scheduler.

$ErrorActionPreference = "Stop"
$here = $PSScriptRoot

Write-Host "[$(Get-Date -Format o)] nightly_eval.ps1 → weekly_eval + nightly_backfill"
& "$here\weekly_eval.ps1"
& "$here\nightly_backfill.ps1"
Write-Host "[$(Get-Date -Format o)] nightly_eval (delegated) done"
