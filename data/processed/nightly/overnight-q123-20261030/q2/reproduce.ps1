# AI 辅助：OpenAI Codex (GPT-5), OpenAI, 2026-10-30
# 人工核验点：命令只写本运行 nightly/q2；不会覆盖 q2_runs、q2_figures 或 data/raw。
[CmdletBinding()]
param(
    [string]$Python = 'C:\Users\10857\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe',
    [int]$Permutations = 5000,
    [int]$Seed = 20261030
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Worktree = 'F:\cs\mothercup\worktrees\overnight-q123-20261030'
$ExpectedBranch = 'night/overnight-q123-20261030'
$OutputDir = Join-Path $Worktree 'data\processed\nightly\overnight-q123-20261030\q2'
$AuditScript = Join-Path $OutputDir 'q2_robustness_audit.py'
$env:PYTHONDONTWRITEBYTECODE = '1'

$ObservedBranch = (& git -C $Worktree branch --show-current).Trim()
if ($ObservedBranch -ne $ExpectedBranch) {
    throw "Unexpected branch: $ObservedBranch"
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python not found: $Python"
}

& $Python $AuditScript `
    --repo-root $Worktree `
    --output-dir $OutputDir `
    --permutations $Permutations `
    --seed $Seed
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $Python -c "import json,pandas as pd; from pathlib import Path; d=Path(r'$OutputDir'); s=json.loads((d/'robustness_summary.json').read_text(encoding='utf-8')); assert all(s['required_invariants']['checks'].values()); assert len(pd.read_csv(d/'signed_spearman_main_top3.csv'))==150; assert len(pd.read_csv(d/'published_robust_holdout_review.csv'))==32; print('nightly q2 audit checks: PASS')"
exit $LASTEXITCODE
