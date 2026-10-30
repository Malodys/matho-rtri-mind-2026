# AI 辅助：OpenAI Codex (GPT-5), OpenAI, 2026-10-30
# 人工核验点：仅调用 nightly Q3 审计脚本；不改写 data/raw 与正式 q3_runs。
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..\..\..')).Path
$python = 'C:\Users\10857\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$env:PYTHONPATH = 'F:\cs\mothercup\matho-rtri-mind-2026\.venv\Lib\site-packages'
& $python (Join-Path $PSScriptRoot 'q3_bounded_validation.py') --seeds 7,42,20261030 --n-gen 128
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python (Join-Path $PSScriptRoot 'q3_holdout_figure.py')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
