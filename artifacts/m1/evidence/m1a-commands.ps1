$ErrorActionPreference = "Stop"

# Rerunnable: remove only this phase's own prior outputs, never a directory wholesale.
# A clean tree needs no --force; deleting our own outputs keeps reruns deterministic.
$outputs = @(
  "artifacts/m1/evidence/m1a-test-output.txt",
  "artifacts/m1/evidence/m1a-run-a.json",
  "artifacts/m1/evidence/m1a-run-b.json",
  "artifacts/m1/evidence/m1a-oracle.json",
  "artifacts/m1/evidence/m1a-trace-summary.json",
  "artifacts/m1/evidence/m1a-files.sha256",
  "artifacts/m1/raw/m1a-trace.jsonl",
  "artifacts/m1/raw/m1a-logits.pt",
  "artifacts/m1/raw/m1a-trace-run.json"
)
foreach ($path in $outputs) {
  Remove-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
}

New-Item -ItemType Directory -Force -Path artifacts/m1/evidence | Out-Null
New-Item -ItemType Directory -Force -Path artifacts/m1/raw | Out-Null

# A passing test intentionally writes one "error:" line to stderr (the malformed-trace
# rejection path). Under Stop + 2>&1, PowerShell 5.1 would promote that benign native
# stderr into a terminating error, so relax to Continue for the capture and gate on the
# real exit code: a genuine test failure still aborts the script.
$ErrorActionPreference = "Continue"
./.venv/Scripts/python.exe tests/test_moe_trace.py 2>&1 |
  Tee-Object artifacts/m1/evidence/m1a-test-output.txt
$testExit = $LASTEXITCODE
$ErrorActionPreference = "Stop"
if ($testExit -ne 0) { throw "test suite failed with exit $testExit" }
./.venv/Scripts/llm-lab.exe moe run --backend tiny --seed 0 --threads 1 `
  --prompt-token-ids 1,3,4 --max-new-tokens 32 `
  --output artifacts/m1/evidence/m1a-run-a.json
./.venv/Scripts/llm-lab.exe moe run --backend tiny --seed 0 --threads 1 `
  --prompt-token-ids 1,3,4 --max-new-tokens 32 `
  --output artifacts/m1/evidence/m1a-run-b.json
./.venv/Scripts/llm-lab.exe moe compare `
  artifacts/m1/evidence/m1a-run-a.json artifacts/m1/evidence/m1a-run-b.json `
  --output artifacts/m1/evidence/m1a-oracle.json
./.venv/Scripts/llm-lab.exe moe trace --backend tiny --seed 0 --threads 1 `
  --prompt-token-ids 1,3,4 --max-new-tokens 32 `
  --trace artifacts/m1/raw/m1a-trace.jsonl `
  --logits artifacts/m1/raw/m1a-logits.pt `
  --run-record artifacts/m1/raw/m1a-trace-run.json `
  --summary artifacts/m1/evidence/m1a-trace-summary.json
# Get-FileHash is absent from this PowerShell install, so compute SHA-256 via .NET.
# Same Algorithm/Hash/Path table shape a Get-FileHash manifest would carry.
$sha256 = [System.Security.Cryptography.SHA256]::Create()
Get-ChildItem artifacts/m1/raw/m1a-trace.jsonl,artifacts/m1/raw/m1a-logits.pt | ForEach-Object {
  $bytes = [System.IO.File]::ReadAllBytes($_.FullName)
  $hash = ([BitConverter]::ToString($sha256.ComputeHash($bytes))).Replace("-", "")
  [PSCustomObject]@{ Algorithm = "SHA256"; Hash = $hash; Path = $_.FullName }
} | Format-Table -AutoSize | Out-File `
  artifacts/m1/evidence/m1a-files.sha256 -Encoding utf8 -Width 200
