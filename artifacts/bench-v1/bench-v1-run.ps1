# ADR-0006 benchmark v1 - cross-vendor commodity sweep (OLMoE, expert-count 8/4/2)
#
# Run once per machine. Pass -Backend gpu on the RTX 4060 and the Arc B580,
# -Backend cpu on the 5600G box.  For each expert setting it records:
#   quality  = perplexity on the committed corpus (the only quality claim)
#   speed    = decode tok/s, median of 3 runs
#   proof    = n_expert_used (from llama.cpp) + how many layers went to the GPU
# Runs on Windows PowerShell 5.1 and 7. No time pressure - correctness over speed.
#
# Usage:  powershell -ExecutionPolicy Bypass -File .\bench-v1-run.ps1 -Backend gpu
# Then share bench-v1-report.txt.

param([ValidateSet('gpu','cpu')][string]$Backend = 'gpu')

$ErrorActionPreference = 'Continue'
$ProgressPreference    = 'SilentlyContinue'
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}

$root = 'C:\arc-spike'
$log  = Join-Path $root 'bench-v1-report.txt'
New-Item -ItemType Directory -Force -Path $root | Out-Null
if (Test-Path $log) { Clear-Content $log -ErrorAction SilentlyContinue }
function Log($m) { $m | Tee-Object -FilePath $log -Append }

$ngl = if ($Backend -eq 'gpu') { '99' } else { '0' }
$corpusSha = '56cd65efc2341222ef8be930da8b36ae6079a04b699ed47a0702cd134dc0a34e'
$expectedModelBytes = 4000000000

Log ('=== ADR-0006 BENCH v1 | ' + (Get-Date -Format o) + ' ===')
Log ('host=' + $env:COMPUTERNAME + ' ps=' + $PSVersionTable.PSVersion.ToString() + ' backend=' + $Backend + ' (ngl=' + $ngl + ')')

# ---- locate llama-cli + llama-perplexity (reuse spike download; else fetch) ----
$bin = Join-Path $root 'llama'
function Find-Exe($name) { (Get-ChildItem $bin -Recurse -Filter $name -ErrorAction SilentlyContinue | Select-Object -First 1).FullName }
if (-not (Find-Exe 'llama-cli.exe')) {
  try {
    $rels = Invoke-RestMethod 'https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=15' -Headers @{ 'User-Agent' = 'bench-v1' }
    $asset = $null; foreach ($r in $rels) { $c = $r.assets | Where-Object { $_.name -match 'win-vulkan-x64\.zip$' } | Select-Object -First 1; if ($c) { $asset = $c; break } }
    $zip = Join-Path $root $asset.name
    if (-not (Test-Path $zip)) { Invoke-WebRequest $asset.browser_download_url -OutFile $zip }
    Expand-Archive $zip -DestinationPath $bin -Force
  } catch { Log ('CLI_FETCH_FAILED: ' + $_.Exception.Message) }
}
$cli  = Find-Exe 'llama-cli.exe'
$ppl  = Find-Exe 'llama-perplexity.exe'
Log ('llama_cli=' + $cli)
Log ('llama_perplexity=' + $ppl)

# ---- OLMoE model (robust curl.exe resume + size/magic guard, as in the spike) ----
$model = Join-Path $root 'olmoe.gguf'
function Get-Size($p) { if (Test-Path $p) { (Get-Item $p).Length } else { 0 } }
function Is-Gguf($p) { try { $fs=[IO.File]::OpenRead($p); $b=New-Object byte[] 4; [void]$fs.Read($b,0,4); $fs.Close(); return ([Text.Encoding]::ASCII.GetString($b) -eq 'GGUF') } catch { return $false } }
$murls = @(
  'https://huggingface.co/allenai/OLMoE-1B-7B-0924-GGUF/resolve/main/olmoe-1b-7b-0924-q4_k_m.gguf',
  'https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct-GGUF/resolve/main/olmoe-1b-7b-0924-instruct-q4_k_m.gguf'
)
$mi = 0
while ((Get-Size $model) -lt $expectedModelBytes -and $mi -lt $murls.Count) {
  Log ('downloading model (curl.exe, resumable): ' + $murls[$mi])
  & curl.exe -L -C - --retry 5 --retry-delay 3 --fail -o $model $murls[$mi] 2>$null
  if ((Get-Size $model) -lt $expectedModelBytes) { Remove-Item $model -ErrorAction SilentlyContinue; $mi++ }
}
$modelOk = ((Get-Size $model) -ge $expectedModelBytes) -and (Is-Gguf $model)
Log ('model_ok=' + $modelOk + ' size_mb=' + [math]::Round((Get-Size $model)/1MB,1))

# ---- fixed committed corpus (download + verify sha) ----
$corpus = Join-Path $root 'corpus.txt'
if (-not (Test-Path $corpus)) {
  try { Invoke-WebRequest 'https://raw.githubusercontent.com/MdRaf1/llm-lab/master/artifacts/bench-v1/corpus.txt' -OutFile $corpus } catch { Log ('CORPUS_FETCH_FAILED: ' + $_.Exception.Message) }
}
$gotSha = if (Test-Path $corpus) { (Get-FileHash $corpus -Algorithm SHA256).Hash.ToLower() } else { '' }
$corpusOk = ($gotSha -eq $corpusSha)
Log ('corpus_ok=' + $corpusOk + ' sha=' + $gotSha)

# <<SWEEP>>

function RunK($k) {
  Log ''
  Log ('=== expert_used = ' + $k + ' ===')
  if (-not ($cli -and $ppl -and $modelOk -and $corpusOk)) { Log 'SKIPPED - missing prerequisite'; return }
  $kv = if ($k -ne 8) { @('--override-kv', ('olmoe.expert_used_count=int:' + $k)) } else { @() }

  # quality: perplexity (verbose, so the same run proves n_expert_used + GPU placement)
  $pf = Join-Path $root ('ppl-k' + $k + '.txt')
  $pa = @('-m', $model, '-f', $corpus, '-ngl', $ngl, '--seed', '0', '-v') + $kv
  try { & $ppl @pa *> $pf } catch { Log ('PPL_FAILED: ' + $_.Exception.Message) }
  $pl  = (Get-Content $pf -Raw -ErrorAction SilentlyContinue) -split "`n"
  $neu = ($pl | Select-String -Pattern 'n_expert_used\s*=' | Select-Object -First 1)
  $vk  = ($pl | Select-String -Pattern 'assigned to device Vulkan').Count
  $cpu = ($pl | Select-String -Pattern 'assigned to device CPU').Count
  $pplLine = ($pl | Select-String -Pattern 'Final estimate: PPL' | Select-Object -First 1)
  $pplVal = ''
  if ($pplLine) { $mm = [regex]::Match($pplLine.Line, 'PPL\s*=\s*([\d.]+)'); if ($mm.Success) { $pplVal = $mm.Groups[1].Value } }
  if ($neu) { Log ('  ' + $neu.Line.Trim()) } else { Log '  n_expert_used: NOT FOUND' }
  Log ('  layers_on_vulkan=' + $vk + '  layers_on_cpu=' + $cpu)
  Log ('  perplexity=' + $(if ($pplVal) { $pplVal } else { 'NOT FOUND' }))

  # speed: 3 clean generation runs, report each + median
  $rates = @()
  for ($i = 1; $i -le 3; $i++) {
    $gf = Join-Path $root ('gen-k' + $k + '-r' + $i + '.txt')
    $ga = @('-m', $model, '-p', 'The capital of France is', '-n', '64', '-ngl', $ngl, '-st', '--seed', '0') + $kv
    try { & $cli @ga *> $gf } catch {}
    $g = (Get-Content $gf -Raw -ErrorAction SilentlyContinue)
    $gm = [regex]::Match($g, 'Generation:\s*([\d.]+)')
    if ($gm.Success) { $rates += [double]$gm.Groups[1].Value }
  }
  $median = if ($rates.Count) { ($rates | Sort-Object)[[int]([math]::Floor($rates.Count/2))] } else { 'NONE' }
  Log ('  gen_tok_s_runs=' + ($rates -join ', '))
  Log ('  gen_tok_s_median=' + $median)
}

RunK 8
RunK 4
RunK 2

Log ''
Log '=== DONE. Share bench-v1-report.txt. ==='
Log 'Each expert setting: perplexity (quality), median tok/s (speed), and the'
Log 'n_expert_used + layer-placement proof. Backend + host are in the header.'

