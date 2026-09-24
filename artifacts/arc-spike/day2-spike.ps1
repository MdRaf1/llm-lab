# ADR-0005 day-2 spike - Intel Arc B580, dynamic method (expert-count reduction)
#
# Runs OLMoE-1B-7B-0924 Q4_K_M three ways and records the ADR-0005 evidence:
#   A arc_topk8 : Arc GPU (-ngl 99), default top-8   -> baseline on Arc
#   B arc_topk4 : Arc GPU (-ngl 99), --override to 4  -> the method (less active compute)
#   C cpu_topk8 : same box CPU (-ngl 0), default top-8 -> backend-fidelity reference
#
# For each: n_expert_used (from llama.cpp itself), how many layers went to the GPU
# (anti-CPU-fallback proof), generation tok/s, and the generated text (divergence).
# Reuses whatever day-1 already downloaded. Runs on PS 5.1 and 7.
#
# Usage:  powershell -ExecutionPolicy Bypass -File .\day2-spike.ps1
# Then share day2-report.txt AND the three case-*.txt files.

$ErrorActionPreference = 'Continue'
$ProgressPreference    = 'SilentlyContinue'
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}

$root = 'C:\arc-spike'
$log  = Join-Path $root 'day2-report.txt'
New-Item -ItemType Directory -Force -Path $root | Out-Null
if (Test-Path $log) { Clear-Content $log -ErrorAction SilentlyContinue }
function Log($m) { $m | Tee-Object -FilePath $log -Append }

$PROMPT = 'The capital of France is'
$NTOK   = 48

Log ('=== ADR-0005 ARC DAY-2 SPIKE | ' + (Get-Date -Format o) + ' ===')
Log ('host=' + $env:COMPUTERNAME + ' ps=' + $PSVersionTable.PSVersion.ToString())
Log ('prompt=' + $PROMPT + ' | n_tokens=' + $NTOK + ' | seed=0')

# ---- locate llama-cli (reuse day-1's download; else fetch b* win-vulkan) ----
$bin = Join-Path $root 'llama'
$cliItem = Get-ChildItem $bin -Recurse -Filter 'llama-cli.exe' -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $cliItem) {
  Log 'llama-cli not found from day-1; fetching...'
  try {
    $rels = Invoke-RestMethod 'https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=15' -Headers @{ 'User-Agent' = 'arc-spike' }
    $asset = $null
    foreach ($r in $rels) { $c = $r.assets | Where-Object { $_.name -match 'win-vulkan-x64\.zip$' } | Select-Object -First 1; if ($c) { $asset = $c; break } }
    $zip = Join-Path $root $asset.name
    if (-not (Test-Path $zip)) { Invoke-WebRequest $asset.browser_download_url -OutFile $zip }
    Expand-Archive $zip -DestinationPath $bin -Force
    $cliItem = Get-ChildItem $bin -Recurse -Filter 'llama-cli.exe' -ErrorAction SilentlyContinue | Select-Object -First 1
  } catch { Log ('CLI_FETCH_FAILED: ' + $_.Exception.Message) }
}
$cli = if ($cliItem) { $cliItem.FullName } else { $null }
Log ('llama_cli=' + $cli)

# ---- get the OLMoE Q4_K_M model (verified URLs; skip if present) ----
$model = Join-Path $root 'olmoe.gguf'
if (-not (Test-Path $model)) {
  $urls = @(
    'https://huggingface.co/allenai/OLMoE-1B-7B-0924-GGUF/resolve/main/olmoe-1b-7b-0924-q4_k_m.gguf',
    'https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct-GGUF/resolve/main/olmoe-1b-7b-0924-instruct-q4_k_m.gguf',
    'https://huggingface.co/bartowski/OLMoE-1B-7B-0924-Instruct-GGUF/resolve/main/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf'
  )
  foreach ($u in $urls) {
    try { Log ('downloading model: ' + $u); Invoke-WebRequest $u -OutFile $model -ErrorAction Stop; Log 'model_ok'; break }
    catch { Log ('  failed: ' + $_.Exception.Message) }
  }
}
$modelPresent = Test-Path $model
Log ('model_present=' + $modelPresent)
if ($modelPresent) { Log ('model_size_mb=' + [math]::Round((Get-Item $model).Length / 1MB, 1)) }

# <<CASES>>

function RunCase($label, $ngl, $kv) {
  Log ''
  Log ('=== CASE ' + $label + ' (ngl=' + $ngl + ') ===')
  if (-not ($cli -and $modelPresent)) { Log 'SKIPPED - missing cli or model'; return }
  # Pass 1 - metadata (verbose, 1 token): proves n_expert_used and GPU offload.
  $metaFile = Join-Path $root ('meta-' + $label + '.txt')
  $ma = @('-m', $model, '-p', $PROMPT, '-n', '1', '-ngl', $ngl, '-st', '--seed', '0', '-v')
  if ($kv) { $ma += @('--override-kv', $kv) }
  try { & $cli @ma *> $metaFile } catch { Log ('META_FAILED: ' + $_.Exception.Message) }
  $ml  = (Get-Content $metaFile -Raw -ErrorAction SilentlyContinue) -split "`n"
  $neu = ($ml | Select-String -Pattern 'n_expert_used\s*=' | Select-Object -First 1)
  $vk  = ($ml | Select-String -Pattern 'assigned to device Vulkan').Count
  $cpuL= ($ml | Select-String -Pattern 'assigned to device CPU').Count
  if ($neu) { Log ('  ' + $neu.Line.Trim()) } else { Log '  n_expert_used: NOT FOUND' }
  Log ('  layers_on_vulkan=' + $vk + '  layers_on_cpu=' + $cpuL)
  # Pass 2 - clean generation (no -v): tok/s and the actual text.
  $genFile = Join-Path $root ('gen-' + $label + '.txt')
  $ga = @('-m', $model, '-p', $PROMPT, '-n', $NTOK, '-ngl', $ngl, '-st', '--seed', '0')
  if ($kv) { $ga += @('--override-kv', $kv) }
  try { & $cli @ga *> $genFile } catch { Log ('GEN_FAILED: ' + $_.Exception.Message) }
  $raw = Get-Content $genFile -Raw -ErrorAction SilentlyContinue
  $gl  = $raw -split "`n"
  $tps = ($gl | Select-String -Pattern 'Generation:' | Select-Object -First 1)
  if ($tps) { Log ('  ' + $tps.Line.Trim()) } else { Log '  tok_per_s: NOT FOUND' }
  $gen = ''
  $mm = [regex]::Match($raw, [regex]::Escape($PROMPT) + '(.*?)\[\s*Prompt:', 'Singleline')
  if ($mm.Success) { $gen = ($mm.Groups[1].Value -replace '\s+', ' ').Trim() }
  Log ('  generated="' + $gen + '"')
}

RunCase 'arc_topk8' '99' $null
RunCase 'arc_topk4' '99' 'olmoe.expert_used_count=int:4'
RunCase 'cpu_topk8' '0'  $null

Log ''
Log '=== SUMMARY (for ADR-0005) ==='
Log 'Criterion 2 (ran on GPU): arc cases must show layers_on_vulkan > 0.'
Log 'Criterion 3 (method applied): arc_topk4 must show n_expert_used = 4.'
Log 'Method effect: compare arc_topk8 vs arc_topk4 Generation tok/s.'
Log 'Backend fidelity: compare arc_topk8 generated text vs cpu_topk8 (divergence, not identity, is expected).'
Log ''
Log ('=== END | report: ' + $log + ' ===')
Log 'Share day2-report.txt (and gen-*.txt / meta-*.txt only if asked).'

