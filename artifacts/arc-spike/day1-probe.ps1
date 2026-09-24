# ADR-0005 day-1 kill-switch probe - Intel Arc B580
#
# Answers ONE question: does any GPU workload run on the B580?
# Downloads into C:\arc-spike, installs nothing, needs no admin, no driver changes.
# Delete that folder to undo everything.
#
# Runs on Windows PowerShell 5.1 and PowerShell 7. Verified by execution on both.
# Usage:  powershell -ExecutionPolicy Bypass -File .\day1-probe.ps1
# Then share C:\arc-spike\day1-report.txt

$ErrorActionPreference = 'Continue'
$ProgressPreference    = 'SilentlyContinue'
# PS 5.1 does not always negotiate TLS 1.2, which GitHub and Hugging Face require.
# No-op on PS 7. Prevents a silent download failure on 5.1.
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}

$root = 'C:\arc-spike'
$log  = Join-Path $root 'day1-report.txt'
New-Item -ItemType Directory -Force -Path $root | Out-Null
if (Test-Path $log) { Clear-Content $log -ErrorAction SilentlyContinue }  # reliable truncate; Remove-Item can be held by a prior handle
function Log($m) { $m | Tee-Object -FilePath $log -Append }

Log ('=== ADR-0005 ARC DAY-1 PROBE | ' + (Get-Date -Format o) + ' ===')
$os = Get-CimInstance Win32_OperatingSystem
$cs = Get-CimInstance Win32_ComputerSystem
Log ('host=' + $env:COMPUTERNAME + ' user=' + $env:USERNAME)
Log ('os=' + $os.Caption + ' build=' + $os.Version)
Log ('cpu=' + (Get-CimInstance Win32_Processor).Name)
Log ('ram_gb=' + [math]::Round($cs.TotalPhysicalMemory / 1GB, 2))
Log ('free_disk_gb=' + [math]::Round((Get-PSDrive C).Free / 1GB, 2))
Log ('ps_version=' + $PSVersionTable.PSVersion.ToString())

# ---------- 1. GPU present and driver healthy ----------
Log ''
Log '--- [1] GPUs present ---'
$arcFound = $false
foreach ($g in (Get-CimInstance Win32_VideoController)) {
  Log ('name=' + $g.Name + ' driver=' + $g.DriverVersion + ' status=' + $g.Status)
  if ($g.Name -match 'Arc|B580') { $arcFound = $true }
}
Log ('arc_detected=' + $arcFound)
$vk = Test-Path (Join-Path $env:SystemRoot 'System32\vulkan-1.dll')
Log ('vulkan_runtime_present=' + $vk)

# ---------- 2. Fetch a prebuilt llama.cpp Vulkan build ----------
Log ''
Log '--- [2] Fetching prebuilt llama.cpp (Vulkan, win-x64) ---'
# llama.cpp marks every release prerelease, so /releases/latest returns the WRONG
# thing. List releases and take the newest one that has the win-vulkan asset.
$bin = Join-Path $root 'llama'
$asset = $null
$tag = $null
try {
  $rels = Invoke-RestMethod 'https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=15' -Headers @{ 'User-Agent' = 'arc-spike' }
  foreach ($r in $rels) {
    $cand = $r.assets | Where-Object { $_.name -match 'win-vulkan-x64\.zip$' } | Select-Object -First 1
    if ($cand) { $asset = $cand; $tag = $r.tag_name; break }
  }
} catch { Log ('RELEASE_QUERY_FAILED: ' + $_.Exception.Message) }

if (-not $asset) {
  Log 'ASSET_NOT_FOUND - no win-vulkan-x64.zip in the last 15 releases'
} else {
  Log ('llama_cpp_tag=' + $tag)
  Log ('asset=' + $asset.name + ' size_mb=' + [math]::Round($asset.size / 1MB, 1))
  $zip = Join-Path $root $asset.name
  try {
    if (-not (Test-Path $zip)) { Invoke-WebRequest $asset.browser_download_url -OutFile $zip }
    Remove-Item $bin -Recurse -Force -ErrorAction SilentlyContinue
    Expand-Archive $zip -DestinationPath $bin -Force
    Log ('unzipped_ok=' + (Test-Path $bin))
  } catch { Log ('DOWNLOAD_OR_UNZIP_FAILED: ' + $_.Exception.Message) }
}

$cliItem = Get-ChildItem $bin -Recurse -Filter 'llama-cli.exe' -ErrorAction SilentlyContinue | Select-Object -First 1
$cli = if ($cliItem) { $cliItem.FullName } else { $null }
Log ('llama_cli_found=' + [bool]$cli)
if ($cli) { Log ('llama_cli_path=' + $cli) }

# ---------- 3. DECISIVE: does Vulkan enumerate the B580? ----------
Log ''
Log '--- [3] Device enumeration (decisive) ---'
if ($cli) {
  try {
    $devs = & $cli --list-devices 2>&1 | Out-String
    Log $devs
  } catch { Log ('LIST_DEVICES_FAILED: ' + $_.Exception.Message) }
} else {
  Log 'SKIPPED - no llama-cli.exe'
}

# ---------- 4. Dispatch a real GPU workload (tiny model) ----------
Log ''
Log '--- [4] Real GPU workload ---'
$model = Join-Path $root 'tiny.gguf'
if (-not (Test-Path $model)) {
  $cands = @(
    'https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf',
    'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf',
    'https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf'
  )
  foreach ($u in $cands) {
    try {
      Log ('trying: ' + $u)
      Invoke-WebRequest $u -OutFile $model -ErrorAction Stop
      Log ('model_downloaded_from: ' + $u)
      break
    } catch { Log ('  failed: ' + $_.Exception.Message) }
  }
}
$modelPresent = Test-Path $model
Log ('model_present=' + $modelPresent)
if ($modelPresent) { Log ('model_size_mb=' + [math]::Round((Get-Item $model).Length / 1MB, 1)) }

if ($cli -and $modelPresent) {
  Log ''
  Log '--- generation with FULL GPU offload, ngl 99 ---'
  try {
    $outGpu = & $cli -m $model -p 'The capital of France is' -n 24 -ngl 99 --seed 0 -st 2>&1 | Out-String
    Log $outGpu
  } catch { Log ('GPU_RUN_FAILED: ' + $_.Exception.Message) }

  Log ''
  Log '--- control: same run forced to CPU, ngl 0 ---'
  try {
    $outCpu = & $cli -m $model -p 'The capital of France is' -n 24 -ngl 0 --seed 0 -st 2>&1 | Out-String
    Log $outCpu
  } catch { Log ('CPU_RUN_FAILED: ' + $_.Exception.Message) }
} else {
  Log 'SKIPPED generation - missing binary or model'
}

Log ''
Log ('=== END | report: ' + $log + ' ===')
Log 'Share this whole file.'
