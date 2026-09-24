# ADR-0005 day-1 kill-switch probe — Intel Arc B580
#
# Answers ONE question: does any GPU workload run on the B580?
# Read-only w.r.t. the system: downloads into C:\arc-spike, installs nothing,
# needs no admin, touches no drivers. Delete that folder to undo everything.
#
# Usage (normal PowerShell on the Arc PC):  .\day1-probe.ps1
# Then share C:\arc-spike\day1-report.txt

$ErrorActionPreference = 'Continue'
$ProgressPreference    = 'SilentlyContinue'
$root = 'C:\arc-spike'
$log  = "$root\day1-report.txt"
New-Item -ItemType Directory -Force -Path $root | Out-Null
Remove-Item $log -ErrorAction SilentlyContinue
function Log($m) { $m | Tee-Object -FilePath $log -Append }

Log "=== ADR-0005 ARC DAY-1 PROBE | $(Get-Date -Format o) ==="
$os = Get-CimInstance Win32_OperatingSystem
$cs = Get-CimInstance Win32_ComputerSystem
Log "host=$env:COMPUTERNAME user=$env:USERNAME"
Log "os=$($os.Caption) build=$($os.Version)"
Log "cpu=$((Get-CimInstance Win32_Processor).Name)"
Log "ram_gb=$([math]::Round($cs.TotalPhysicalMemory/1GB,2))"
Log "free_disk_gb=$([math]::Round((Get-PSDrive C).Free/1GB,2))"

# ---------- 1. Is the card there, and is its driver healthy? ----------
Log "`n--- [1] GPUs present ---"
$arcFound = $false
Get-CimInstance Win32_VideoController | ForEach-Object {
  Log ("name='{0}' driver={1} status={2} drv_date={3}" -f `
       $_.Name, $_.DriverVersion, $_.Status, $_.DriverDate)
  if ($_.Name -match 'Arc|B580|Intel.*Graphics') { $arcFound = $true }
}
Log "arc_detected=$arcFound"

Log "`n--- [1b] Vulkan runtime present? ---"
$vk = Test-Path "$env:SystemRoot\System32\vulkan-1.dll"
Log "vulkan_1_dll=$vk"
$icd = "$env:SystemRoot\System32\DriverStore\FileRepository"
Log "intel_vulkan_icd_hits=$((Get-ChildItem $icd -Recurse -Filter '*intel_icd*' -ErrorAction SilentlyContinue).Count)"

# ---------- 2. Get a prebuilt llama.cpp Vulkan build (no compiler, no oneAPI) ----------
Log "`n--- [2] Fetching prebuilt llama.cpp (Vulkan, win-x64) ---"
$bin = "$root\llama"
try {
  $rel = Invoke-RestMethod 'https://api.github.com/repos/ggml-org/llama.cpp/releases/latest' `
           -Headers @{ 'User-Agent' = 'arc-spike' }
  Log "llama_cpp_release=$($rel.tag_name)"
  $asset = $rel.assets | Where-Object { $_.name -match 'win.*vulkan.*x64.*\.zip$' } | Select-Object -First 1
  if (-not $asset) {
    Log "ASSET_NOT_FOUND. available:"
    $rel.assets | ForEach-Object { Log "  $($_.name)" }
  } else {
    Log "asset=$($asset.name) size_mb=$([math]::Round($asset.size/1MB,1))"
    $zip = "$root\$($asset.name)"
    if (-not (Test-Path $zip)) { Invoke-WebRequest $asset.browser_download_url -OutFile $zip }
    Remove-Item $bin -Recurse -Force -ErrorAction SilentlyContinue
    Expand-Archive $zip -DestinationPath $bin -Force
    Log "unzipped_ok=$(Test-Path $bin)"
  }
} catch { Log "DOWNLOAD_FAILED: $($_.Exception.Message)" }

$cli = Get-ChildItem $bin -Recurse -Filter 'llama-cli.exe' -ErrorAction SilentlyContinue | Select-Object -First 1
Log "llama_cli_found=$($null -ne $cli)"
if ($cli) { Log "llama_cli_path=$($cli.FullName)" }

# ---------- 3. THE DECISIVE SIGNAL: does Vulkan enumerate the B580? ----------
Log "`n--- [3] Device enumeration (decisive) ---"
if ($cli) {
  $dir = $cli.Directory.FullName
  try {
    $devs = & $cli.FullName --list-devices 2>&1 | Out-String
    Log $devs
  } catch { Log "LIST_DEVICES_FAILED: $($_.Exception.Message)" }
} else { Log "SKIPPED — no llama-cli.exe" }

# ---------- 4. Actually dispatch a GPU workload (small model) ----------
Log "`n--- [4] Real GPU workload ---"
$model = "$root\tiny.gguf"
if (-not (Test-Path $model)) {
  $cands = @(
    'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf',
    'https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf',
    'https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf'
  )
  foreach ($u in $cands) {
    try {
      Log "trying: $u"
      Invoke-WebRequest $u -OutFile $model -ErrorAction Stop
      Log "model_downloaded_from=$u"
      break
    } catch { Log "  failed: $($_.Exception.Message)" }
  }
}
Log "model_present=$(Test-Path $model)"
if (Test-Path $model) { Log "model_size_mb=$([math]::Round((Get-Item $model).Length/1MB,1))" }

if ($cli -and (Test-Path $model)) {
  Log "`n>>> generation with FULL GPU offload (-ngl 99) <<<"
  try {
    $out = & $cli.FullName -m $model -p "The capital of France is" -n 24 -ngl 99 --seed 0 -no-cnv 2>&1 | Out-String
    Log $out
  } catch { Log "GENERATION_FAILED: $($_.Exception.Message)" }

  Log "`n>>> control: same run forced to CPU (-ngl 0) <<<"
  try {
    $out0 = & $cli.FullName -m $model -p "The capital of France is" -n 24 -ngl 0 --seed 0 -no-cnv 2>&1 | Out-String
    Log $out0
  } catch { Log "CPU_RUN_FAILED: $($_.Exception.Message)" }
} else { Log "SKIPPED — missing binary or model" }

Log "`n=== END | report: $log ==="
Log "Share this whole file."
