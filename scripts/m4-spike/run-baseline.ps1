#requires -Version 5.1
<#
run-baseline.ps1 — one baseline decode run for the M4 spike.
Cold runs evict the OS standby list first (so the model streams from disk);
every run samples the PHYSICAL disk-read counter across the decode window,
so warm residual miss traffic is the observed drive-served bytes — never
inferred from cache-size arithmetic. Emits one JSON line for aggregate.py.
Thin I/O glue: exercised for real in Task 3, no unit test here.
#>
param(
    [Parameter(Mandatory = $true)][string]$Bin,          # dir holding llama-bench.exe
    [Parameter(Mandatory = $true)][string]$Model,        # path to .gguf
    [int]$NGen = 128,
    [int]$Threads = 8,
    [switch]$Cold,
    [string]$State = "warm",                              # cold | warm | warmup
    [string]$OutJson                                      # append one JSONL line here; else stdout
)
$ErrorActionPreference = "Stop"

$bench = Join-Path $Bin "llama-bench.exe"
if (-not (Test-Path $bench)) { throw "llama-bench.exe not found at $bench" }

function Get-StandbyBytes {
    try { (Get-Counter "\Memory\Standby Cache Standby List Bytes" -ErrorAction Stop).CounterSamples[0].CookedValue }
    catch { $null }
}

function Invoke-StandbyEviction {
    # Primary: purpose-built tools if on PATH (need elevation; likely absent on a bare box).
    $rammap = Get-Command RAMMap64.exe -ErrorAction SilentlyContinue
    if ($rammap) { & $rammap.Source -Et | Out-Null; Start-Sleep -Seconds 2; return }
    $esl = Get-Command EmptyStandbyList.exe -ErrorAction SilentlyContinue
    if ($esl) { & $esl.Source standbylist | Out-Null; Start-Sleep -Seconds 1; return }

    # Fallback (no elevation): read a >RAM scratch file so the cache fills with junk
    # and the model's cached pages are pushed out of the standby list.
    $ramBytes = [int64](Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
    $need = $ramBytes + 2GB
    $scratch = Join-Path $env:TEMP "m4-spike-cacheflush.bin"
    # ponytail: reuse a big scratch file across runs (recreating a >RAM file each cold run is slow);
    #           delete it manually when the spike is done. Recreate a per-run temp if isolation matters.
    if ((-not (Test-Path $scratch)) -or ((Get-Item $scratch).Length -lt $need)) {
        $buf = New-Object byte[] (64MB)
        (New-Object Random).NextBytes($buf)   # random so it can't be dedup/compressed away
        $fs = [System.IO.File]::Create($scratch)
        try { $written = [int64]0; while ($written -lt $need) { $fs.Write($buf, 0, $buf.Length); $written += $buf.Length } }
        finally { $fs.Dispose() }
    }
    $buf = New-Object byte[] (64MB)
    $fs = [System.IO.File]::OpenRead($scratch)
    try { while ($fs.Read($buf, 0, $buf.Length) -gt 0) {} } finally { $fs.Dispose() }
    Start-Sleep -Seconds 1
}

$coldVerified = $null
if ($Cold) {
    $pre = Get-StandbyBytes
    Invoke-StandbyEviction
    $post = Get-StandbyBytes
    # Verified only if the standby list actually dropped. The fallback path often
    # can't guarantee this (scratch data refills standby), so it honestly reports false.
    $coldVerified = ($null -ne $pre) -and ($null -ne $post) -and ($post -lt $pre)
}

# --- run llama-bench (tg-only) while sampling the physical disk-read rate ---
$tmpOut = [System.IO.Path]::GetTempFileName()
$tmpErr = [System.IO.Path]::GetTempFileName()
$benchArgs = @("-m", $Model, "-ngl", "0", "--n-cpu-moe", "999",
               "-t", "$Threads", "-n", "$NGen", "-p", "0", "-r", "1", "-o", "json")
$proc = Start-Process -FilePath $bench -ArgumentList $benchArgs -NoNewWindow -PassThru `
                      -RedirectStandardOutput $tmpOut -RedirectStandardError $tmpErr

$interval = 1.0
$samples = New-Object System.Collections.Generic.List[double]
while (-not $proc.HasExited) {
    try {
        $c = Get-Counter "\PhysicalDisk(_Total)\Disk Read Bytes/sec" -ErrorAction Stop
        $samples.Add([double]$c.CounterSamples[0].CookedValue)
    } catch {}
    Start-Sleep -Seconds $interval
}
$proc.WaitForExit()

# Riemann sum of the bytes/sec rate over the sampled window → total physical bytes served.
$diskBytes = (($samples | Measure-Object -Sum).Sum) * $interval

$raw = Get-Content $tmpOut -Raw
Remove-Item $tmpOut, $tmpErr -ErrorAction SilentlyContinue
if ([string]::IsNullOrWhiteSpace($raw)) { throw "llama-bench produced no JSON (exit $($proc.ExitCode))" }
$results = $raw | ConvertFrom-Json
$tg = $results | Where-Object { $_.n_gen -gt 0 } | Select-Object -First 1
if ($null -eq $tg) { throw "no token-generation (n_gen>0) result in llama-bench JSON" }

$out = [ordered]@{
    state      = $State
    tok_s      = [double]$tg.avg_ts        # decode throughput = avg_ts of the tg result
    disk_bytes = $diskBytes                # physical drive-served bytes over the run window
    tokens     = [int64]$tg.n_gen          # generated-token count
}
if ($Cold) { $out.cold_verified = [bool]$coldVerified }

$line = $out | ConvertTo-Json -Compress
if ($OutJson) { Add-Content -Path $OutJson -Value $line } else { Write-Output $line }
