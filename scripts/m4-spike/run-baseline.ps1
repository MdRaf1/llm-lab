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
    # This box (and modern Windows) exposes the standby list as three priority counters,
    # not the single "Standby Cache Standby List Bytes" path. Sum them for the real total.
    try {
        $n = (Get-Counter "\Memory\Standby Cache Normal Priority Bytes" -ErrorAction Stop).CounterSamples[0].CookedValue
        $c = (Get-Counter "\Memory\Standby Cache Core Bytes" -ErrorAction Stop).CounterSamples[0].CookedValue
        $r = (Get-Counter "\Memory\Standby Cache Reserve Bytes" -ErrorAction Stop).CounterSamples[0].CookedValue
        [double]($n + $c + $r)
    }
    catch { $null }
}

function Invoke-StandbyEviction {
    # Primary: purpose-built tools if on PATH (need elevation; likely absent on a bare box).
    $rammap = Get-Command RAMMap64.exe -ErrorAction SilentlyContinue
    if ($rammap) { & $rammap.Source -Et | Out-Null; Start-Sleep -Seconds 2; return }
    $esl = Get-Command EmptyStandbyList.exe -ErrorAction SilentlyContinue
    if ($esl) { & $esl.Source standbylist | Out-Null; Start-Sleep -Seconds 1; return }

    # Fallback (no elevation, disk-free): commit + touch private memory to force the OS to
    # drop clean file-cache (standby) pages — including the mmap'd model — to satisfy the
    # allocation, then release it. Preferred over a >RAM disk scratch: needs no free disk
    # (this box has < model+scratch headroom) and evicts directly via memory pressure.
    # ponytail: RAM-pressure eviction, single-shot per cold run; measured to crush standby
    #           ~8GB -> ~1GB here. Swap in RAMMap -Et if you get an elevated box.
    $chunkBytes = [int64](512MB)
    $chunks = New-Object System.Collections.Generic.List[byte[]]
    $cap = [int64](13GB)                 # hard ceiling so we never over-commit
    $touched = [int64]0
    try {
        while ($touched -lt $cap) {
            $avail = (Get-Counter "\Memory\Available Bytes" -ErrorAction Stop).CounterSamples[0].CookedValue
            if ($avail -lt 900MB) { break }          # leave the OS ~900MB headroom
            $b = New-Object byte[] $chunkBytes
            for ($i = 0; $i -lt $b.Length; $i += 4096) { $b[$i] = 1 }   # touch every page -> real commit
            $chunks.Add($b); $touched += $chunkBytes
        }
    } catch { }                                       # OutOfMemory just means we've applied enough pressure
    $chunks.Clear(); $chunks = $null
    [GC]::Collect(); [GC]::WaitForPendingFinalizers()
    Start-Sleep -Seconds 2
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
