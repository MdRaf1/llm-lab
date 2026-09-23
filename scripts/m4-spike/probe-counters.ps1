#!/usr/bin/env pwsh
# Compute-headroom probe: run llama-bench at a fixed thread config while sampling
# all-core CPU% and physical disk read B/s, so the post-tuning binding ceiling
# (compute / disk-I/O / RAM-bandwidth) can be attributed, per ADR-0002 rule (b).
#
# The counters are the same OS sources M4 used:
#   \Processor(_Total)\% Processor Time      -> 0..100, avg across all logical cores
#   \PhysicalDisk(_Total)\Disk Read Bytes/sec -> physical bytes the drive served
# One llama-bench process = one model load (disk-heavy) followed by -r tg reps;
# the timeline is emitted raw (elapsed,cpu_pct,disk_read_bps) so the load spike and
# the steady tg regime are visible and split honestly in analysis.
param(
  [int]$Threads = 6,
  [int]$NGen = 128,
  [int]$Reps = 5,
  [string]$OutTag = ""
)
$ErrorActionPreference = "Stop"
$bench = "C:/Rafi/Projects/llama.cpp/build/bin/llama-bench.exe"
$model = "models/m1/qwen3-gguf/Qwen3-30B-A3B-Q4_K_M.gguf"
$evdir = "artifacts/m4/evidence"
if ($OutTag -eq "") { $OutTag = "t$Threads" }
$jsonOut = "$evdir/probe-signals-$OutTag.json"
$tlOut   = "$evdir/probe-counters-$OutTag.txt"

$benchArgs = @("-m",$model,"-ngl","0","--n-cpu-moe","999","-t","$Threads","-n","$NGen","-p","0","-r","$Reps","-o","json")
$p = Start-Process -FilePath $bench -ArgumentList $benchArgs -NoNewWindow -PassThru -RedirectStandardOutput $jsonOut -RedirectStandardError "$evdir/probe-signals-$OutTag.err"

$t0 = Get-Date
$rows = New-Object System.Collections.Generic.List[string]
$rows.Add("elapsed_s`tcpu_pct`tdisk_read_bps")
while (-not $p.HasExited) {
  $s = Get-Counter -Counter '\Processor(_Total)\% Processor Time','\PhysicalDisk(_Total)\Disk Read Bytes/sec' -ErrorAction SilentlyContinue
  if ($s) {
    $cpu  = ($s.CounterSamples | Where-Object { $_.Path -like '*processor time*' }).CookedValue
    $disk = ($s.CounterSamples | Where-Object { $_.Path -like '*disk read bytes*' }).CookedValue
    $el = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
    $rows.Add(("{0}`t{1:N1}`t{2:N0}" -f $el, $cpu, $disk))
  }
  Start-Sleep -Milliseconds 500
}
$rows | Set-Content -Path $tlOut -Encoding utf8
Write-Host "DONE threads=$Threads exit=$($p.ExitCode) samples=$($rows.Count-1) -> $tlOut ; bench json -> $jsonOut"
