#!/usr/bin/env pwsh
# Active-set pivot probe (option 2): force a WIDE hot-expert working set on the
# SAME model with a high-entropy multi-domain prompt (~2.7k varied tokens ->
# near-full 128-expert/layer coverage), then watch the physical drive.
#   disk saturates near ~1.02 GB/s + tok/s falls toward ~4.3  -> streaming DORMANT (real elsewhere)
#   disk stays idle, CPU threads peg                           -> streaming genuinely DEAD here
param([int]$Ctx = 8192, [int]$NGen = 48, [int]$Threads = 6, [string]$PromptFile = "", [string]$Tag = "wideset")
$ErrorActionPreference = "Stop"
$cli   = "C:/Rafi/Projects/llama.cpp/build/bin/llama-cli.exe"
$model = "models/m1/qwen3-gguf/Qwen3-30B-A3B-Q4_K_M.gguf"
$evdir = "artifacts/m4/evidence"
if ($PromptFile -eq "") { $PromptFile = "$evdir/probe-wideset-prompt.txt" }
$promptFile = $PromptFile
$errOut = "$evdir/probe-$Tag.err"
$outOut = "$evdir/probe-$Tag.out"
$tlOut  = "$evdir/probe-counters-$Tag.txt"
$cliArgs = @("-m",$model,"-ngl","0","--n-cpu-moe","999","-t","$Threads","-c","$Ctx","-n","$NGen","--temp","0","-st","-no-cnv","-f",$promptFile)
$p = Start-Process -FilePath $cli -ArgumentList $cliArgs -NoNewWindow -PassThru -RedirectStandardOutput $outOut -RedirectStandardError $errOut
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
Write-Host "DONE exit=$($p.ExitCode) samples=$($rows.Count-1) ctx=$Ctx"
Get-Content $outOut | Select-String -Pattern "Prompt:|Generation:|t/s" | ForEach-Object { $_.Line }
