#!/usr/bin/env pwsh
# Diverse-routing check: llama-bench decodes ONE repeated token (minimal active
# expert set, trivially cache-resident). This runs llama-cli on a varied
# multi-topic prompt so prompt-processing + generation route to many experts,
# then reads (a) llama-cli's own load-excluded eval tok/s and (b) physical disk
# read B/s DURING generation. If a realistic active set exceeds RAM it re-faults
# per token -> disk saturates near ~1 GB/s and tok/s drops toward M4's ~4.3
# (a real streaming regime); if it stays cache-resident -> disk idle, tok/s ~warm.
$ErrorActionPreference = "Stop"
$cli   = "C:/Rafi/Projects/llama.cpp/build/bin/llama-cli.exe"
$model = "models/m1/qwen3-gguf/Qwen3-30B-A3B-Q4_K_M.gguf"
$evdir = "artifacts/m4/evidence"
$prompt = "Explain how a jet engine produces thrust, then describe the water cycle, list three causes of the French Revolution, summarize the plot of Hamlet, give the recipe for a simple omelette, explain compound interest, describe photosynthesis at the molecular level, and outline how TCP establishes a connection."
$promptFile = "$evdir/probe-diverse-prompt.txt"
Set-Content -Path $promptFile -Value $prompt -Encoding utf8 -NoNewline
$errOut = "$evdir/probe-diverse.err"
$tlOut  = "$evdir/probe-counters-diverse.txt"
$cliArgs = @("-m",$model,"-ngl","0","--n-cpu-moe","999","-t","6","-n","256","--temp","0","-st","-no-cnv","-f",$promptFile)
$p = Start-Process -FilePath $cli -ArgumentList $cliArgs -NoNewWindow -PassThru -RedirectStandardOutput "$evdir/probe-diverse.out" -RedirectStandardError $errOut
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
Write-Host "DONE exit=$($p.ExitCode) samples=$($rows.Count-1)"
Write-Host "--- llama-cli timing (stderr tail) ---"
Get-Content $errOut | Select-String -Pattern "eval time|tokens per second|load time|prompt eval" | ForEach-Object { $_.Line }
