# M4 Baseline Spike — Report (Task 3)

**Box:** AMD Ryzen 5 5600G (Zen 3), 6 physical cores / 12 logical, 15.7 GB RAM,
Windows 11 Pro. Non-elevated shell.
**Date:** 2026-09-23
**Branch:** `m4-baseline-spike`

## Verdict (one line)

**`no_go` → branch `escalate`, reason `model_fits_cache_moot`.** Warm steady
state decodes at **4.278 tok/s median** (min 4.220, max 4.583) while serving
only **111 MB/token median** of physical disk reads (min 107, max 161) — 4.6×
below the ~500 MB/token streaming-regime anchor. The stock CPU/mmap baseline on
this box is **not** bottlenecked on per-token expert scatter I/O, so the M4
patch premise does not survive contact here. Full gate:
`artifacts/m4/evidence/m4-gate.json`.

## Run parameters (Step 1)

- **Threads = 6** (physical cores): `(Get-CimInstance Win32_Processor).NumberOfCores` → 6.
- **NGen = 128.** Warm decode windows ran ~31 s (avg_ns ≈ 3.1e10) / ~87–101 s
  wall — far above the ~10 s floor at which the 1 s disk-counter Riemann sum
  would be too coarse (resolution 4). No NGen bump needed.
- Other heavy disk users quiesced during the sweep.

## Exact llama-bench command (per run)

Emitted by `scripts/m4-spike/run-baseline.ps1` (Task 2), one invocation per run:

```
llama-bench.exe -m models/m1/qwen3-gguf/Qwen3-30B-A3B-Q4_K_M.gguf \
  -ngl 0 --n-cpu-moe 999 -t 6 -n 128 -p 0 -r 1 -o json
```

Sweep driver: `1..5 cold` (evict each), `1..3 warmup` (discarded), `1..5 warm`.
Raw per-run JSON: `artifacts/m4/evidence/baseline-runs.jsonl` (13 lines:
5 cold + 3 warmup + 5 warm).

## The dominating physical fact: model (18.55 GB) > RAM (15.7 GB)

`model_size = 18550716416` (llama-bench JSON). RAM = 16886743040 bytes. The
model **cannot fully reside in cache**. Consequently *both* cold and warm runs
stream heavily from disk — warm runs read ~14 GB just like cold runs, because
the ~14 GB working set exceeds the ~10–13 GB reclaimable cache and is re-faulted
each process. This is why the naive "warm ≈ cached ≈ near-zero disk" expectation
does **not** hold here, and it is central to the verdict below.

## Measured numbers (every value from the PhysicalDisk counter + llama-bench avg_ts)

`miss_bytes_per_token = disk_bytes / tokens` per run, where `disk_bytes` is the
Riemann sum of `\PhysicalDisk(_Total)\Disk Read Bytes/sec` over each process —
**physical bytes the drive actually served, never cache-size arithmetic**
(`m4-gate.json:miss_bytes_source`). Source command per run:
`scripts/m4-spike/run-baseline.ps1`; aggregation: `scripts/m4-spike/emit-gate.py`.

### Warm steady state — PRIMARY gate (`gated_on: warm_steady_state`)

| metric | median | min | max |
|---|---|---|---|
| tok/s | **4.278** | 4.220 | 4.583 |
| miss bytes/token | **111.2 MB** | 107.0 MB | 161.1 MB |

- **warm_ceiling_tok_s (headline)** = scatter_bps / warm_miss_median = **9.18 tok/s**
- **warm_ceiling_min_tok_s (conservative, most misses)** = scatter_bps / warm_miss_max = **6.34 tok/s**

### Cold — worst-case bound only (NOT gated on)

| metric | median | min | max |
|---|---|---|---|
| tok/s | 4.829 | 3.362 | 5.399 |
| miss bytes/token | 101.95 MB (median) | — | — |
| cold_ceiling_tok_s | 10.02 tok/s | | |

`cold_verified` = **true for all 5 cold runs** (`cold_verified_all: true`). See
eviction note below.

## Scatter bandwidth + provenance (Step 3)

- **scatter_bps = 1021317250.7** (≈1.02 GB/s).
- Provenance: `"M3 microbench, same host (artifacts/m3/evidence/m3-gate.json:throughput.stock_bps.median)"`.
- Path taken (resolution 1): **PRIMARY — reuse the committed M3 same-host
  figure.** No M3 throughput-probe script is locatable under `scripts/` or
  `benchmarks/` (only `run-baseline.ps1` matches; `benchmarks/` holds unrelated
  speculative-decode benchmarks), so the same-session re-run path was not taken.
  Details: `artifacts/m4/evidence/scatter-bandwidth.txt`.
- This is independent of the decode run's own tok/s; the ceiling is
  `scatter_bps / miss_bytes_per_token`, never derived from decode throughput
  (non-circular).

## `miss_frac` ≈ 500 MB/token band anchor (Step 5 rationale)

`miss_frac = 0.5` → threshold `0.5 × PER_TOKEN_EXPERT_BYTES` = 0.5 × 1.019e9 ≈
**510 MB/token** (`aggregate.PER_TOKEN_EXPERT_BYTES` = top-8 × 48 layers ×
2.65 MB/expert). It sits mid-band between the ~2.6 tok/s pure-drive floor and
the ~10 tok/s interactive line: above it, each token is re-reading most of its
~1 GB expert working set from disk (a true streaming regime the M4 patch would
attack); below it, the OS is caching/reusing enough that decode is not
scatter-I/O-bound.

## Regime determination + which reason fired

- **streaming_regime = false.** warm_miss_median 111 MB/token ≤ 510 MB anchor.
- **headroom = true** (warm_tok_s_max 4.583 < warm_ceiling_min 6.34) — but moot,
  since a non-streaming box gates to `model_fits_cache_moot` regardless.
- **`reason = model_fits_cache_moot` → `no_go` / `escalate` / `passed: false`.**

**This is a clean determination, NOT a near-boundary judgment call.** The whole
warm spread (107–161 MB/token) lies 3.2×–4.8× below the 510 MB anchor; no warm
value comes within its own spread of the boundary. It is further **physically
bounded**: at the measured 1.02 GB/s scatter rate, crossing 510 MB/token over a
~31 s decode would require ~65 GB of reads in 31 s (~2.1 GB/s sustained) —
impossible at 1.02 GB/s. So this box cannot be in the streaming regime at
NGen=128 no matter the measurement noise.

**Why not-streaming despite model > RAM (the honest nuance):** within a single
128-token `llama-bench` process the ~14 GB working set faults in *once* (mmap
pages stay resident for the process) and is reused across tokens — so per-token
*marginal* disk traffic is low; the ~14 GB total is dominated by first-touch
load faults amortised over 128 tokens. `disk_bytes` therefore conflates
one-time load with steady-state decode miss, making the reported miss/token an
**upper bound** on true steady-state decode miss. Since even that upper bound is
4.6× under the anchor (and a longer NGen would only lower it), the moot verdict
is conservative and robust. The real ~4.3 tok/s ceiling on this box is **CPU
compute** (6 Zen 3 cores on a 3B-active MoE at Q4_K_M), not expert scatter I/O —
which is exactly the condition under which the M4 I/O patch is moot.

## `--n-cpu-moe` inertness note

`m4-gate.json:ncpumoe_note` = "flag ~inert on CPU-only box; baseline is OS-paged
mmap". With `-ngl 0 --n-cpu-moe 999` on a CPU-only build (`backends: CPU`,
`use_mmap: true` in the llama-bench JSON), there is no GPU to offload experts
*from*, so `--n-cpu-moe` changes nothing physical — every expert is already a
host mmap page faulted by the OS pager. The gate measures that OS-paged mmap
baseline, which is the correct stock reference for gate #2.

## `cold_verified` status + eviction (resolution 3)

All 5 cold runs report `cold_verified: true`. To get there I fixed two real
defects in the Task 2 harness `run-baseline.ps1` (this is the "fix eviction"
resolution 3 authorises; committed with the evidence):

1. **Wrong standby counter.** `\Memory\Standby Cache Standby List Bytes` does
   **not exist** on this box (it returned null → `cold_verified` was
   unconditionally false, a false negative). Replaced with the sum of the three
   real counters that do exist: `Standby Cache Normal Priority + Core + Reserve
   Bytes`.
2. **Infeasible disk-scratch eviction.** The fallback wrote a >RAM (~18.9 GB)
   scratch file, but only 12.5 GB is free on C: — it would fill the disk and
   throw. Replaced with a **disk-free RAM-pressure eviction**: commit+touch
   private memory (capped 13 GB, 900 MB avail floor) to force the OS to drop
   clean file-cache (standby) pages, then release. Measured before committing:
   standby 8.2 GB → 1.2 GB, i.e. genuine eviction. RAMMap/EmptyStandbyList
   primary paths are retained for elevated boxes.

Both fixes are instrument corrections (they make `cold_verified` reflect
reality); neither touches the tok/s or disk_bytes measurement path.

## AVX2 build evidence (resolution 5 — resolves deferred Task-1 minor)

This llama-bench build emits **no** runtime `system_info: … AVX2 = 1 …` line
(neither to stdout nor stderr, in `-o json` or markdown mode — the ggml
system-info print is absent in this revision). AVX2 is instead evidenced by
build config, which is definitive:

- `artifacts/m4/evidence/build-log.txt`: "Adding CPU backend variant ggml-cpu:
  **-march=native**"; configured with `-DGGML_NATIVE=ON`.
- `artifacts/m4/evidence/build-env.json`: `cpu_variant` = "-march=native (AVX2
  box; ggml x86 CPU backend)".
- Runtime JSON: `build_commit 1a064ab09`, `built with Clang 23.1.2`,
  `backends: CPU`, `cpu_info: AMD Ryzen 5 5600G` (Zen 3 → AVX2/FMA/F16C).
- `--n-cpu-moe` present in `llama-bench --help` (`-ncmoe, --n-cpu-moe <n>`).

`-march=native` on a Zen 3 CPU compiles in AVX2/FMA/F16C, so the baseline binary
is the intended CPU/AVX2 stock build even though the explicit `GGML_AVX2` cache
flag reads OFF (native arch supersedes it).

## Measurement caveats (honesty, spec §10)

- **disk_bytes sampling cadence:** `Get-Counter` itself costs ~1 s and the loop
  adds `Start-Sleep 1 s`, so true sample spacing is ~2 s while the Riemann sum
  multiplies by `interval = 1.0 s` — disk_bytes could be off by up to ~2× in
  magnitude. The verdict is robust to this: 2× either direction keeps warm
  miss/token (56–222 MB) below the 510 MB anchor, and the drive-bandwidth bound
  above forbids crossing it regardless.
- **warm #5 outlier:** 20.6 GB read (vs ~14 GB others), max miss 161 MB/token —
  background OS/disk activity on a bare box; still well sub-threshold, and the
  gate commits it as `warm_miss_bytes_per_token_max`.
- Cold is a worst-case bound only; the go/no-go rests entirely on warm.

## Files

- `artifacts/m4/evidence/baseline-runs.jsonl` — 13 raw runs.
- `artifacts/m4/evidence/scatter-bandwidth.txt` — bandwidth + provenance.
- `artifacts/m4/evidence/m4-gate.json` — committed gate (`spike_verdict: no_go`).
- `scripts/m4-spike/emit-gate.py` — aggregation/emit driver.
- `scripts/m4-spike/run-baseline.ps1` — eviction/counter fixes (this task).

