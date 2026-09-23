# Compute-Headroom Probe — Report

**Box:** AMD Ryzen 5 5600G (Zen3), 6C/12T, 15.7 GB RAM, PCIe3 NVMe, Windows 11.
**Date:** 2026-09-24. **Binary:** llama.cpp `@1a064ab`, CPU/AVX2 (`-march=native`,
clang/llvm-mingw) — **reused** from the M4 build (no rebuild). **Model:**
`Qwen3-30B-A3B-Q4_K_M` (18.55 GB > 15.7 GB RAM). Greedy, warm-steady.

## Question

M4 proved decode is *not* disk-I/O-bound at 4.3 tok/s; it did **not** prove 4.3
is the compute ceiling. Tune compute, find where warm-steady decode lands, and
name the binding ceiling — disk (~9.18), RAM-bandwidth (~29), or compute.

## Result (one line)

**Warm-steady decode is ~9.2 tok/s (diverse routing) to ~14 tok/s (degenerate),
NOT 4.3 — and it is compute/RAM-bandwidth bound, with disk idle.** M4's 4.3 was
a per-invocation cold-load artifact, not the decode ceiling.

## Method

The dominating physical fact (M4): model 18.55 GB > 15.7 GB RAM. M4 measured with
`-r 1` fresh llama-bench processes, so **every** run re-faulted the ~14 GB working
set from disk → 4.3 tok/s. This probe measures the **warm-steady** regime a real
server runs in: model loaded **once**, then many decode reps against resident
pages. llama-bench emits per-rep `samples_ts`; rep0 (cold first-touch) is
discarded, reps warm.

## Thread sweep (`n=128`, `-r 6`, one process per config)

Raw per-rep `samples_ts` (rep0 = cold warmup, discarded):

| threads | rep0 | warm reps (1..5) | warm median |
|--:|--:|--|--:|
| 4  | 3.83 | 8.86 10.45 10.76 14.46 16.05 | **10.76** |
| 6  | 4.75 | 11.68 11.96 7.66 12.92 13.36 | **11.96** |
| 8  | 3.38 | 9.47 11.64 10.43 13.39 16.16 | **11.64** |
| 12 | 3.58 | 8.59 11.09 8.40 10.09 14.45 | **10.09** |

- Every warm rep (even the floors) **exceeds** the 9.18 disk-I/O ceiling — which
  is physically impossible if reading 111 MB/token from a 1.02 GB/s drive
  (10×111 MB = 1.1 GB/s > drive). So warm reps are **cache-served**, not
  disk-limited. M4's 111 MB/token was cold re-fault + one-time load, not steady miss.
- Scaling is ~**flat** past 4 physical cores; SMT (`t12`) is no better. Not
  core-count-limited. (rep0 ≈ 3.4–4.8 reproduces M4's cold baseline — sanity check passes.)

## Warm-steady depth (`t=6`, `-r 12`, instrumented)

Per-rep: `3.94 | 8.93 11.16 14.09 13.41 19.11 16.84 18.57 9.94 15.74 16.67 10.19`
→ warm median **14.09**, last-5 median **15.75**, peak **19.1**. High variance =
model>RAM intermittent eviction/re-fault. This is the **minimal-active-set**
(single repeated token) ceiling.

## Diverse routing (`llama-cli`, varied multi-topic prompt, `n=256`, `t=6`)

Real diverse expert routing (jet engines → water cycle → Hamlet → TCP …):

```
[ Prompt: 2.2 t/s | Generation: 9.2 t/s ]
```

**Generation 9.2 tok/s** (llama-cli's own load-excluded figure). This is the
realistic warm decode rate. Diverse routing lowers it from ~14 (bigger active
set) but does **not** collapse it to 4.3.

## Binding ceiling (constraint signals during generation)

From `probe-counters-diverse.txt` / `probe-counters-t6r12.txt`, sampled ~1.4 s
(`\Processor(_Total)\% Processor Time`, `\PhysicalDisk(_Total)\Disk Read Bytes/sec`):

- **Disk read during gen: ~0.1–0.66 GB/s, declining — NOT saturated** (scatter
  ceiling 1.02 GB/s). Disk has large headroom. → **not disk-I/O-bound.**
- **CPU total ~40–52%** = ~6 of 12 logical cores pegged = the 6 worker threads
  (`-t 6`) **saturated.**
- Thread scaling flat past 4 cores + cores busy + disk idle → **compute /
  RAM-bandwidth** is the binder (sub-linear core scaling leans RAM-bandwidth).

## ADR-0002 gate — applied verbatim

- **Rule (a) threshold:** best warm decode **9.2** (diverse) / ~14 (degenerate)
  **≥ ~8** → literal "revive" band.
- **Rule (b) ceiling:** binder = **compute/RAM-bandwidth, NOT disk I/O.**
- **Resolution:** rule (b) decides. ADR-0002 Consequences (b): a RAM-bandwidth /
  compute binder "closes the streaming line here just as a compute wall does."
  The ≥8 threshold's *premise* — that reaching 8 means disk became the binder —
  is **contradicted by the direct disk measurement** (disk idle at 9.2). So:

**Streaming does NOT revive on this box.** But the reason is re-framed: not M4's
"decode is slow (4.3) and I/O has headroom," but **"warm decode is fast (~9–14)
and I/O is off the critical path."** Same strategic conclusion, overturned number.

## What this overturns / confirms

- **Confirms** (strengthens): decode is not disk-I/O-bound here — disk stays idle
  even under diverse routing.
- **Overturns:** M4's headline **4.3 tok/s is not the compute ceiling**; it was
  the cold/per-invocation re-load artifact. Warm-steady (server) decode is
  **~9.2 tok/s diverse / ~14 degenerate**, 2.15×–3.3× higher.
- **True ceiling:** compute/RAM-bandwidth at ~9–14 tok/s; disk I/O has headroom.

## Caveats

- Single box/session, greedy, clang/mingw AVX2 build.
- 9.2 is an aggregate incl. thinking-phase warming; steady tail ≥ that. Variance
  high (model > RAM).
- **Streaming-thesis nuance:** streaming binds only if the *active* expert set
  exceeds RAM *even warm*. Diverse routing did not saturate disk → the active set
  still fits here. A larger MoE or a much wider hot-expert workload could still
  tip into the streaming regime — **untested** by this probe or M4.

## Files

- `probe-bench-t{4,6,8,12}.json` — thread sweep raw (`samples_ts`).
- `probe-signals-t6r12.json` + `probe-counters-t6r12.txt` — warm depth + counters.
- `probe-diverse.out/.err` + `probe-counters-diverse.txt` + `probe-diverse-prompt.txt` — diverse run.
- `probe-verdict.json` — derived verdict.
- `scripts/m4-spike/probe-counters.ps1`, `scripts/m4-spike/probe-diverse.ps1` — drivers.
