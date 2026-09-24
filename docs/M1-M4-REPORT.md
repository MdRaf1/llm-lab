# M1–M4 Capstone — An Exact MoE Reference Pipeline and a Measured Negative on Low-RAM CPU Streaming

**Scope of this document.** A cross-milestone synthesis sitting *above* the
per-milestone reports (`artifacts/m*/evidence/M*-REPORT.md`). It records the arc
M0→M4 plus the post-M4 compute-headroom probe, cites every claim to committed
evidence by path, and states the negative result at its honest scope. It is the
project's own record; publication is out of scope for this pass.

**Box (unless noted):** AMD Ryzen 5 5600G (Zen3), 6C/12T, 15.7 GiB RAM, PCIe3
NVMe, Windows 11. Greedy, seed 0, 6 threads.

---

## Headline

Two things were built and one thing was learned:

1. **An exact MoE reference pipeline** (M1–M3): a deterministic, byte-identity
   MoE runtime for `Qwen3-30B-A3B-Q4_K_M`, with routing traces and a provably
   lossless expert-contiguous repack. Every number measured on this box.

2. **A rigorous, over-determined negative** on the streaming thesis for *this*
   box+model, **corrected by the probe**: decode here is **not disk-I/O-bound**.
   The M4 baseline's 4.3 tok/s was a *per-invocation cold-load artifact*; **warm,
   sustained, in-process decode is ~9.2 tok/s** (diverse routing; up to ~14
   degenerate), bound by **compute / RAM-bandwidth**, with the drive idle.

3. **Streaming is dormant, not dead — but weak here.** Widening the active expert
   set re-introduces real per-token expert-fault I/O (~26% penalty at equal
   context length), but it stays a *secondary* binder and the drive never
   saturates. Streaming's real home is a larger MoE / higher-bandwidth tier
   (relocation), not this model on this box.

Direction (ADR-0003): streaming **closed as a primary win here**, **validated but
relocation-gated**; conclude and record M1–M4 now; PageCC (M6/M7) stays
novelty-gated.

## The honesty discipline (governs every number below)

From `docs/HANDOFF.md`, applied throughout — the repo violated all five once and
now does not:

1. No number without its command+output committed. 2. Baselines measured
same machine/session/checkpoint, never a literature constant. 3. Projections
labeled **PROJECTED** with formula+inputs. 4. Output identity = token-ID
equality, never `.strip()`. 5. "Lossless/exact" always names
checkpoint+quantization+decode+seed+host.

<!-- APPEND-MARKER -->

## Milestone by milestone

### M0 — truth-in-labeling

The `sim_*` modules are simulations and stay labelled as such; the earlier
fabricated claims and the deleted `DEMO_SCRIPT.md` were addressed as a diff
against `a248a03`. This milestone is *why* the five reporting rules exist.

### M1 — exact reference pipeline (MEASURED)

Deterministic MoE reference, gated in order M1a→M1b→M1c
(`artifacts/m1/evidence/M1-REPORT.md`).

- **M1a** (tiny random OLMoE, FP32, no download): two independent builds produced
  **identical** token IDs (`m1a-oracle.json`: `identical:true`, `n_compared:32`).
- **M1b** (`OLMoE-1B-7B-0924`, both numeric paths): GGUF Q4_K_M **36.33 tok/s**
  (RSS 7.48 GB) and HF int8 **2.00 tok/s** (RSS 12.52 GB), **each deterministic
  against itself** (`m1b-gguf-oracle.json`, `m1b-hf-oracle.json`). The two paths
  *happened* to emit identical tokens on one greedy prompt — reported as
  **observed convergence, not routing/decode identity** (the oracle would refuse
  to compare them: different fingerprints).
- **M1c** (`Qwen3-30B-A3B`, the target): local Q4_K_M **6.02 tok/s**, RSS
  **13.76 GB**, n=256, **deterministic** across two runs (`m1c-oracle.json`,
  `n_compared:256`). Geometry measured: 48 layers × 128 experts, top-8,
  `total_expert_bytes` 17.55 GB (`m1c-meta.json`). BF16 **routing traces**
  (2000 positions, 3 domains) captured on a rented 135 GB Linux host
  (`m1c-gate.json`, `passed:true`).

**Two numeric paths are never asserted equal.** Exactness = token-ID equality
*within one path*, on `Qwen3-30B-A3B-Q4_K_M` under llama.cpp greedy, seed 0,
6 threads, on the recorded host — never "identical to BF16/full precision".
Whether Q4 routing matches BF16 routing is **spec §11 open-Q1, UNRESOLVED**.

### M2 — tier gate (PROJECTED)

The 30B BF16 raw trace was **lost** (rented VPS destroyed; only summaries
survive), so M2 is an explicit **PROJECTION** from int8-OLMoE traces onto the
measured 30B Q4 geometry, over two labelled hops (int8→Q4, OLMoE→30B)
(`artifacts/m2/evidence/M2-REPORT.md`).

- Locality is **real, not flat**: union growth sublinear in the window; miss rate
  falls to ~7% at full residency; reuse distance tightly clustered.
- Projected 16 GB ceiling **25.52 tok/s** → `branch = build_m3m4`
  (`m2-gate.json`). **Every 30B number carries `PROJECTED`.**
- **Superseded** (`m2-gate-NOTE.md`): M2's 2.62 GB/s sequential bandwidth and the
  25.52 projection are optimistic for the real scattered pattern M3 then measured;
  M4 replaced the projection with an on-box measurement entirely.

### M3 — expert-contiguous repack (MEASURED)

Two independent gates, both measured, both PASS (`artifacts/m3/evidence/M3-REPORT.md`).

- **Byte-identity:** 6144 expert blobs × 3 role-slices = **18432 slices, all
  byte-for-byte equal** (`m3-gate.json` `output_exact.all_equal=true`). Inference
  is provably unchanged **without running the model** — the layout moved, the
  content did not.
- **Throughput:** repacked **1.135 GB/s** median vs stock **1.021 GB/s**,
  verdict **`separated_win`** (repacked.min 1.121 ≥ stock.max 1.028 — bands do
  not overlap; parameter-free). Ratios reported honestly: qd16 1.11×, qd1 1.20×
  (the win did *not* amplify under the thread proxy). NVMe re-measured 1.24 GB/s,
  **replacing** the spec §3 constant. `branch = proceed_m4`.

### M4 — baseline spike (MEASURED) → the premise fails contact

The ADR-0001 gate #2 for the streaming loader. Verdict **`no_go` / `escalate` /
`model_fits_cache_moot`** (`artifacts/m4/evidence/M4-SPIKE-REPORT.md`,
`m4-gate.json`, `passed:false`):

- warm **4.278 tok/s**, **111 MB/token** physical disk (upper bound; conflates
  one-time load), disk-I/O ceiling **9.18 tok/s**, cold ≈ warm.
- Four converging lines → decode is **not** per-token expert-scatter-I/O-bound,
  so the M4 loader patch premise does not survive contact here. The C++ loader
  stayed **unbuilt** — the gate failed honestly, as ADR-0001 anticipated.

### Post-M4 probe — the correction (MEASURED, this session)

M4 proved decode is not I/O-bound *at 4.3*; it did not prove 4.3 was the compute
ceiling. The ADR-0002-gated probe (`artifacts/m4/evidence/PROBE-REPORT.md`,
`probe-verdict.json`, `probe-pivot-verdict.json`; commits `f837d2a`, `81fa2e9`):

- **4.3 was a per-invocation cold-load artifact** (fresh `llama-bench` re-faulting
  ~14 GB each run). **Warm, sustained, in-process decode ~9.2 tok/s** (diverse) /
  up to ~14 (degenerate). During warm decode the **drive is idle** (≤0.66 vs 1.02
  GB/s) and the **6 worker threads peg** → **compute / RAM-bandwidth bound**.
- **The rule-(b) guard did its job:** 9.2 ≥ 8 would, on the threshold alone, have
  falsely "revived" streaming; requiring *which ceiling binds* caught that it is
  compute/RAM, not disk.
- **Active-set pivot** (equal-length control isolates the confound):

  | workload | prompt toks | gen tok/s | MB/token |
  |--|--:|--:|--:|
  | short, narrow (diverse) | 60 | 9.2 | ~30 |
  | long, low-entropy (control) | 2700 | 3.9 | ~41 |
  | long, wide (stressor) | 2673 | 2.9 | ~75 |

  9.2→3.9 is long-context **attention compute** (dominant); the isolated 3.9→2.9
  is the **routing-width expert I/O** penalty (~26%, ~1.8×/token). Drive tops
  ~490 of 1020 MB/s — **never saturated**. → streaming **dormant, not dead, but
  weak here**.

## The corrected decode picture

| regime | tok/s | binder | evidence |
|--|--:|--|--|
| M1c local baseline (llama-cpp-python) | 6.02 | mixed | `m1c-run-a.json` |
| M4 per-invocation cold-load | 4.3 | reload I/O + compute | `m4-gate.json` |
| **warm sustained, narrow (server regime)** | **~9.2** | **compute / RAM-bw** | `probe-verdict.json` |
| warm degenerate (minimal active set) | ~14 | compute / RAM-bw | `probe-signals-t6r12.json` |
| warm wide, long context | 2.9 | attention compute + expert I/O | `probe-pivot-verdict.json` |

The number that belongs in any forward statement is **~9.2 tok/s warm-sustained,
compute/RAM-bound** — not 4.3, and not disk-I/O-bound. (`m4-gate-NOTE.md` carries
this breadcrumb next to the immutable M4 gate.)

## Streaming verdict

- **Closed as a primary optimization on this box+model.** Even a max-entropy wide
  workload leaves expert-fault I/O secondary to compute/RAM and never saturates
  the drive.
- **Validated but relocation-gated (Fork 1).** It binds where the *warm* active
  set ≫ RAM and expert I/O dominates attention — a larger MoE, or a
  higher-bandwidth/compute tier. That is the natural next streaming probe (needs a
  model download / different HW); deferred.
- **Not "streaming never works"** — the negative is box+model+workload-scoped.

## What is exact and valuable

- A **byte-identical MoE reference pipeline** for a 30B-A3B Q4 model (M1
  determinism + M3's 18432-slice identity proof), with routing traces and a
  measured lossless repack.
- A **rigorous, over-determined, and now self-corrected negative**: the discipline
  that overturned its *own* headline number (4.3→9.2) when a better measurement
  contradicted it is the contribution, not just the number.

## Open questions & limits

- **§11 open-Q1 (routing precision-stability, int8/BF16 → Q4): UNRESOLVED.** M2
  projections and M1's cross-path convergence do not retire it; M3's gates do not
  depend on it.
- All findings are **this box + this model + these workloads**. The bigger-MoE /
  relocation regime is untested (needs a download).
- M2's 30B tok/s figures are **PROJECTED**; the measured on-box numbers are M3's
  bandwidth and the probe's decode rates.

## Pointers

- Decisions: `docs/adr/0001-*.md` (M4 loader gated on spike), `0002-*.md` (probe
  pre-registration + gate), `0003-*.md` (probe result + direction).
- Per-milestone detail: `artifacts/m{1,2,3}/evidence/M{1,2,3}-REPORT.md`,
  `artifacts/m4/evidence/M4-SPIKE-REPORT.md` + `PROBE-REPORT.md`.
- Forward breadcrumbs: `artifacts/m2/evidence/m2-gate-NOTE.md`,
  `artifacts/m4/evidence/m4-gate-NOTE.md`.
