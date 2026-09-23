# ADR-0002: Direction after the M4 no-go — probe compute headroom on-box, then conclude and write up M1–M4

## Status

Accepted (2026-09-23). Follows ADR-0001 and the M4 baseline-spike no-go (`28ff9c4`). Sets project direction after M4's premise failed contact.

## Context

M4's baseline spike returned `no_go` / `escalate` / `model_fits_cache_moot` (`artifacts/m4/evidence/m4-gate.json`). Re-derived from the raw runs (`artifacts/m4/evidence/baseline-runs.jsonl`), the finding is **over-determined** — four independent lines converge:

1. **cold ≈ warm** — cold decode median 4.83 tok/s ≥ warm 4.28. If per-token faulting gated throughput, uncached cold would crater; it does not. (The clean kill.)
2. **miss/token 4.6× below the streaming anchor** — warm 111 MB/token vs the ~510 MB/token streaming-regime line pre-registered in spec §7:240; and 111 MB is itself an upper bound (`M4-SPIKE-REPORT.md:117`, it conflates one-time load faults with steady-state miss).
3. **2.15× under its own I/O ceiling** — 4.28 measured vs 9.18 tok/s ceiling (scatter 1.02 GB/s ÷ 111 MB, `m3-gate.json` throughput.stock_bps.median). I/O has headroom; a lower ceiling binds.
4. **bandwidth impossibility** — reaching 510 MB/token over the ~31 s decode window needs ~2.1 GB/s sustained; the drive delivers 1.02 GB/s. Not a measurement-noise call.

**Verdict: decode of `Qwen3-30B-A3B-Q4_K_M` on this box (Ryzen 5 5600G, 6C/12T, ~16 GB, PCIe3 NVMe) is CPU-compute-bound at ~4.3 tok/s, not expert-scatter-I/O-bound.** The spec §3:83 founding premise ("decode reads every active param once → memory-bound") is exactly what failed contact: ~3B active params at Q4_K_M cache to ~111 MB/token, well inside what the page cache and drive absorb. M2's 25.5 tok/s projection was a two-hop estimate (int8-OLMoE → Q4-30B, sequential I/O) and is superseded (`m2-gate-NOTE.md`).

**Consequence for the roadmap:** the streaming-optimization line — M3 expert-contiguous repack → M4 tiered async loader → M5 sub-expert paging — cannot yield an end-to-end decode speedup on this box + model, because you cannot speed a compute-bound workload by making its I/O faster.

**Two scoping guardrails, both carried:**

- The negative is **box + model specific**, not "streaming never works." A larger MoE whose active working set exceeds RAM even with reuse, or a box with more compute, can put I/O back on the critical path.
- M1–M3 are **exact and valuable**, not a failure: a byte-identical MoE reference pipeline (`m3-gate.json` output_exact.all_equal=true, 18432 slices) plus a real, over-determined negative result.

**The refinement that scopes the finding (and drives the decision):** the four lines prove "not I/O-bound *at* the measured ~4.3 tok/s" — they do **not** prove 4.3 is the irreducible **compute** ceiling. The 4.28→9.18 gap (2.15×) is the room a compute tune could reclaim. Rough arithmetic (~6 GFLOP/token vs 6 Zen3 cores' tens of GFLOP/s; ~1 GB/token active-weight reads vs ~28.9 GB/s RAM, spec §3) puts plausible compute/RAM ceilings well above 4.3 — so 4.3 looks like *this llama.cpp build/config/threading's* rate, not a hard wall. This matters decisively: if tuned compute pushes this box up toward ~9 tok/s it crosses **into** I/O-bound and the streaming thesis **revives on this very box**; if it plateaus below 9.18, streaming is genuinely dead here.

### Forks considered

| Fork | Cost | Risk | Hardware | Next step + gate |
|---|---|---|---|---|
| **1. Re-target streaming to where I/O binds** | Probe ≈ $0 / 1 hr; relocation = rented/new HW | Low (probe) → Med (relocation: regime exists, use-case must justify it) | Probe: none; relocation: bigger MoE / more cores / GPU-VRAM tier | The compute-headroom probe (below); relocate only if it confirms a compute wall here |
| **2. Pivot to PageCC (M6/M7)** | High (continued-training compute, long horizon) | High — novelty ~55–65% pre-search (§6:209), close prior art incl. Temporally Extended MoE 2604.20156 | Rented GPUs | Prior-art/novelty check first; then research design + kill-criteria; M6 gate = beats MoE/PEER at equal active FLOPs, ≤1% regression |
| **3. Conclude & write up M1–M4** | Low (writing) | Low | None | Write the report; gate = M0 honesty (no unmeasured claims, negative scoped) |

The forks are **not mutually exclusive**: Fork 3 precedes/accompanies 1 and 2 (§5:195 already positions M1–M4 as "engineering contribution + rigorous measurement, not novel research"). M4's finding does not kill PageCC — it *motivates* it (PageCC cuts active compute, which M4 says is the real bottleneck).

## Decision

1. **Immediate next step = the compute-headroom probe on this box** (Fork 1's cheap first move; ~1 hr, no VPS). Sweep thread count (6 physical / 12 logical / fewer), review the llama.cpp CPU build flags and batch (`-b` / ubatch) / config, and measure warm-steady tok/s scaling — reporting *which ceiling binds after tuning*, not merely a tok/s number. Its outcome is read against the pre-registered decision rule in Consequences (thresholds and ceiling attribution fixed now, not interpreted after the fact).
2. **Write up M1–M4 regardless** (Fork 3): the exact MoE reference pipeline + the "compute-bound, not I/O-bound, on a low-RAM CPU" finding, with the negative scoped box+model-specific. The probe's number folds in as the definitive on-box compute ceiling. Low-regret; banks the contribution whatever the probe says.
3. **PageCC (M6/M7) is the deferred research direction** (Fork 2), gated behind a prior-art/novelty check — especially Temporally Extended MoE (2604.20156) — before any rented training compute is committed. Not started here.
4. **Streaming relocation** (larger MoE / higher-compute box / GPU-VRAM tier) is decided **only by the probe result**, not now.
5. **Build nothing this session.** The probe and the write-up are each their own later sessions; this ADR records the direction, it does not execute it.

## Consequences

- **Probe decision rule — (a) pre-registered crossover thresholds (fixed now):** tuned warm-steady decode **≥ ~8 tok/s** (approaching the 9.18 disk-I/O ceiling) → the box has crossed **into I/O-bound**, streaming **revives here** and the M4 loader premise is restored; **≤ ~6 tok/s** and still ~2× under the ceiling → the **compute wall is confirmed** here → streaming needs a new home (Fork 1 relocate) or is dropped. The 6–8 tok/s band is explicitly **inconclusive** → re-probe / attribute the binding ceiling before deciding, never forced either way.
- **Probe decision rule — (b) report which ceiling binds:** the probe attributes the post-tuning limit to one of three — **disk I/O (~9.18 tok/s)**, **RAM bandwidth (~28.9 GB/s ÷ ~1 GB/token active reads ≈ ~29 tok/s)**, or **compute** — not just a tok/s figure. If tuning hits the **RAM-bandwidth wall** rather than disk or compute, that is a third regime neither M4 nor M2 modeled and the probe must surface it explicitly: RAM-bound decode is not fixable by a streaming loader either, so it closes the streaming line here just as a compute wall does.
- The M4 llama.cpp expert-loader C++ patch stays unbuilt — ADR-0001 gate #2 failed honestly, as ADR-0001 anticipated (`m4-gate.json` passed:false, branch:escalate).
- M5 (sub-expert page paging) is moot on this box for the same reason as the loader: it optimizes I/O against a compute-bound decode. Its fate rides on the probe with the rest of the streaming line.
- The streaming line is **paused pending the probe**, not killed: the probe can revive it here, relocate it, or close it "on this box."
- M6/M7 (PageCC) remain live but un-started, gated on novelty; the M4 finding strengthens their motivation without authorizing their cost.
- No spec §6/§7 edits are made until the roadmap change is confirmed; this ADR is the source of truth for direction until then.

## Evidence

- M4 gate + raw + report: `artifacts/m4/evidence/m4-gate.json`, `artifacts/m4/evidence/baseline-runs.jsonl`, `artifacts/m4/evidence/M4-SPIKE-REPORT.md`.
- Scatter bandwidth (ceiling denominator): `artifacts/m3/evidence/m3-gate.json` → throughput.stock_bps.median (1.021 GB/s); reused per `artifacts/m4/evidence/scatter-bandwidth.txt`.
- M2 superseded projection: `artifacts/m2/evidence/m2-gate.json` + `artifacts/m2/evidence/m2-gate-NOTE.md`.
- Physical limits + strategy + milestone gate: spec §3:83, §6, §7:240 (`docs/superpowers/specs/2026-09-11-moe-exact-runtime-design.md`).
- Prior decision: `docs/adr/0001-m4-forward-pass-patch-llamacpp-gated-on-spike.md`.
