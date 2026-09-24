# ADR-0003: Probe result — M4's 4.3 overturned, streaming dormant-but-weak here; conclude & write up M1–M4

## Status

**Accepted (2026-09-24), decided with the user.** Follows ADR-0002 (which
pre-registered this probe) and the probe commits `f837d2a`
(compute-headroom) and `81fa2e9` (active-set pivot). Executes ADR-0002 Decision 4
(streaming relocation decided *only* by the probe) and Decision 1's gate.

## Context

ADR-0002 gated the streaming direction on an on-box compute-headroom probe with a
**pre-registered** decision rule: (a) tuned warm decode ≥~8 → revive / ≤~6 →
compute wall / 6–8 → inconclusive; **and** (b) name the binding ceiling — disk
(~9.18), RAM-bandwidth (~29), or compute. The probe ran; two findings:

1. **M4's 4.3 tok/s is overturned.** It was a *per-invocation cold-load artifact*
   — each fresh `llama-bench` process re-faulted the ~14 GB working set. **Warm,
   sustained, in-process decode** (the regime a server actually runs) is **~9.2
   tok/s** under diverse routing (up to ~14 degenerate). During warm decode the
   drive stays idle (≤0.66 GB/s vs 1.02 ceiling) while the 6 worker threads peg →
   binder is **compute/RAM-bandwidth, not disk I/O**. M4's *direction* (not
   disk-I/O-bound here) is **confirmed and strengthened**; its *number* is corrected.

2. **The rule-(b) guard did its job.** Warm decode 9.2 ≥ 8 would, on the threshold
   *alone*, have falsely "revived" streaming. Rule (b)'s ceiling-attribution
   requirement caught that the binder is compute/RAM, not disk — and ADR-0002
   Consequences (b) says a non-disk binder closes streaming just as a compute wall
   does. The tok/s number alone would have mis-fired; the guard prevented it.

3. **Active-set pivot (option 2): streaming is DORMANT, not dead — but WEAK here.**
   Forcing a wide hot-expert set (high-entropy ~2673-tok prompt), controlled
   against an equal-length low-entropy prompt (identical repeated token → narrow
   routing, same attention/KV compute):

   | workload | prompt toks | gen tok/s | gen disk median | MB/token |
   |--|--:|--:|--:|--:|
   | short, narrow (diverse) | 60 | 9.2 | ~30–300 | ~30 |
   | long, low-entropy (control) | 2700 | 3.9 | ~160 MB/s | ~41 |
   | long, wide (stressor) | 2673 | 2.9 | ~218 MB/s | ~75 |

   The **9.2→3.9** drop is long-context **attention compute** (dominant, present
   even with narrow routing, not I/O); the isolated **3.9→2.9** is the
   **routing-width expert I/O** penalty at equal context length — **~26% slower**,
   ~1.8× per-token disk (75 vs 41 MB/token). The drive **never saturates**:
   random expert faults top **~490 of the 1020 MB/s** scatter ceiling in any
   regime. So widening the workload *does* re-introduce real per-token
   expert-fault I/O (dormant, not dead), but it is a **secondary** binder behind
   compute/RAM, and streaming's payoff on this box+model would be marginal.

## Decision

1. **Record the overturn.** Warm-steady sustained decode on this box+model is
   **~9.2 tok/s** (diverse) / workload-dependent 2.9–14; 4.3 was the cold
   per-invocation regime. The cold-vs-warm (per-invocation-reload vs
   sustained-in-process) distinction is now part of the record.
2. **Streaming on THIS box+model is CLOSED as a primary optimization** — dormant
   and weak: expert-fault I/O is present but secondary, the drive is never
   saturated, compute/RAM binds. Not "streaming never works."
3. **The streaming thesis is VALIDATED as real but RELOCATION-gated (Fork 1).** It
   binds where the *warm* active set ≫ RAM and expert-fault I/O dominates attention
   — a larger MoE, or a higher-bandwidth/compute tier. That is the natural next
   streaming probe (needs a model download / different HW); **deferred, not taken here.**
4. **Immediate next step = write up M1–M4 (Fork 3),** carrying the corrected ~9.2
   warm-steady / compute-RAM-bound finding, the cold-vs-warm regime distinction,
   and the pivot scope. Low-regret; banks the exact reference pipeline + the
   rigorous, now-refined negative.
5. **PageCC (Fork 2) remains deferred, gated on the prior-art/novelty check.** The
   finding (compute/RAM is the real bottleneck) still *motivates* it; it does not
   authorize its cost.

## Consequences

- The spec §3 founding premise (decode is memory/I/O-bound) is **workload- and
  hardware-conditional**: false for narrow/local workloads on this box (compute/RAM
  binds, ~9.2 tok/s), only true in the wide-active-set ≫ RAM / relocated regime.
- The streaming milestones (M4 loader, M5 sub-expert paging) are **re-scoped to the
  relocation target** (Fork 1), not this box+model; they stay unbuilt here.
- The M1–M3 pipeline remains exact and valuable; the write-up states the negative
  as *box+model+workload-scoped*, with the dormant (not dead) nuance and the
  bigger-MoE continuation explicit.
- **Pre-registration credit:** ADR-0002's rule (b) is what kept a 9.2 ≥ 8 from
  mis-reviving streaming. Keep ceiling-attribution (not tok/s alone) in any future
  gate.
- No spec §6/§7 edits until the write-up; this ADR is the direction source of truth
  until then.

## Evidence

- Probe: `artifacts/m4/evidence/PROBE-REPORT.md`, `probe-verdict.json`,
  `probe-pivot-verdict.json`; raw `probe-bench-t*.json`, `probe-signals/counters-t6r12`,
  `probe-diverse.*`, `probe-wideset.*`, `probe-lowent.*`. Commits `f837d2a`, `81fa2e9`.
- Prior decisions: `docs/adr/0001-*.md`, `docs/adr/0002-*.md`.
