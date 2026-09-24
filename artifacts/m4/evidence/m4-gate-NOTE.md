# m4-gate.json — forward note

`m4-gate.json` (and `M4-SPIKE-REPORT.md`) are the faithful, **byte-immutable**
record of what the M4 baseline spike measured with its harness. Its `warm_tok_s_median = 4.278`
and `spike_verdict = no_go` are the honest output of that method; **do not edit them.**
This sibling note is the forward breadcrumb, mirroring `m2-gate-NOTE.md`.

## What 4.3 tok/s is — and is not

M4's harness ran **fresh `llama-bench` processes** (`-r 1` per run), so **every**
run re-faulted the ~14 GB working set from disk. **4.278 tok/s is the
per-invocation cold-load regime**, not the decode ceiling.

The compute-headroom probe (`81fa2e9`, ADR-0002-gated) then measured **warm,
sustained, in-process** decode — the regime a server actually runs (model loaded
once, many tokens):

- **~9.2 tok/s** diverse routing / up to ~14 degenerate (`artifacts/m4/evidence/probe-verdict.json`).
- Binder is **compute / RAM-bandwidth, not disk I/O** — the drive stays idle
  (≤0.66 GB/s vs the 1.02 GB/s ceiling) while the worker threads peg.
- Active-set pivot (`probe-pivot-verdict.json`): streaming is **dormant, not dead,
  but weak here** — a wide workload adds ~26% routing-width I/O, the drive never saturates.

So M4's **direction** (decode is not disk-I/O-bound on this box) stands and is
strengthened; M4's **number** (4.3 as the decode rate) is corrected to ~9.2
warm-sustained. **No future read should take 4.3 tok/s as the decode ceiling.**
Decision recorded in `docs/adr/0003-probe-result-streaming-dormant-conclude-m1-m4.md`;
synthesis in `docs/M1-M4-REPORT.md`.
