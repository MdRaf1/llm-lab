# ADR-0005: Pre-registered gate — Intel Arc B580 dynamic-method spike (3 working days, hard)

## Status

**Pre-registered 2026-09-24 (Thursday), before any toolchain work.** Written in the
same discipline as ADR-0002, whose rule (b) is the reason this file exists: M4's probe
would have produced a *false pass* on a threshold alone, and was saved by requiring
that the binding ceiling be named. This gate repeats that requirement.

**This file decides the outcome, not the person running it.** On day 3 with something
half-working, the criteria below are authoritative.

## Why this spike exists

ADR-0004 concluded the research bets and chose a two-step pivot: (a) publish the
write-up — **done, `bb6de9c`, repository public**; (b) attempt a cross-vendor
commodity-tier benchmark, **gated on its hardest part first**.

The hardest part is Intel Arc. The differentiated claim of the whole benchmark is
measurement across CPU + consumer NVIDIA + **non-CUDA** silicon. The CPU and NVIDIA
legs are de-risked (existing work, standard toolchains). If the Arc leg cannot be made
to work, the benchmark reduces to a two-vendor CUDA-and-CPU comparison that overlaps
existing work — so this leg is tested *first*, cheaply, before any scaffolding is built.

## Time-box

| | |
|---|---|
| Start | 2026-09-24 (Thu) |
| Working days | **3, hard** — Thu 24, Fri 25, Mon 28 Sep 2026 |
| Expiry | end of 2026-09-28 (Mon) |
| Day-1 kill-switch | end of 2026-09-24 (Thu) — see below |

The box exists because IPEX/SYCL driver archaeology is a known time sink, and because
the owner relocates in Q4 2026; hardware access ends with the move. **No extension.**

## Scope — deliberately minimal

**One dynamic compute-cutting method, on one model, producing one honest number.
No benchmark scaffolding, no harness, no multi-method matrix.** Scaffolding is only
justified *after* this gate passes.

- **Hardware under test:** Intel Arc B580 (12 GB VRAM) in a host with **8 GB system RAM**.
- **Model:** `olmoe-1b-7b-0924-q4_k_m.gguf` (4.2 GB, 64 experts, top-8), already on disk.
  **Not** Qwen3-30B-A3B — at 18.5 GB it exceeds both the 12 GB VRAM and the 8 GB host
  RAM of that box. Recording this now so a mid-spike model substitution is not mistaken
  for a finding.
- **Method:** inference-time **expert-count reduction** (top-k 8 → 4) via llama.cpp
  `--override-kv` on the `expert_used_count` key. This is the runtime form of the
  compute-cutting lever the literature trains for (ZEDA, 2605.18643); it needs no
  training, which is what makes it testable inside 3 days.
- **Backend:** llama.cpp SYCL (oneAPI) preferred; **Vulkan is an acceptable substitute**
  and counts as cleared, provided the offload evidence below holds. The claim is
  "non-CUDA commodity silicon," not "SYCL specifically."

## What "honest" means here

A number is honest only if it names its reference and its divergence. Two distinct
questions, never conflated:

1. **Backend fidelity.** At *identical* configuration (same model, quant, seed, greedy,
   same top-k), does the Arc backend emit **token-identical** output to the established
   CPU reference for this checkpoint (M1b, `artifacts/m1/evidence/m1b-gguf-oracle.json`)?
   - Cross-vendor bit-identity is **not** expected to hold, and failing it is **not** a
     gate failure. Reduction order and kernel differences across vendors are known to
     move results. The deliverable is the **measured divergence** — first-divergence
     token index over a fixed prompt and seed — not a claim of identity.
   - Reported via the existing oracle semantics: token-ID equality, first-divergence
     index, never a similarity score, and never `.strip()` text comparison.
2. **Method effect.** At top-k 8 vs top-k 4 *on the same backend*, what is the measured
   change in tokens/sec, and what is the output divergence from the top-k 8 run on that
   same backend?

Both must carry the full configuration fingerprint (backend, driver/runtime version,
model, quant, seed, threads, top-k, host) per the repository's five reporting rules.

## Pass criteria — ALL must hold

1. **It runs.** At least one greedy generation completes on the B580 with the MoE model.
2. **It actually ran on the GPU** — positive evidence, not absence of an error.
   Required: non-zero layer offload reported by llama.cpp **and** corroborating
   device-side activity (GPU utilisation / VRAM occupancy) during generation.
   *This is the rule-(b) analogue.* llama.cpp silently falling back to CPU for MoE
   expert FFN ops is the most likely false pass in this whole spike: the run would
   "work," produce a plausible number, and be measuring the CPU. **A run that cannot
   demonstrate the expert FFN executed on the Arc device is a FAIL, regardless of its
   tok/s.**
3. **The method applied.** top-k 4 demonstrably took effect (verified from llama.cpp's
   own reported `expert_used_count`, not inferred from the timing being different).
4. **Named ceiling.** The report must state *what bound the top-k 8 run* — GPU compute,
   VRAM bandwidth, host↔device transfer, or CPU fallback — with the evidence for that
   attribution. **Naming "unknown" is permitted and is an honest FAIL, not a pass.**
5. **Reference stated.** Fidelity reported per the section above: either token-identical
   to the CPU reference, or a measured first-divergence index. "It produced reasonable
   text" is **not** a result.

**Explicitly NOT a pass criterion:** that top-k 4 is *faster*. A measured null or
negative speed result with a correctly named ceiling is a **PASS of this gate** — it is
exactly the kind of finding this project exists to produce. The gate tests whether the
Arc leg can be *measured honestly*, not whether the method helps.

## Fail criteria — ANY triggers stop

1. **Day-1 kill-switch (end of Thu 2026-09-24):** if **no GPU workload of any kind** has
   run on the B580 by end of day 1 — not this method, not this model, *any* GPU
   workload — stop. That is strong evidence the toolchain, not the experiment, is the
   wall. Days 2–3 are explicitly **not** authorised for driver archaeology.
2. **Expiry (end of Mon 2026-09-28)** with any pass criterion unmet.
3. Criterion 2 unsatisfiable — i.e. it runs, but expert execution on the device cannot
   be demonstrated.
4. The model does not fit, or requires a substitution not pre-registered above.

## Consequences — decided in advance, either way

**On PASS:** the Arc leg is viable. Proceed to scope a three-vendor benchmark v1 — as a
*separate* effort with its own gate, and only if the relocation timeline allows. Passing
this gate does **not** authorise open-ended benchmark work.

**On FAIL (either kill-switch or expiry):** stop immediately. Do **not** extend, and do
not attempt the NVIDIA leg as consolation within this box. Write up what was measured
and **scope it honestly to two vendors** (CPU + whatever else was actually measured),
stating plainly that the Arc leg was attempted, time-boxed, and did not clear — with the
failure mode recorded. Per ADR-0004's reviewer: *a clean partial measurement is a
portfolio piece; an abandoned three-vendor benchmark is not.*

In both cases the publication from step (a) stands and is unaffected. This spike
carries **no risk to the banked result** — that is why it was sequenced second.

## Evidence (to be filled on completion — empty until measured)

- Verdict: `artifacts/arc-spike/evidence/arc-gate.json`
- Report: `artifacts/arc-spike/evidence/ARC-SPIKE-REPORT.md`
- Raw: command transcripts + llama.cpp output under `artifacts/arc-spike/evidence/`
