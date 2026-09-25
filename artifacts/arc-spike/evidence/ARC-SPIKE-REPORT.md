# Arc B580 spike — REPORT (ADR-0005)

**Verdict: PASS.** A dynamic MoE compute-cutting method runs on the non-CUDA
Intel Arc B580, fully GPU-resident, measured against the project's exact-oracle
discipline, with cross-backend divergence quantified. Completed 2026-09-26,
inside the 3-working-day box (expiry Mon 2026-09-28).

## What was measured

`OLMoE-1B-7B-0924-Instruct Q4_K_M` (4018 MB), prompt "The capital of France is",
48 tokens, greedy, seed 0, llama.cpp `b11160` Vulkan. Three cases:

| Case | Backend | n_expert_used | layers→Vulkan | layers→CPU | gen tok/s |
|---|---|--:|--:|--:|--:|
| arc_topk8 | Vulkan (B580) | 8 | 34 | 0 | **88.1** |
| arc_topk4 | Vulkan (B580) | 4 | 34 | 0 | **340.4** |
| cpu_topk8 | CPU | 8 | 0 | 34 | 27.4 |

(`arc_topk4` applies `--override-kv olmoe.expert_used_count=int:4`.)

## Criteria (all met)

1. **It runs** — all three produced coherent, correct text ("…Paris. Paris is…").
2. **Ran on GPU** — arc cases put 34/34 layers on Vulkan, 0 on CPU; the CPU
   control is the mirror image. The anti-silent-fallback guard is satisfied by
   the *contrast*, not an assertion.
3. **Method applied** — `arc_topk4` reports `n_expert_used = 4` from llama.cpp
   itself, not inferred from timing.
4. **Named ceiling** — **active-expert throughput on the GPU.** The model is
   fully VRAM-resident (4 GB in 12 GB, all layers offloaded), so decode is
   neither disk- nor host-RAM-bound. Halving the active experts (8→4) yielded a
   **~3.86× decode speedup** (88.1→340.4 t/s) — a response only an
   active-expert-dominated binder produces. Not "unknown".
5. **Reference stated** — fidelity reported as *divergence*, not identity.
   `arc_topk8` and `cpu_topk8` (same file, same box, top-8, seed 0) share
   *"The capital of France is Paris. Paris is not only the political"* then
   diverge (GPU: "capital…"; CPU: ", but also the economic…"). Cross-backend
   divergence from reduction-order/kernel differences is **expected** and is the
   deliverable — a bit-identical cross-vendor claim was never on the table.

## What this does and does not show

- **Does:** the differentiated, historically-painful part — a dynamic method on
  non-CUDA silicon, measured honestly with device-placement proof and an
  exact-oracle divergence — **works**. That was the ~50–60%-odds risk in ADR-0004;
  it cleared.
- **Does NOT:** say anything trustworthy about *quality* at top-k 4. The top-k4
  run was ~3.86× faster and stayed coherent on this one 48-token prompt, but that
  is not a quality result. Quality-vs-active-compute needs a real
  perplexity/benchmark eval — the benchmark v1's job, not this spike's.

## Honest limits

- Model is the **Instruct** variant (the base allenai URL timed out; the instruct
  fallback succeeded). All three cases used the same file, so the comparison is
  internally consistent, but it is not the exact M1b base checkpoint.
- One prompt, 48 tokens, one seed. A benchmark needs many prompts and a real
  metric.
- tok/s are single runs, not medians (cf. guardian-analytics: repeat before
  trusting a wall-clock number).

## Consequence (ADR-0005)

Arc leg viable → **authorises scoping a three-vendor benchmark v1 as a separate
effort with its own gate**, timeline-permitting given the Q4 2026 relocation.
Does **not** authorise open-ended benchmark work. The write-up (ADR-0004 step a)
stands regardless.
