# ADR-0006: Pre-registered gate — cross-vendor commodity benchmark v1 (perishable-first)

## Status

**Pre-registered 2026-09-26, before collection.** Follows ADR-0005 (Arc spike
PASS, which authorised *scoping* this — not open-ended work) and ADR-0004 (the
career-leverage success criterion). Same discipline as ADR-0002/0005: the file
decides the outcome, not the person mid-run.

## Why this exists, and the one constraint that shapes it

The Arc spike proved the hard leg runnable. The distinctive contribution is the
*quality-vs-active-compute tradeoff of a dynamic MoE method, measured exactly,
across commodity multi-vendor hardware nobody benchmarks* — CPU (5600G),
consumer NVIDIA (RTX 4060), Intel Arc (B580).

**The perishable resource is hardware access, not time.** The owner relocates in
Q4 2026; the three machines in one place cannot be recreated after the move. The
write-up (ADR-0004 step a) is already banked and public and travels anywhere.
Therefore: **collect raw data on all three machines BEFORE the move; defer all
analysis and write-up to after.** Data collection is the only move-gated work.

## Scope — deliberately minimal (v1)

- **One method:** active-expert reduction, `--override-kv <arch>.expert_used_count` at **{8 (baseline), 4, 2}**.
- **One model:** OLMoE-1B-7B-0924 Q4_K_M (fits all three machines; already validated in the spike).
- **Two numbers per cell:** (a) **quality** = perplexity via `llama-perplexity` on a **fixed, committed** corpus; (b) **speed** = decode tok/s, **median of ≥3 runs** (per the guardian-analytics lesson: never trust a single wall-clock).
- **Three machines × 3 expert settings = 9 cells**, each carrying the spike's proofs: device-placement (layers on GPU vs CPU) and `n_expert_used` from llama.cpp itself.
- **Backend fidelity** recorded as measured divergence vs the CPU reference, never as an identity claim.

**Out of scope for v1:** the 30B model; multiple models; quantization sweeps;
training; any method other than expert-count reduction. Those are v2 if v1 lands.

## Pass / partial / fail

- **PASS:** comparable perplexity + median tok/s collected for the {8,4,2} sweep on **all three** vendors before the move, each with device-placement + expert-count proof, on the fixed committed corpus.
- **ACCEPTABLE PARTIAL (not a failure):** the move cuts collection to **two** vendors. Write it up honestly as a two-vendor result — a clean partial is a portfolio piece (ADR-0004 reviewer).
- **FAIL:** no comparable, proof-carrying data collected on ≥2 vendors, or a metric that cannot be reproduced from a committed corpus + command.

## Per-machine time-box

Each machine gets **one day**. If a toolchain fights (RTX 4060 CUDA build, Arc
IPEX/SYCL, whatever), fall back to the **Vulkan** llama.cpp build that already
worked on the Arc — it runs on all three vendors and keeps the comparison
apples-to-apples. Do not spend a second machine-day on driver archaeology; record
the blocker and move to the next machine.

## Honesty rules (inherited)

Every number traces to a committed command + output on a committed corpus.
Perplexity is the only quality claim — no "the text looked fine". tok/s are
medians with the spread shown. "Exact" names checkpoint + quant + decode + seed +
host + backend + expert count.

## Consequences

- **On PASS/PARTIAL:** the data is banked and safe from the move; analysis + a
  results section (or short note) are written afterwards, from anywhere, and
  folded into the public write-up.
- **On FAIL:** the spike result already stands in the write-up as proven
  feasibility; nothing is lost, and the effort concludes there.

## Evidence (empty until collected)

- Per-machine raw + verdict under `artifacts/bench-v1/evidence/`
- Fixed corpus committed under `artifacts/bench-v1/` before any run
