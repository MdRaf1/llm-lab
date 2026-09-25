# ADR-0006: Pre-registered gate — cross-vendor commodity benchmark v1

## Status

**Pre-registered 2026-09-25, before collection. Revised the same day** to remove
a time/relocation gate. The first draft assumed the owner would lose access to
some of the three machines after the Q4 2026 move. **That assumption was wrong:
the owner has permanent access to all three machines regardless of the move.**
There is therefore **no time pressure** on this work — it proceeds at a
sustainable pace, correctness over speed. Follows ADR-0005 (Arc spike PASS,
which authorised *scoping* this) and ADR-0004.

## What this measures

The quality-vs-active-compute tradeoff of a dynamic MoE method, measured exactly,
across commodity multi-vendor hardware nobody benchmarks: CPU (5600G), consumer
NVIDIA (RTX 4060), Intel Arc (B580).

## Scope — deliberately minimal (v1)

- **One method:** active-expert reduction, `--override-kv <arch>.expert_used_count` at **{8 (baseline), 4, 2}**.
- **One model:** OLMoE-1B-7B-0924 Q4_K_M (fits all three machines; validated in the spike).
- **Two numbers per cell:** (a) **quality** = perplexity via `llama-perplexity` on a **fixed, committed** corpus; (b) **speed** = decode tok/s, **median of ≥3 runs** (guardian-analytics lesson: never trust a single wall-clock).
- **Three machines × 3 expert settings = 9 cells**, each carrying the spike's proofs: device-placement (layers on GPU vs CPU) and `n_expert_used` from llama.cpp itself.
- **Backend fidelity** recorded as measured divergence vs the CPU reference, never as an identity claim.

**Out of scope for v1:** the 30B model; multiple models; quantization sweeps;
training; any method other than expert-count reduction. Those are v2 if v1 lands.

## Pass / fail

- **PASS:** comparable perplexity + median tok/s for the {8,4,2} sweep on **all three** vendors, each with device-placement + expert-count proof, on the committed corpus.
- **Acceptable reduced result:** if *one vendor's toolchain proves genuinely intractable* after a bounded effort (below), a two-vendor result is a legitimate outcome — a **technical** limit, not a time limit. Record the blocker.
- **FAIL:** no comparable, proof-carrying data on ≥2 vendors, or a metric that cannot be reproduced from a committed corpus + command.

## Bounded effort per machine (engineering hygiene, not a deadline)

If a machine's toolchain fights (RTX 4060 CUDA build, Arc IPEX/SYCL), fall back to
the **Vulkan** llama.cpp build that already worked on the Arc — it runs on all
three vendors and keeps the comparison apples-to-apples. Don't sink open-ended
effort into driver archaeology; record the blocker and use the fallback. This is
about not *wasting* effort, not about running out of time.

## Honesty rules (inherited)

Every number traces to a committed command + output on a committed corpus.
Perplexity is the only quality claim — no "the text looked fine". tok/s are
medians with the spread shown. "Exact" names checkpoint + quant + decode + seed +
host + backend + expert count.

## Consequences

- **On PASS / reduced result:** analysis + a results section are written and
  folded into the public write-up, at whatever pace suits.
- **On FAIL:** the spike result already stands in the write-up as proven
  feasibility; nothing is lost.

## Evidence (empty until collected)

- Per-machine raw + verdict under `artifacts/bench-v1/evidence/`
- Fixed corpus committed under `artifacts/bench-v1/` before any run
