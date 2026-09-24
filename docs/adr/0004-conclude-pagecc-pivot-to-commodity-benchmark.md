# ADR-0004: Conclude PageCC as a novelty bet; pivot to a standout commodity-tier measurement output

## Status

**Accepted (2026-09-24), decided with the user.** Executes the PageCC novelty gate that ADR-0003 (Fork 2) deferred. Evidence base: wayfinder map [#3](https://github.com/MdRaf1/llm-lab/issues/3) and its closed tickets #4–#13, plus the committed assessment `docs/research/pagecc-novelty-assessment.md` (commit `3671063`).

## Context

ADR-0003 left PageCC (M6/M7) deferred and **novelty-gated**. A wayfinder effort ran that gate as six parallel prior-art reads plus two bounded "ceiling-scans," and the user refined the success criterion mid-effort: optimize for a **standout, career-leverage public output** (a global remote job / study-abroad transfer), not field-scale novelty — given a fixed resource envelope (5600G CPU, RTX 4060 8GB, Intel Arc B580 12GB, Kaggle ~2×T4, ~$300 Vultr; ≤1B-param training + cross-vendor commodity measurement, not frontier scale).

Findings:

1. **PageCC is comprehensively pre-empted.** Each named mechanism is independently published — cross-layer atom sharing (Memory Layers at Scale, 2412.09764); deterministic early routing (Pre-gated MoE 2308.12066, Engram 2601.07372); the page-locality training loss (StickyMoE 2607.08780; **Cacheable by Design? 2608.18261 runs the near-twin on the same target model, Qwen3-30B-A3B, as a pre-registered negative**). Only the exact conjunction is unclaimed, and it targets *memory locality* — the bottleneck M1–M4/ADR-0003 already showed is **not** the binder (compute/RAM is). The general lever ("cut active FLOPs/token at equal quality via continued training") is itself pre-empted (ZEDA 2605.18643).

2. **Mechanism moonshot — NO-GO (~3–5%), structurally.** A novel active-compute mechanism's whole claim is *equal quality at less compute*, which can only be **staked** where quality is hard-won (at scale). The ≤1B tier this project can afford is exactly where MoR/MoRE (2609.18176)/AdaMoE already published — a run there can only reproduce, never distinguish. The one genuinely-open corner (static-shape adaptive compute) is open because it is a hard problem unsolved by far larger labs. A lottery ticket, not a fundable prototype.

3. **Cross-vendor commodity benchmark — QUALIFIED GO.** MoEXBench (2608.21693, three weeks old) planted a flag on MoE *compression* on H100/Apple M1, but left open: *dynamic* compute-cutting methods (ZEDA/adaptive-k/MoR) × the sub-$300 **tri-vendor commodity tier** (CPU / RTX 4060 / Intel Arc) × **exact-oracle backend-divergence** measurement. Odds ~50–60% as a strong portfolio artifact, ~15–20% as a cited standard. The distinctive part (a dynamic method on the **non-CUDA Intel Arc**) is also the hardest to pull off solo.

## Decision

1. **Conclude PageCC as a novelty research bet.** It is not pursued. This is a novelty KILL accepted on evidence, in the same discipline as M4's premise not surviving contact.
2. **Drop the mechanism moonshot.** ~3–5%, closed for a structural (not merely competitive) reason. Reversible only if a concrete, stakeable candidate surfaces later.
3. **Pivot to a standout public output, sequenced:**
   - **(a) Bank the floor first** — a public, polished write-up + repo of the M1–M4 exact reference pipeline, the honest streaming negative, and this novelty assessment (optionally a short arXiv/workshop note). The hard work exists; this is low-effort, high-signal for both remote-job and study-abroad goals, and it stands regardless of what follows.
   - **(b) Then the reach** — the cross-vendor, commodity-tier, exact-oracle benchmark of *dynamic* MoE compute-cutting methods across CPU / RTX 4060 / Intel Arc, **gated on a go/no-go spike of the hardest part first**: running one dynamic method (e.g. ZEDA) on the non-CUDA Arc (IPEX/SYCL). If the spike fails, stop with the floor already banked.

## Consequences

- The next work is a **fresh execution effort** (the write-up first), out of scope for the now-concluded wayfinder map #3.
- **Spec §6/§7 PageCC references are superseded** by this ADR; `docs/research/pagecc-novelty-assessment.md` is the evidence of record. No spec edits required beyond that pointer.
- The benchmark's viability hinges on the Arc-integration spike; treating it as the first gate keeps the effort low-regret.
- Success is now measured as a **career-leverage artifact** (visible, citable, reproducible), not field novelty. A paid FTO/patent search remains a prerequisite before any formal disclosure/filing (per #8).

## Evidence

- Wayfinder map [#3](https://github.com/MdRaf1/llm-lab/issues/3); closed tickets #4–#13; assessment `docs/research/pagecc-novelty-assessment.md` (`3671063`).
- Key citations: 2412.09764, 2308.12066, 2601.07372, 2607.08780, **2608.18261**, 2605.18643, 2609.18176, **2608.21693**.
- Prior decisions: `docs/adr/0001-*.md`, `0002-*.md`, `0003-*.md`.

