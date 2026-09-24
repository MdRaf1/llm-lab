# PageCC Prior-Art / Novelty Assessment

- **Date:** 2026-09-24
- **Scope:** Is there a genuinely novel, defensible research contribution left in the **PageCC** direction — a *small resident core + page-aligned compositional atoms shared across layers + a deterministic early router + a training loss penalizing distinct physical page bytes over a sliding token window* (spec §6) — or is it already published? Plus: does M1–M4's compute-bound finding open a **better** bet, and what is it?
- **Method:** Six parallel literature/patent reading passes (GitHub issues #4–#9 on `MdRaf1/llm-lab`), each reading the closest named prior art directly and building on the prior survey in `docs/research/raw/`. Reading + searching only — no compute, no training. Per-claim verdicts, each cited.
- **Honesty note:** a novelty KILL is a valid, valuable outcome (cf. M4's premise not surviving contact). Verdicts below are deliberately skeptical; residual uncertainty is stated, not hidden. One session over public web + Google Patents is **shallower than a professional FTO search** — "clean" means nothing surfaced, not that nothing exists.
- **Ground truth:** HEAD `7209d05`, tree clean, `tests/test_moe_trace.py` green (2026-09-24).

---

## TL;DR

**PageCC as framed is comprehensively pre-empted.** Each of its three named mechanisms is independently published; only the *exact conjunction* is technically unclaimed, and that sliver is weak because it optimizes the wrong bottleneck (memory locality) — the one M1–M4 already showed is **not** the binder on the target hardware (compute/RAM is) — and its nearest twin is a **fresh, same-model negative result**. The general lever ("continue-train an MoE to cut active FLOPs/token at equal quality") is itself already published.

The honest high-leverage direction is **not** to out-invent the labs on architecture, but to exploit the project's real durable asset: **rigorous exact measurement on the commodity RAM-bound tier nobody else benchmarks.**

## Per-claim verdict

| PageCC claim | Verdict | Closest prior art |
|---|---|---|
| (1) Small resident core + page-aligned atoms **shared across layers** | **PUBLISHED (kill)** | Memory Layers at Scale (2412.09764); Relaxed Recursive Transformers (2410.20672); PEER (2407.04153); MoRE (2609.18176) |
| (2) **Deterministic early** router | **PUBLISHED (kill)** | Pre-gated MoE (2308.12066); Engram (2601.07372, DeepSeek); Temporally Extended MoE (2604.20156) |
| (3) Training loss on **distinct physical page bytes over a sliding window** | **PUBLISHED (kill)** | StickyMoE (2607.08780); **Cacheable by Design? (2608.18261 — same model, negative)**; ReMoE (2605.27081 / see note); Cache-Conditional Experts (2412.00099) |
| (4) The **combination** of (1)+(2)+(3) as one trained architecture | **CLEAN, but weak** | No single work; see below |
| General lever: cut active FLOPs/token at equal quality via continued training | **PRE-EMPTED** | ZEDA (2605.18643); REAP (2510.13999); LExI (2509.02753); SEER-MoE (2404.05089) |

## Claim by claim

**(1) Cross-layer atom sharing + resident core — PUBLISHED.** *Memory Layers at Scale* (2412.09764) is a direct hit: a shared pool of memory parameters used across all (strided) memory layers, sitting on top of live dense FFN layers ("sparse and dense both needed and complementary") — precisely "small resident core + fine-grained atoms shared across layers, softly composed by top-k retrieval." *Relaxed Recursive Transformers* (2410.20672) establishes whole-block cross-layer sharing (one looped block + per-depth LoRA) as old. *PEER* (2407.04153) supplies the single-neuron-expert granularity (but per-layer, not cross-layer). *MatFormer* (2310.07707) nests within a layer, not across. The only thing none touch is the *page-aligned physical* qualifier — which is claim (3)/(4), not (1).

**(2) Deterministic early router — PUBLISHED.** "Deterministic" adds nothing: standard top-k gating is already deterministic given the hidden state. "Early" is published three ways: *Pre-gated MoE* (2308.12066) computes block N+1's experts at block N to decouple selection from execution and prefetch; *Engram* (2601.07372, DeepSeek) makes the retrieval index a deterministic hash of the input n-gram "computable before the layer executes" to prefetch over PCIe — deterministic + early + prefetch-motivated is PageCC's exact pitch, already shipped (caveat: Engram is a KV-memory module beside MoE, not the expert-FFN router); *Temporally Extended MoE* (2604.20156) holds a persistent expert mask across spans.

**(3) Page-locality training loss — PUBLISHED.** The core idea — a differentiable loss that shrinks the activated working set over nearby tokens to cut memory-hierarchy traffic — is published 3–4× over: *StickyMoE* (2607.08780) penalizes abrupt expert switches between adjacent tokens, motivated verbatim by slow-storage↔fast-memory swapping on edge devices; **_Cacheable by Design?_ (2608.18261)** trains routers with the same adjacent-token locality loss and **traces the exact target model (Qwen3-30B-A3B, 48L/128E/top-8) over prose/code/math/medical**, reporting a **pre-registered negative** (cuts cache misses up to 60% but fails a ≤1% perplexity gate at 137M multi-domain); *Cache-Conditional Experts* (2412.00099) is the inference-time ancestor. For a standard MoE, "distinct physical page bytes" ≈ "distinct experts" and "sliding window" is an obvious generalization of the published adjacent-token term.
> Note: ticket #6 flagged that `ReMoE (2412.14711)` is a ReLU-router training method, **not** sticky routing; the sticky/temporal-consistency paper is StickyMoE (2607.08780). The locality-regularizer ReMoE cited by ticket #7 is a distinct 2026 work (2605.27081). Both IDs are recorded here for the record; the load-bearing kill is 2608.18261.

**(4) The combination — CLEAN, but weak.** No 2026 paper or patent lands on the whole (1)+(2)+(3) conjunction, and the *literal page-byte training unit over cross-layer-shared atoms decoupled from the expert* is not explicitly done. But this sliver is thin: (a) it optimizes **memory locality**, and M1–M4/ADR-0003 measured the binder on this box+model to be **compute/RAM-bandwidth, not locality** — it targets a bottleneck the project already closed; (b) its nearest twin (2608.18261) is a **fresh same-model negative**; (c) it cannot be pitched on the general lever (below), only on the unproven mechanism. Nearest patents (US11972137B2, US12235778B2) *exploit* access locality in hardware; they don't *induce* it via training — a real gap, but a narrow, unproven one. Google's conditional-compute continuation US20260044710A1 is worth a look before any filing.

**General lever — PRE-EMPTED.** "Continue-train an MoE to cut active FLOPs/token at equal quality" is published, most directly by **ZEDA — Post-Trained MoE Can Skip Half Experts via Self-Distillation (2605.18643)**, which self-distills to deactivate ~50% of experts at equal quality; REAP (2510.13999), LExI (2509.02753), SEER-MoE (2404.05089) fill out a crowded field. The bare lever is not a novel contribution — a chosen direction must stand on mechanism, never on "fewer FLOPs at equal quality."

## Is there a better bet? (scout)

The training-time "less active compute/token at equal quality" frontier maps to seven clusters, most crowded:
- **Fine-grained + shared experts** (DeepSeekMoE 2401.06066): won and closed — the baseline.
- **Adaptive-k routing** (AdaMoE 2406.13233): real but ~10–20%, hurt by dynamic-shape batching friction.
- **Mixture-of-Depths** (2404.02258): crowded, stuck at the dynamic-shape wall that blocked production adoption.
- **Nested/elastic** (MatFormer 2310.07707, Flextron; shipping in Gemma 3n): device-adaptation, a different axis.
- **Recursion/looping** (Mixture-of-Recursions 2507.10524, Huginn 2502.05171, Tied Experts): the hottest, most on-point cluster — but crowding weekly, all small-scale/GPU-only.
- **Pruning/distillation** (SlimMoE 2506.18349, REAP): mature, reliable, low-novelty — but the only budget-feasible training-time lever without a GPU.

Blunt read: the adaptive-recursion/compute line hits the *actual* binder (compute), which PageCC does not — but it is crowded, GPU-heavy, and pushed weekly by large labs. Out-inventing them without training compute is unlikely. The project's durable, unique asset is its **exact measurement discipline on the commodity RAM-bound tier** — the box nobody else benchmarks.

## Recommendation (for the ADR-0004 direction call — to be made with the user, not here)

1. **Conclude PageCC as a novelty bet.** Its three mechanisms are each published; the surviving conjunction targets the wrong binder and has a fresh same-model negative. Continuing it would be defending novelty the evidence doesn't support — the exact thing this project refuses to do.
2. **Leading honest alternative — a measurement/referee contribution.** Take the labs' active-compute methods (Mixture-of-Recursions, adaptive-k, ZEDA-style expert-skipping, SlimMoE) and run them through the exact, deterministic, commodity-tier pipeline M1–M3 already built — producing the rigorous positives/negatives on 8–16 GB no-GPU hardware that the GPU-only papers never report. Low novelty-risk, plays to the built asset, directly serves "help people run these on weak hardware."
3. **Adaptive-compute architecture research** (a recursion/adaptive-depth line) — only if training compute (the available-later VPS/GPU) is committed; higher risk, crowded, but the one path that hits the real binder.

My honest lean: **(1) + (2)** — conclude PageCC, pivot the *ambition* into the measurement niche where the project is already the best in the world at exactly one thing. But this is your call to make in #11.

## Residual uncertainty / limits

- One session, literature + public patents only — no professional FTO; ~18-month patent-publication lag, embargoed camera-readys, CPC full-text and non-English filings unchecked. A paid patent search is warranted before any disclosure/filing.
- Several papers read via abstract/query-focused chunks, not full linear ingest (noted per ticket): Temporally Extended MoE appendix, Engram v2, PEER v1 only.
- The claim-(4) sliver is *unrefuted*, not *validated* — its mechanism differs from what 2608.18261 tested; the kill is a novelty/priority judgment, not a proof it cannot work.

## Sources

Full per-ticket reports with all citations: issues [#4](https://github.com/MdRaf1/llm-lab/issues/4), [#5](https://github.com/MdRaf1/llm-lab/issues/5), [#6](https://github.com/MdRaf1/llm-lab/issues/6), [#7](https://github.com/MdRaf1/llm-lab/issues/7), [#8](https://github.com/MdRaf1/llm-lab/issues/8), [#9](https://github.com/MdRaf1/llm-lab/issues/9) (all closed). Prior survey: `docs/research/raw/novelty-research.md`. Direction decision: issue #11 (ADR-0004, pending).

**Adjacent flag:** *Serving 35B MoEs from SSD with Trained Routing Prediction* overlaps llm-lab's own M2–M4 SSD-MoE runtime niche — relevant to positioning the exact-pipeline asset.


