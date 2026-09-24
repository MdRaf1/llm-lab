# The bottleneck wasn't the disk

*I spent months building an exact Mixture-of-Experts inference pipeline to prove that a cheap, GPU-less PC could run models far larger than its memory by streaming weights from disk. Careful measurement disproved the premise. Then a better measurement disproved my own headline number. Then a prior-art search closed the follow-on idea too. Here is the whole thing — every number traceable to a committed command and its output.*

---

## The method claim, stated up front

This project operates under five reporting rules, adopted because an earlier version of it violated all five:

1. **No number appears in any README, report, or commit message unless the command that produced it and that command's output are committed in this repository.**
2. Baselines are measured on the same machine, same session, same checkpoint — never a literature constant borrowed and presented as a measurement.
3. Projections are labelled `PROJECTED` and shown with their formula and inputs.
4. Output identity means **token-ID equality**, never a `.strip()` text comparison.
5. "Lossless" always names its reference checkpoint, quantization, decode mode, and seed.

Every claim below is checkable against those rules. That is deliberate: the rules are as much the contribution as the results are.

## What I was trying to do

Mixture-of-Experts (MoE) models are attractive for constrained hardware because only a few "experts" fire per token. In principle you can leave the rest on disk and fetch what each token needs. And because the router is deterministic, fetching only the selected experts and demand-loading on a miss computes **the same thing, just later** — bit-identical output, less RAM, some extra time.

The honest form of that claim, which I held to throughout:

> bit-identical to `Qwen3-30B-A3B-Q4_K_M` under greedy decoding, seed 0, on the recorded host — never "identical to full precision."

**The box:** AMD Ryzen 5 5600G (6 cores, **no discrete GPU**), ~16 GB RAM, PCIe-3 NVMe, Windows 11. The machine a lot of people actually own, and one almost no paper in this literature benchmarks.

## First, I deleted my own numbers

An earlier version of this repository advertised *"70B inference at 15–19 tok/s on a 16 GB PC, 108x speedup, 100% lossless."*

Those numbers were never produced by running a language model. They came from simulation code that allocated layer buffers and left them zero-filled, advanced the hidden state with `hidden + 0.01*tanh(hidden)` instead of a matmul, emitted random token IDs, read a 1024-byte slice of the weight file per layer and discarded it — and divided a measured numpy speed by a **hardcoded** `0.08 tok/s` "baseline" to manufacture the 108x.

I removed the claims, deleted the demo script that existed to record a video of them, and kept the code itself — renamed `sim_*`, each file carrying a header stating exactly what it fakes — as a record of what went wrong.

The original claim was **impossible, not merely unverified**: a 70B dense model at Q4_K_M is ~40 GB; with ~10 GB usable RAM, ~30 GB must cross the NVMe bus per token. At this drive's measured bandwidth that is ~0.087 tok/s. Reaching 15 tok/s would require roughly 170x this drive. No amount of code changes that.

## What I built, and proved

**An exact MoE reference pipeline**, gated in three phases so a schema bug never costs an 18 GB re-download:

| Phase | Model | Result |
|---|---|---|
| M1a | tiny random OLMoE (no download) | two independent builds produced **identical token IDs** |
| M1b | OLMoE-1B-7B | GGUF Q4_K_M ~36 tok/s, HF int8 ~2 tok/s — **each deterministic against itself** |
| M1c | Qwen3-30B-A3B (target) | Q4_K_M deterministic over 256 tokens at ~6 tok/s; 2,000 routing positions across prose/code/factual prompts |

**A provably lossless expert-contiguous repack.** Reordering the checkpoint so each expert's weights are contiguous: **18,432 slices, all byte-for-byte identical** before and after. That proves inference is unchanged *without running the model at all* — and it still measured a real throughput gain (1.135 vs 1.021 GB/s, non-overlapping bands).

**An oracle that refuses unsound comparisons.** It compares token-ID sequences, and if two runs' configuration fingerprints differ it **raises** rather than reporting a similarity score. An "identical" verdict is only ever emitted for runs of the same measured configuration.

I also never asserted that the two numeric paths (4-bit llama.cpp decode vs BF16 routing capture) select the same experts. They are different executions; whether Q4 routing matches BF16 routing is recorded as an **open question**, not quietly assumed.

## The measurement that killed the premise

The streaming thesis assumes decode is **disk-I/O-bound**. I built a baseline spike to test exactly that, with a pre-registered go/no-go gate.

It returned `no_go`: warm decode ~4.3 tok/s, with four independent signals converging on *not I/O-bound*. The C++ loader I had designed stayed **unbuilt**, because the gate that justified it failed. That was the right outcome — the gate existed precisely so enthusiasm couldn't override it.

**Then I distrusted my own number.** The spike proved decode wasn't I/O-bound *at 4.3 tok/s*; it did not prove 4.3 was the ceiling. A follow-up probe found that **4.3 was a per-invocation cold-load artifact** — each fresh benchmark process was re-faulting the ~14 GB working set from disk. Warm, sustained, in-process decode — the regime a real server runs in — is **~9.2 tok/s**, and during it the **drive sits idle** (at most 0.66 of 1.02 GB/s available) while all six worker threads peg.

The binder is **compute and RAM bandwidth**, not disk.

A detail I'm glad I built in: the probe's decision rule required naming *which ceiling binds*, not merely clearing a tok/s threshold. On the threshold alone, 9.2 >= 8 would have "revived" the streaming plan. Requiring ceiling-attribution caught that the binder was compute — the guard prevented a false positive in favour of my own preferred outcome.

**Then I scoped the negative honestly.** Using a controlled prompt pair (equal length, wide vs. narrow expert routing, isolating routing width from context length), widening the active expert set *does* re-introduce real per-token expert-fault I/O — ~26% slower, ~1.8x the per-token disk traffic. But it stays a **secondary** binder behind compute, and the drive never saturates (~490 of 1020 MB/s at worst).

So streaming here is **dormant, not dead**: it would bind on a larger MoE or a higher-bandwidth tier, where the warm active set greatly exceeds RAM. The negative is scoped to *this box, this model, these workloads* — not inflated into "streaming never works."

## Was the follow-on idea new? No.

The compute-bound finding sharpened the research target: if compute is the wall, change the *model* so it does less compute per token at equal quality. My candidate ("PageCC") combined a small resident core, page-aligned parameter atoms shared across layers, a deterministic early router, and a training loss keeping a token's active weights physically page-local.

I ran a structured prior-art check **before writing any training code** — six parallel investigations, each reading the closest published work directly. Every component came back published:

| Claim | Verdict | Closest prior art |
|---|---|---|
| Cross-layer shared atoms + resident core | **published** | *Memory Layers at Scale* (2412.09764); *Relaxed Recursive Transformers* (2410.20672) |
| Deterministic early router | **published** | *Pre-gated MoE* (2308.12066); *Conditional Memory via Scalable Lookup* — the **Engram** module (2601.07372) |
| Page-locality training loss | **published** | *Sticky Routing: Training MoE Models for Memory-Efficient Inference* — **StickyMoE** (2607.08780); *"Cacheable by Design?"* (2608.18261) |
| The general lever itself | **pre-empted** | *Post-Trained MoE Can Skip Half Experts via Self-Distillation* — **ZEDA** (2605.18643), among others |

(Where a short name appears above, it is the paper's own name for its method, not my shorthand.)

The decisive find was *"Cacheable by Design?"*: it trains essentially my proposed loss, on **my exact target model** (Qwen3-30B-A3B), and reports it as a **pre-registered negative** — locality training cuts cache misses substantially but fails a 1% perplexity gate.

Only the precise *conjunction* of all four elements was unclaimed — and that conjunction optimizes **memory locality**, the bottleneck I had just measured *isn't* the binder here. So I concluded it on evidence, the same way I concluded the streaming bet. A novelty kill is a valid result, and finding it cost a few days of reading instead of months of training.

## What this leaves, honestly

**What died:** the streaming thesis (scoped), and the PageCC architecture bet (closed on prior art).

**What survives and is worth something:** a bit-exact MoE oracle, a byte-identity layout proof, routing traces, and a measurement discipline that overturned its own headline number. Across the compute-cutting methods I surveyed for this assessment (expert-skipping, adaptive routing, recursion), evaluation is reported on datacenter-class GPUs; the closest published benchmark of MoE compression adds an Apple M1 Max. I did not find results for a commodity CPU, a consumer GPU, or a non-NVIDIA card — and that survey was one session of literature search, not an exhaustive one. How those methods' quality-versus-compute tradeoff actually behaves on such hardware, measured exactly against a bit-identical reference, is an open question, and the tooling here is built to answer it.

That is where the work points next.

## What this project demonstrates

- **Correctness discipline** — determinism proofs, a byte-identity proof, an oracle that refuses unsound comparisons, pre-registered decision gates.
- **Intellectual honesty under pressure** — I deleted fabricated claims I had inherited, overturned my own headline number when a better measurement contradicted it, and killed two of my own bets on evidence rather than defending them.
- **Systems depth** — GGUF internals, quantization, memory-hierarchy and bandwidth analysis, exact tiered-loading design, grounded in arithmetic rather than enthusiasm.

## This is the third time

This project is not a one-off. It is the third of three where the deliverable is *"I measured it, and the measurement disagreed with my design."*

- **[retrieval-eval](https://github.com/MdRaf1/the-data-guardian/tree/main/retrieval-eval)** — an offline retrieval evaluation harness: a 300-document corpus built from the public-domain NIST SP 800-53 Rev 5 OSCAL catalog, 36 graded queries **frozen before any retriever ran**, seven retrievers scored on nDCG@10 / Recall@10 / MRR@10. Its headline output is **three negative results**: title boosting came out *bit-identical* to baseline on the paraphrase subset; RRF hybrid fusion landed strictly between its two legs on every subset (0.571, between 0.476 and 0.623); cross-encoder re-ranking was net flat over dense retrieval (0.622 vs 0.623 nDCG@10) and *cost* recall (0.782 → 0.734). Nothing was deployed. The negatives are the point.

- **[guardian-analytics](https://github.com/MdRaf1/guardian-analytics)** — a normalised five-table PostgreSQL model of a compliance-audit domain (`data_source`, `document_ref`, `policy`, `scan_run`, `violation`), deployed and operated as a small production-style service. Three candidate indexes were built, measured, and **dropped, recorded with the numbers that disqualified them**. Two measurements there changed my mind: repeating a benchmark five times instead of once moved the honest comparison from 251.6 ms vs 178.6 ms to medians of 246 ms vs 178 ms — and the *faster* configuration produced the slowest single run in the set. And in a **deliberately induced** incident on the live database (synthetic data, no real users), dropping one composite index regressed a query 424 → 592 ms: the window sort lost its pre-sorted input and fell back to an external merge spilling **5,144 kB** to disk, diagnosed from `EXPLAIN (ANALYZE, BUFFERS)`. The trustworthy signal was the plan shape and temp-buffer counts, not the wall-clock figure — the post-restore run measured 252 ms on the same plan, just warmer.

- **llm-lab** (this project) — the core hypothesis measured and disproved, my own headline number corrected when a better measurement contradicted it, and a prior-art search that closed the follow-on idea on evidence.

Pre-registering the gate, then publishing the result that fails it, is the habit these three share. I think it is worth more than a demo that only ever works.

## Evidence

Every number above traces to a committed command and its output.

- **Cross-milestone synthesis:** [`docs/M1-M4-REPORT.md`](M1-M4-REPORT.md)
- **Per-milestone evidence:** `artifacts/m{1,2,3}/evidence/`, `artifacts/m4/evidence/` (spike + probe reports, gate verdicts, raw JSON)
- **Decisions:** `docs/adr/` — ADR-0001 through ADR-0004, each recording what was decided, on what evidence, and what it superseded
- **Prior-art / novelty assessment:** [`docs/research/pagecc-novelty-assessment.md`](research/pagecc-novelty-assessment.md)
- **Original design spec:** `docs/superpowers/specs/2026-09-11-moe-exact-runtime-design.md`

**Limits, stated plainly:** all performance findings are scoped to this box, this model, and these workloads. The 30B BF16 routing trace was captured on a rented host and the raw trace was lost when that host was destroyed — the M2 analysis built on it is labelled `PROJECTED` throughout. The novelty assessment is one session of literature and public-patent search, not a professional freedom-to-operate search; no formal novelty claim is made.
