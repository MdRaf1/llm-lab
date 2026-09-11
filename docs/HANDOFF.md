# Session handoff — resume prompt

Paste the block below as the **first message** of a new Claude Code session in
`C:\Rafi\Projects\llm-lab`. It reconstructs the decisions, the constraints, and the
things that must not be re-litigated.

---

```
Read docs/superpowers/specs/2026-09-11-moe-exact-runtime-design.md first. It is the
approved plan and the memory of a long prior session. Then read this context before doing
anything:

WHAT THIS REPO CLAIMS VS WHAT IT DOES
- Do NOT trust README.md or DEMO_SCRIPT.md. Their headline numbers (70B at 15-19 tok/s,
  4.80 GB RAM, 108x, 76.4% sparsity, "100% lossless") do not come from running a language
  model. stream_engine.py and gguf_stream_engine.py simulate inference: the per-layer
  "compute" is `hidden + 0.01*np.tanh(hidden)`, tokens are `np.random.randint`, the 121 MB
  layer buffers stay all-zero, prefetch does no I/O, and 4.8 GB / 0.08 tok/s are hardcoded
  constants. sparse_router.py predictors are untrained `np.random.randn`. speculative_real.py
  is real but drafts with the FULL target model K times with no KV cache, so it is slower
  than baseline, not 2.7x faster.
- Genuinely real and worth keeping: core/kernels.py (correct INT4 pack/unpack + numba GEMV),
  core/layers.py, frontier/gguf_partitioner.py, models/runner.py,
  benchmarks/benchmark_speculative_real.py.

THE DECISION (do not reopen without new evidence)
- Dense 70B on 16 GB at interactive speed is PHYSICALLY EXCLUDED: ~30 GB/token over a
  2.62 GB/s NVMe = 11.5 s/token = 0.087 tok/s. 15 tok/s would need ~450 GB/s of storage.
- So: target MoE, not dense. Build an EXACT tiered MoE runtime for existing checkpoints
  first. PageCC (the new page-native architecture) is deferred to milestones M6/M7 and will
  be a continued-training evolution of a pretrained MoE, never a from-scratch pretrain.
- We are NOT first at draft-guided expert prefetch. SP-MoE (2510.10302), MoE-SpeQ
  (2511.14102), and Apple's SpecMD (2602.03921) all published it; SP-MoE Figure 2 already
  measures the union-growth sublinearity I once claimed was unpublished. Our contribution is
  engineering + honest measurement in the GPU-free, SSD-backed, low-RAM Windows niche
  (see open llama.cpp discussion #27149), not novel research.
- There is no 5-10x lossless compression of a Q4 checkpoint. Packed 4-bit payloads barely
  compress (arXiv 2508.19263); only BF16 exponents do (~30%, already taken by DFloat11 /
  Unweight / ZipServ).

HARDWARE
- AMD Ryzen 5 5600G (Cezanne, PCIe 3.0 ONLY), 16 GB DDR4-3200 (~28.93 GB/s),
  NVMe ~2.62 GB/s measured (near platform ceiling), Windows 11.
- Highest-leverage upgrade is 32 GB RAM (~$40-60). Do NOT buy an SSD - PCIe 3.0 caps it.
- The 16 GB box is the showcase / worst-case tier. A rented ~128 GB machine is needed to
  hold Qwen3-30B-A3B in BF16 for routing traces.

APPROVED NEXT STEP: Milestone 1 only
Target: Qwen3-30B-A3B, Q4_K_M GGUF. Build ONLY:
  src/llm_lab/moe/{meta,baseline,trace,oracle}.py + tests/test_moe_trace.py
Goal: a real generation, proven determinism, an exactness oracle that compares TOKEN IDS
(never .strip() text), and expert-routing traces. No custom streaming engine, no speedup
claim, no PageCC code, no GPU work. Section 8 of the spec has the full design, the JSONL
trace schema, the RunRecord shape, error handling, and the definition of done.
Recommended: validate the pipeline on a small MoE (OLMoE-1B-7B or Qwen1.5-MoE-A2.7B) that
fits 16 GB before the 18 GB download.

REPORTING RULES (non-negotiable, this repo violated all five)
1. No number anywhere unless its command and output are saved in the repo.
2. Baselines measured on the same machine/session/checkpoint. Never a literature constant.
3. Projections labeled PROJECTED, with the formula shown.
4. Output identity = token-ID equality, never text comparison.
5. "Lossless" always names checkpoint + quantization + decode mode + seed. Q4 is itself
   lossy vs FP16, so the only honest phrasing is "bit-identical to Qwen3-30B-A3B-Q4_K_M
   under greedy decoding with seed S".

TOOLING NOTES
- WebSearch is broken. Use the `tvly` CLI (Tavily). Write results with
  `--json --output file.json` and read with UTF-8 (`PYTHONIOENCODING=utf-8`,
  `io.open(path, encoding='utf-8')`) - rich crashes on the Windows legacy console and
  cp1252 breaks on paper text. `pdftotext -layout` works for PDFs.
- Long-running subagents hit gateway 524 timeouts. Keep subagent tasks short and check
  docs/research/raw/ for already-saved results before re-searching anything.

Start by invoking the writing-plans skill to turn Milestone 1 (spec section 8) into a
task-by-task implementation plan. Do not write implementation code until that plan is
approved.
```

---

## Also worth knowing in the new session

- `docs/research/raw/` holds the prior session's evidence: `novelty-research.md` (prior-art
  survey), `unweight.txt` and `backslash.txt` (full paper extractions), and the raw Tavily
  JSON results. Check there before re-running any search.
- **M0 (truth-in-labeling)** is unstarted and is the cheapest credibility win: rewrite
  `README.md` / `DEMO_SCRIPT.md` to claim only what runs, and rename the simulated engines
  to `sim_*`. Do it before publishing anything.
