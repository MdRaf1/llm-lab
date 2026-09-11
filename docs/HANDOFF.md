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
- M0 (truth-in-labeling) is DONE. README.md now states plainly that the old headline
  numbers (70B at 15-19 tok/s, 4.80 GB RAM, 108x, 76.4% sparsity, "100% lossless") were
  not produced by running a language model. DEMO_SCRIPT.md was removed (recoverable at
  commit a248a03). The simulated modules are renamed sim_* and each carries a
  "SIMULATION - NOT REAL INFERENCE" header: frontier/sim_stream_engine.py,
  frontier/sim_gguf_stream_engine.py, frontier/sim_converter.py, core/sim_engine.py.
  The canned-text chat path is deleted; sim engines now raise RuntimeError instead of
  printing fake answers with fake telemetry.
- Modules with WARNING headers (real code, known defects): core/speculative_real.py
  (drafts with the full target model, no KV cache -> slower than baseline),
  core/speculative.py (residual subtracts a scalar, so not actually lossless),
  core/sparse_router.py (untrained random predictor; "sparsity %" is a config constant),
  core/self_drafter.py (untrained).
- Genuinely real: core/kernels.py, core/layers.py, frontier/gguf_partitioner.py,
  models/runner.py, profiler.py, benchmarks/benchmark_speculative_real.py,
  cli/chat.py --engine local-llama.

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
Build ONLY: src/llm_lab/moe/{meta,baseline,trace,oracle}.py + tests/test_moe_trace.py
M1 is THREE phases and the order is normative (spec section 8.2). The tiering is forced by
measured hardware: 11.9 GB total RAM (~4 GB free under load), 28.3 GB free disk.
  M1a - tiny RANDOM-weight MoE, no download (e.g. OlmoeConfig(num_hidden_layers=2,
        hidden_size=64, num_experts=8)). Output is gibberish; this validates the trace
        schema, RunRecord, oracle, determinism, and the test. Runs in seconds.
  M1b - OLMoE-1B-7B for a real trace. NOTE: it is 13.8 GB at BF16 and DOES NOT FIT this
        machine - load it 8-bit (~7 GB) and record the precision in the trace header, or
        run BF16 on the rented box. Do NOT substitute Qwen1.5-MoE-A2.7B; at 14.3B params
        it is LARGER (28.6 GB BF16), not smaller.
  M1c - Qwen3-30B-A3B Q4_K_M (~18 GB) for the GGUF baseline. Its BF16 routing trace is
        ~61 GB and requires the rented machine.
ALREADY VERIFIED (2026-09-11, do not re-investigate): transformers 5.16.1 exposes
output_router_logits on OlmoeConfig and Qwen3MoeConfig, and MoeCausalLMOutputWithPast has
a router_logits field. Qwen3MoeConfig defaults confirm 128 experts / top-8.
Goal: a real generation, proven determinism, an exactness oracle that compares TOKEN IDS
(never .strip() text), and expert-routing traces. No custom streaming engine, no speedup
claim, no PageCC code, no GPU work. Spec section 8 has the JSONL trace schema, the
RunRecord shape, error handling, and the full definition of done.

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
- Bare `python` is not on PATH; use `./.venv/Scripts/python.exe`.
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
- Commits so far: `a248a03` (original, contains the fabricated claims and the deleted demo
  script), `58d1171` (the design spec + handoff), and the M0 truth-in-labeling commit.
