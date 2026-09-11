# LLM-LAB

Research toward running **Mixture-of-Experts models that do not fit in your fast memory**
(RAM or VRAM), with output **bit-identical to the same quantized checkpoint run fully
resident**, at the cost of some extra time.

> **Status: pre-measurement.** Nothing in this repository currently performs verified
> low-memory inference. The plan is
> [`docs/superpowers/specs/2026-09-11-moe-exact-runtime-design.md`](docs/superpowers/specs/2026-09-11-moe-exact-runtime-design.md).
> Milestone 1 — a real baseline, a determinism proof, and expert-routing traces — is not
> yet implemented.

---

## Read this before anything else

An earlier version of this README claimed 70B inference on a 16 GB PC at 15–19 tok/s in a
4.8 GB envelope with "100% lossless" output and a "108× speedup". **Those numbers were not
produced by running a language model.** They came from simulation code that:

- allocated layer buffers and never filled them (they stayed all-zero),
- advanced the hidden state with `hidden + 0.01*tanh(hidden)` instead of a matmul,
- emitted `np.random.randint` tokens,
- read a 1024-byte slice of the weight file per layer and discarded it,
- and divided a measured numpy speed by a **hardcoded** `0.08 tok/s` "baseline" to get 108×.

Those modules are still in the tree, renamed `sim_*` and carrying a
`SIMULATION — NOT REAL INFERENCE` header. They are kept as architectural sketches and as
a record of what went wrong. The old demo script, which existed to record a video of those
numbers, has been removed (recoverable from git history at commit `a248a03`).

**Why the original claim was impossible, not just unverified:** a 70B dense model at
Q4_K_M is ~40 GB. With ~10 GB of usable RAM, ~30 GB must cross the NVMe bus *per token*.
At this machine's measured 2.62 GB/s that is 11.5 s/token ≈ **0.087 tok/s**. Reaching
15 tok/s would require ~450 GB/s of storage bandwidth — about 170× this drive. No amount
of code changes that.

---

## What is actually in here

### Real and working

| Path | What it does |
|---|---|
| `core/kernels.py` | INT4 pack/unpack + numba GEMV. Correct signed 4-bit with per-row scales. |
| `core/layers.py` | `FusedInt4Linear`, a drop-in `nn.Linear` replacement over packed 4-bit weights. |
| `frontier/gguf_partitioner.py` | Parses a real GGUF and splits tensors into hot/cold streams with an offset manifest. |
| `models/quantizer.py`, `models/runner.py` | Converts a real HF model to fused INT4 and compares generations. |
| `profiler.py` | Measures this machine's memory and NVMe bandwidth. |
| `benchmarks/benchmark_speculative_real.py` | Real llama.cpp generation with a prompt-lookup drafter. |
| `cli/chat.py --engine local-llama` | Real interactive generation via llama.cpp. |

### Simulations (labelled, not deleted)

`frontier/sim_stream_engine.py`, `frontier/sim_gguf_stream_engine.py`,
`frontier/sim_converter.py`, `core/sim_engine.py`. Each has a header explaining exactly
what it fakes. The `llm-lab simulate` and `llm-lab frontier` subcommands are marked
`[SIMULATION]` in `--help`.

### Real code with known defects

| Path | Defect |
|---|---|
| `core/speculative_real.py` | Drafts by calling the **full target model** K times with no KV cache — slower than plain decoding, not faster. |
| `core/speculative.py` | `verify_lossless_rejection()` subtracts a scalar draft probability from the whole target vector; the correct residual is elementwise `max(0, p(x) − q(x))`. Not currently lossless. |
| `core/sparse_router.py` | Predictor weights are random and never trained; its "sparsity %" is a config constant. Thresholded sparsity on SwiGLU/SiLU is lossy regardless. |
| `core/self_drafter.py` | `w_fuse` is random and never trained. |
| `core/kv_cache.py` | 4-bit KV quantization and window eviction are **lossy** (0.9942 cosine ≠ 1.0; eviction discards tokens). |

---

## The plan

Target: **MoE**, not dense. Only a few experts fire per token, the router is
deterministic, so fetching just the selected experts and demand-loading on a miss is
**bit-identical** — the model computes the same thing, just later.

| Milestone | Goal |
|---|---|
| M0 | Truth-in-labeling. **Done** — this README and the `sim_*` renames. |
| M1 | Real baseline on Qwen3-30B-A3B Q4_K_M + determinism proof + expert-routing traces. **Next.** |
| M2 | Trace analysis: cache-hit curves, union growth, reuse distance → go/no-go gate. |
| M3 | Expert-contiguous repack (layout only, output-exact). |
| M4 | Tiered async exact loader (VRAM/RAM/NVMe) with real queued I/O. |
| M5–M7 | Page-level analysis, then page-sparse continued training (the "PageCC" research bet). |

Prior art is **not** ours: draft-guided expert prefetch is published in
SP-MoE (arXiv 2510.10302), MoE-SpeQ (arXiv 2511.14102), and Apple's SpecMD
(arXiv 2602.03921). The gap we target is the GPU-free, SSD-backed, low-RAM **Windows**
configuration, which those systems do not cover — see open llama.cpp discussion
[#27149](https://github.com/ggml-org/llama.cpp/discussions/27149).

---

## Reporting rules

Adopted because the first iteration violated all five:

1. No number in any README, report, or commit message unless its command and output are
   saved in the repo.
2. Baselines are measured on the same machine, same session, same checkpoint — never a
   literature constant.
3. Projections are labelled `PROJECTED` and shown with their formula.
4. Output identity is **token-ID equality**, never a `.strip()` text comparison.
5. "Lossless" always names its reference. Q4 is itself lossy versus FP16, so the honest
   form is *"bit-identical to `Qwen3-30B-A3B-Q4_K_M` under greedy decoding with seed S"*.

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -e .
```

Measure your own hardware (this is real):

```bash
python -m llm_lab.profiler
```

Run real generation (needs a GGUF file on disk; edit the path in `cli/chat.py`):

```bash
python src/llm_lab/cli/chat.py --engine local-llama
```

Run the test suite:

```bash
python src/llm_lab/__main__.py test
```

## License

MIT — see [LICENSE](LICENSE).
