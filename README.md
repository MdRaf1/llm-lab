# LLM-LAB

Research toward running **Mixture-of-Experts models that do not fit in your fast memory**
(RAM or VRAM), with output **bit-identical to the same quantized checkpoint run fully
resident**, at the cost of some extra time.

> ## 📄 Start here: [**The bottleneck wasn't the disk**](docs/exact-moe-on-commodity-hardware.md)
>
> The full story in one readable document — what was built, what the measurements found,
> and why the central hypothesis was abandoned on evidence.

> **Status: concluded (2026-09-24).** Milestones M0–M4 are complete and measured. The
> streaming hypothesis this project was built to test was **disproved for this hardware**:
> decode here is bound by **compute and RAM bandwidth, not disk I/O**. The follow-on
> architecture bet ("PageCC") was **closed on prior art**. What remains — and is the useful
> output — is an exact, deterministic MoE reference pipeline and a rigorously scoped
> negative result. See [`docs/adr/0004-conclude-pagecc-pivot-to-commodity-benchmark.md`](docs/adr/0004-conclude-pagecc-pivot-to-commodity-benchmark.md).

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
`SIMULATION — NOT REAL INFERENCE. DO NOT QUOTE ITS NUMBERS.` header that names exactly what
each one fakes. They are kept deliberately as architectural sketches and as a record of what
went wrong. The old demo script, which existed to record a video of those numbers, has been
removed (recoverable from git history at commit `a248a03`).

**Why the original claim was impossible, not just unverified:** a 70B dense model at
Q4_K_M is ~40 GB. With ~10 GB of usable RAM, ~30 GB must cross the NVMe bus *per token*.
At this machine's measured 2.62 GB/s that is 11.5 s/token ≈ **0.087 tok/s**. Reaching
15 tok/s would require ~450 GB/s of storage bandwidth — about 170× this drive. No amount
of code changes that.

Everything below obeys the [reporting rules](#reporting-rules) adopted in response.

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

## Results

| Milestone | Goal | Outcome |
|---|---|---|
| **M0** | Truth-in-labeling | **Done** — fabricated claims removed, `sim_*` renames, demo deleted. |
| **M1** | Exact baseline + determinism proof + routing traces | **Done, measured.** Deterministic over 256 tokens on Qwen3-30B-A3B Q4_K_M; 2000 BF16 routing positions across 3 domains. |
| **M2** | Trace analysis → go/no-go gate | **Done, `PROJECTED`.** Locality is real and sublinear; the 30B raw trace was lost with the rented host, so every 30B figure is labelled a projection. Superseded by M3/M4 measurement. |
| **M3** | Expert-contiguous repack (layout only, output-exact) | **Done, measured.** 18,432 slices byte-for-byte identical; 1.135 vs 1.021 GB/s, non-overlapping bands. |
| **M4** | Tiered async exact loader | **`no_go` — premise disproved.** Decode is not per-token expert-scatter-I/O-bound here. The loader stayed **unbuilt** because its gate failed. |
| **Probe** | Is 4.3 tok/s the compute ceiling? | **Correction.** 4.3 was a per-invocation cold-load artifact; warm sustained decode is **~9.2 tok/s**, drive idle, threads pegged → **compute/RAM-bound**. |
| **M5–M7** | PageCC (page-sparse continued training) | **Closed on prior art.** Every component published independently; see the [novelty assessment](docs/research/pagecc-novelty-assessment.md). |

**The headline number for any forward statement is ~9.2 tok/s warm-sustained,
compute/RAM-bound** — not 4.3, and not disk-I/O-bound.

Full detail: [`docs/M1-M4-REPORT.md`](docs/M1-M4-REPORT.md) and the per-milestone reports
under `artifacts/m*/evidence/`.

### Prior art is not ours

Draft-guided expert prefetch is published in SP-MoE (arXiv 2510.10302), MoE-SpeQ
(arXiv 2511.14102), and Apple's SpecMD (arXiv 2602.03921). The PageCC components are
published in Memory Layers at Scale (2412.09764), Pre-gated MoE (2308.12066),
Engram (2601.07372), StickyMoE (2607.08780), and "Cacheable by Design?" (2608.18261).
This project claims **engineering rigour and measurement**, not novel research.

---

## What is actually in here

### Real and working

| Path | What it does |
|---|---|
| `moe/meta.py`, `moe/baseline.py`, `moe/trace.py`, `moe/oracle.py` | The M1 exact pipeline: architecture facts, deterministic runs, routing traces, token-ID oracle. |
| `core/kernels.py` | INT4 pack/unpack + numba GEMV. Correct signed 4-bit with per-row scales. |
| `core/layers.py` | `FusedInt4Linear`, a drop-in `nn.Linear` replacement over packed 4-bit weights. |
| `frontier/gguf_partitioner.py` | Parses a real GGUF and splits tensors into hot/cold streams with an offset manifest. |
| `models/quantizer.py`, `models/runner.py` | Converts a real HF model to fused INT4 and compares generations. |
| `profiler.py` | Measures this machine's memory and NVMe bandwidth. |
| `cli/chat.py --engine local-llama` | Real interactive generation via llama.cpp. |

### Simulations (labelled, not deleted)

`frontier/sim_stream_engine.py`, `frontier/sim_gguf_stream_engine.py`,
`frontier/sim_converter.py`, `core/sim_engine.py`. Each carries a header explaining exactly
what it fakes. The `llm-lab simulate` and `llm-lab frontier` subcommands are marked
`[SIMULATION]` in `--help`. **Do not quote any number these produce.**

### Real code with known defects

| Path | Defect |
|---|---|
| `core/speculative_real.py` | Drafts by calling the **full target model** K times with no KV cache — slower than plain decoding, not faster. |
| `core/speculative.py` | `verify_lossless_rejection()` subtracts a scalar draft probability from the whole target vector; the correct residual is elementwise `max(0, p(x) − q(x))`. Not currently lossless. |
| `core/sparse_router.py` | Predictor weights are random and never trained; its "sparsity %" is a config constant. Thresholded sparsity on SwiGLU/SiLU is lossy regardless. |
| `core/self_drafter.py` | `w_fuse` is random and never trained. |
| `core/kv_cache.py` | 4-bit KV quantization and window eviction are **lossy** (0.9942 cosine ≠ 1.0; eviction discards tokens). |

---

## Exactness, stated precisely

For the local target, exactness means: *bit-identical to `Qwen3-30B-A3B-Q4_K_M` under
llama.cpp greedy decoding, seed 0, 6 threads, on the recorded host*. It always names
checkpoint + quant + decode + seed + host, and never claims identity to full precision
or BF16.

**Two numeric paths, never asserted equal.** The GGUF Q4_K_M path (via `llama-cpp-python`)
gives speed/RSS/determinism/token IDs; the HF `transformers` path (with
`output_router_logits=True`) gives exact per-layer expert IDs for the routing trace. Whether
Q4 routing matches BF16 routing remains an **open question**, not an assumption.

**Oracle refuses mismatched fingerprints.** `moe compare` raises rather than comparing two
runs whose config fingerprints differ, so an "identical" verdict is only ever reported for
runs of the same measured configuration.

**Energy caveat.** SSD expert offloading costs roughly 4.9× the per-token energy of HBM and
3.1× versus CPU DRAM (arXiv 2508.06978). The tok/s figures are baselines, not an energy
endorsement.

**Artifact policy.** Raw traces, logits, and checkpoints are gitignored. Only compact
evidence is committed under `artifacts/*/evidence/`: JSON summaries, oracle/gate verdicts,
command transcripts, and `*.sha256` manifests binding each summary to the raw file it
describes.

---

## The `moe` commands

```bash
# Architecture facts from a checkpoint (HF config or GGUF), to JSON
llm-lab moe meta --backend llama-cpp --model-path models/m1/qwen3-gguf/Qwen3-30B-A3B-Q4_K_M.gguf --output meta.json

# Deterministic greedy decode to a run record (tiny needs no download)
llm-lab moe run --backend tiny --prompt-token-ids 1,3,4 --seed 0 --max-new-tokens 3 --output run.json

# Greedy decode recording per-token expert routing (HF path, needs a 40-char revision)
llm-lab moe trace --backend hf-bf16 --model-id Qwen/Qwen3-30B-A3B --revision <40-char-sha> \
  --prompt "..." --trace t.jsonl --logits l.pt --run-record r.json --summary s.json

# Compare two run records by token ID (raises if config fingerprints differ)
llm-lab moe compare run_a.json run_b.json --output oracle.json

# Replay expert-cache hits over one trace at a chosen capacity (observational)
llm-lab moe replay-cache trace.jsonl --logits l.pt --capacity-experts 512 --output cache.json
```

Output-writing commands refuse to overwrite an existing path unless `--force` is passed, so a
measurement is never silently clobbered.

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

Run the test suite:

```bash
python src/llm_lab/__main__.py test
```

## License

MIT — see [LICENSE](LICENSE).
