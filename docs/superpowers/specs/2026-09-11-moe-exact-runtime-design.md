# Design: Exact Tiered MoE Runtime — and the PageCC research path

- **Date:** 2026-09-11
- **Status:** Approved for Milestone 1 only. M2+ are gated on measured results.
- **Repo:** `C:\Rafi\Projects\llm-lab` (branch `master`)
- **Approved M1 target:** Qwen3-30B-A3B, Q4_K_M GGUF

---

## 1. Goal

Let people run an MoE model that does **not fit** in their fast memory (RAM or VRAM),
with output **bit-identical to the same quantized checkpoint run fully resident**, at
the cost of some extra time — across tiers from ~4 GB RAM laptops up to large GPUs.

Two claims we are allowed to chase:

| Claim | Status |
|---|---|
| Same checkpoint, less fast memory, identical tokens | Achievable for MoE. This is the project. |
| Same checkpoint, less fast memory, identical tokens, *no* slowdown | **Not** promised. Physics decides per model/device. |

---

## 2. What the existing repo actually does (audit, 2026-09-10)

**This must not be forgotten.** The README's headline numbers do not come from running a
language model. Do not trust `README.md` or `DEMO_SCRIPT.md` until M0 rewrites them.

### Simulated (produces the README's flagship numbers)

- **`src/llm_lab/frontier/stream_engine.py`** (`Frontier70BEngine`, the "70B @ 15–19 tok/s, 108×" table)
  - 80 layer buffers are 121 MB of zeros and stay zeros — verified: `buffer is all zeros? True`.
  - `async_prefetch_slice()` does **no I/O**; its worker computes a byte count and immediately
    sets the ready event. Its own comment says `# Simulate high-speed NVMe/RAM block transfer`.
  - Per-layer "compute" is `curr_hidden = curr_hidden + 0.01 * np.tanh(curr_hidden)`. No matmul,
    no weights.
  - Output tokens are `np.random.randint(0, vocab_size)`. Acceptance is faked by adding `+10.0`
    to the logit of the token already chosen.
  - `active_weights_gb = 4.8` is a hardcoded constant. `naive_tok_s = 0.08` is hardcoded, and
    "108×" is `measured / 0.08`.
- **`src/llm_lab/frontier/gguf_stream_engine.py`** ("real partitioned GGUF, 19.6 tok/s, 988 MB RSS")
  - Does genuinely mmap the real 1.4 GB hot / 3.2 GB cold files — the partitioner works.
  - But inference reads a **1024-byte slice and discards it** (`q_slice = self.mmap_hot[...:...+1024]`),
    then runs `curr_hidden + 0.005*np.tanh(...)` and `np.random.randint` tokens. It never multiplies
    the hidden state by any weight. Low RSS is *because* nothing is faulted in or computed.
- **`src/llm_lab/core/sparse_router.py`** ("trained predictors", "76.4% skipped")
  - Predictors are `np.random.randn(...)*0.02` — random, never trained; `total_tokens_observed: 0`.
  - The "sparsity %" is `1 - (hot + top_k)/d_intermediate`: a fixed function of config, not a measurement.
- **`src/llm_lab/frontier/converter.py`** — calibration "profiling" is `np.random.zipf`.

### Real, but broken by design

- **`src/llm_lab/core/speculative_real.py`** — loads a real HF model, but drafts K tokens by calling
  the **full target model** K times, then runs it once more to verify, with **no KV cache**
  (`self.target_model(generated)`, never `past_key_values`). That is more forward passes than plain
  decoding over an O(n²) recompute, so it is strictly **slower**, not "2.7–3.2× faster". Verification
  is greedy argmax, not the rejection sampling the README's math section describes.

### Real and worth keeping

- **`src/llm_lab/core/kernels.py`** — correct INT4 pack/unpack + numba GEMV (signed 4-bit, per-row scale).
- **`src/llm_lab/core/layers.py`** — `FusedInt4Linear` works as a drop-in.
- **`src/llm_lab/frontier/gguf_partitioner.py`** — really parses a GGUF and splits tensors with a manifest.
  Its instinct (contiguous grouping + manifest offsets) is what M3 needs.
- **`src/llm_lab/models/runner.py`**, **`benchmarks/benchmark_speculative_real.py`** — real
  llama-cpp-python / HF paths, usable as starting points.

---

## 3. Physical limits (the arithmetic that bounds everything)

Reference machine (the constrained tier, and the showcase box):

| Component | Measured / known |
|---|---|
| CPU | AMD Ryzen 5 5600G, 6C/12T — **Cezanne APU, PCIe 3.0 only** |
| RAM | 16 GB DDR4-3200, ~28.93 GB/s measured |
| NVMe | ~2.62 GB/s sequential measured (near the PCIe 3.0 x4 ceiling) |
| OS | Windows 11 Pro |

Decode is memory-bound: every token must read every **active** parameter once.

### Dense 70B on 16 GB is excluded, not unsolved

Llama-3-70B @ Q4_K_M ≈ 40 GB; ~10 GB usable → ~30 GB crosses NVMe per token:

```
30 GB / 2.62 GB/s = 11.5 s/token = 0.087 tok/s
```

That is exactly the repo's invented `0.08` "baseline". To reach 15 tok/s you would need
30 GB in 66 ms = **~450 GB/s of storage bandwidth**, ~170× this drive. Four PCIe 5.0 SSDs in
RAID still fall ~9× short. **No code fixes this.** Dense large models on tiny RAM are out of scope
for interactive speed; we will say so publicly.

### Throughput ceiling as a function of cold traffic (this drive)

```
tokens/s  ≤  B_cold_tier / cold_bytes_per_token
```

| Cold bytes/token | Ceiling @ 2.62 GB/s |
|---:|---:|
| 1 GB | 2.6 tok/s |
| 500 MB | 5.2 tok/s |
| 250 MB | 10.5 tok/s |
| 100 MB | 26.2 tok/s |
| 50 MB | 52.4 tok/s |

Optimistic — sequential, before compute, read amplification, and contention. **Interactive
(~10 tok/s) on this box needs well under 262 MB/token, realistically 50–100 MB.**

### Why MoE changes the picture (Qwen3-30B-A3B, arithmetic to be verified by `meta.py`)

Expected shape: ~48 layers, 128 experts/layer, top-8, hidden 2048, moe_intermediate 768.

```
per expert  = 3 matrices (gate, up, down) x 2048 x 768 = 4.72M params
            ≈ 2.65 MB at ~4.5 bits (Q4_K_M)
per token   = 8 experts x 48 layers x 2.65 MB ≈ 1.0 GB   (uncached)
all experts = 128 x 48 x 2.65 MB ≈ 16.3 GB
```

So ~1 GB/token uncached → ~2.6 tok/s floor; the whole project is about driving that number
down with **exact** caching and layout, and measuring what is left.

### Energy caveat (must appear in any public claim)

SSD expert offloading costs roughly **4.9× the per-token energy of HBM** and **3.1× vs CPU DRAM**,
with SSD access ~80% of per-token energy (arXiv 2508.06978). Also relevant: sustained streaming
write/read endurance. Laptop battery life is part of "helping low-end users".

---

## 4. Exactness taxonomy (the vocabulary we must not blur)

**Exactly lossless**

- **MoE expert streaming** — the router is deterministic; fetching only the selected experts and
  demand-loading on a miss computes the same thing, just later. Bit-identical.
- **Async prefetch / double buffering** — pure I/O scheduling.
- **Speculative decoding** — distribution-identical with proper rejection sampling; bit-identical
  under greedy match verification.

**Lossy — do not call these lossless**

- Thresholded activation sparsity on SwiGLU/SiLU (SiLU is never exactly zero).
- SnapKV / sliding-window KV eviction (it discards tokens).
- KIVI 4-bit KV cache (0.9942 cosine ≠ 1.0).

**Q4 is itself lossy vs FP16.** Therefore the only honest phrasing is:

> bit-identical to `Qwen3-30B-A3B-Q4_K_M` under greedy decoding with seed `S`

…never "identical to full precision".

**For cross-machine bit-identity we must also fix:** quantized arithmetic format, operation and
reduction order, router tie-breaking, sampling PRNG, and thread count where it changes reduction order.
Cache size must **never** influence routing.

---

## 5. Prior art — we are not first

Established before we started; cite, do not reinvent:

| Work | What it already does |
|---|---|
| **SP-MoE** (arXiv 2510.10302, Oct 2025) | Draft model's per-layer attention output → target's gating net to predict experts. Async prefetcher, batched I/O, LRU, profiled cutoff layer. 1.07–3.5× over Mixtral-Offloading / MoE-Infinity / AdapMoE. **Its Figure 2 already measures union-growth sublinearity and pairwise expert overlap**; ~88% top-1 prediction accuracy. DRAM→GPU over PCIe. |
| **MoE-SpeQ** (arXiv 2511.14102, Nov 2025) | Small on-device draft model predicts the expert sequence; runtime orchestrator prefetches. 2.34× over SOTA offloading, >96% cache hit. Host DRAM, not SSD. |
| **SpecMD** (Apple, arXiv 2602.03921) | "A Comprehensive Study on Speculative Expert Prefetching." 4 MoE models, 3 hardware configs, >88% hit rates, eviction policy beating LRU by up to 85%. |
| MoE-Infinity, ExpertFlow, AdapMoE, HOBBIT, Fiddler, ktransformers, MoE-Beyond, FineMoE (EuroSys'26), DALI, OD-MoE, Pre-gated MoE, Mixtral-offloading (2312.17238) | The expert-offloading/caching/prefetch design space. |
| **FlashMoE** (arXiv 2601.17063) | SSD-tier MoE offloading for edge devices — closest to our niche. |
| DFloat11 (2504.11651), Unweight (Cloudflare Cf-TR-2026.04.v1), ZipServ (ASPLOS'26), ZipNN, Huff-LLM, NeuZip, EntroLLM, BackSlash (ICML'25) | Lossless weight compression + fused decode. Crowded. |

**Two hard facts from that literature:**

1. **Packed 4-bit payloads barely compress losslessly** (arXiv 2508.19263): FP4/INT4 values show
   essentially no useful gain; only scale metadata compresses (~5% total on DeepSeek-R1 NVFP4).
   BF16 gets ~30% only because its 8-bit exponent is redundant. **There is no 5–10× lossless
   compression of a Q4 checkpoint.** Do not plan around one.
2. Draft-guided expert prefetch is **published three times over**, including by Apple. We may
   implement it; we may not claim it.

**The remaining gap we are targeting:** almost all of the above assumes a GPU and prefetches from
**host DRAM**. The GPU-free, SSD-backed, low-RAM **Windows** configuration is comparatively
underserved, and none of it is in llama.cpp — see open discussion
[ggml-org/llama.cpp#27149](https://github.com/ggml-org/llama.cpp/discussions/27149),
"Expert-Aware SSD Streaming for MoE Models on Memory-Constrained Hardware", which proposes exactly
our target ("machines with no GPU and insufficient RAM (8-16GB consumer laptops)") and is a
discussion, not an implementation.

**Honest positioning:** engineering contribution + rigorous measurement, not novel research.

---

## 6. Strategy: MoE first, PageCC later

Decision (2026-09-10): do **not** start by training a new architecture. Build the exact MoE runtime
first, because it (a) works on checkpoints that exist, (b) gives real exactness against a real model,
(c) produces the routing traces any future architecture work needs, and (d) helps users even if the
research bet fails.

**PageCC** — Page-Coherent Compositional Parameter Memory — remains the research hypothesis:
a small resident core plus page-aligned *atoms* shared across layers, a deterministic early router,
and a training loss penalizing **distinct physical page bytes over a sliding token window**.
Novelty confidence ~55–65% before a professional patent search. Closest prior art: PEER (2407.04153),
Memory Layers at Scale (2412.09764), Engram (2601.07372), ReMoE/StickyMoE,
**Temporally Extended MoE (2604.20156 — already does learned persistent expert masks over spans)**,
Relaxed Recursive Transformers, MatFormer/Matryoshka.

Invariant for any PageCC design: **quality mode chooses the graph and addresses; hardware chooses
only where the bytes live.** A cache-conditioned router breaks the whole premise.

PageCC arrives as **M6/M7**, as a continued-training evolution of a pretrained MoE — never a
from-scratch pretrain.

---

## 7. Milestone decomposition

Each milestone has a gate. We stop or escalate on the measured result, not on enthusiasm.

| # | Milestone | Output | Gate to proceed |
|---|---|---|---|
| **M0** | **Truth-in-labeling** | README/DEMO_SCRIPT rewritten to claim only what runs; simulated engines renamed/quarantined as `sim_*` or deleted | No public artifact contains an unmeasured number |
| **M1** | **Reference baseline + exactness oracle + traces** *(approved, designed below)* | Real generation on Qwen3-30B-A3B Q4_K_M; determinism proven; expert-routing + timing traces | Two identical-config runs give identical token IDs; ≥2000 tokens traced across ≥3 domains |
| **M2** | **Trace analysis → decision gate** | Cache-hit curves, union growth vs window, reuse distance, co-activation, predicted tok/s per tier | See numeric gate below |
| **M3** | **Expert-contiguous repack** | Aligned, contiguous `(layer, expert)` blobs + manifest; layout-only, output-exact | Measured read throughput improves vs stock GGUF access; token IDs unchanged |
| **M4** | **Tiered exact async loader** | VRAM/RAM/NVMe cache, real async I/O at QD≥16 (IOCP/io_uring), demand-load on miss | Output identical at *every* cache size; beats llama.cpp `--n-cpu-moe` baseline on the same box |
| **M5** | **Sub-expert page analysis** | Quantify what fraction of a selected expert's pages is actually required | Establishes whether runtime-only subdivision can ever help (expected: it cannot, for dense experts) |
| **M6** | **Page-sparse continued training** | New checkpoint from a pretrained MoE, page-sparse experts, distillation | Beats MoE/PEER at equal active FLOPs; ≤1% quality regression |
| **M7** | **PageCC proper** | Cross-layer compositional atoms + page-locality loss | The full kill-criteria list from the research plan |

### M2 numeric gate

- Realistic RAM cache on the 16 GB profile yields a **predicted ceiling ≥ ~8 tok/s** → build M3/M4.
- Miss traffic stays **> ~500 MB/token** → whole-expert transfer is the bottleneck; escalate to
  M5/M6 rather than polishing the loader.
- The non-expert resident core does not fit a tier → **that tier is publicly dropped**, not faked.

---

## 8. Milestone 1 design

### 8.1 Scope

In scope: measure the truth about one real MoE checkpoint.
Out of scope: any custom streaming engine, any speedup claim, any PageCC code, any GPU work.

### 8.2 Recommended sequencing (de-risk before the big download)

Build and validate the whole pipeline on a **small** MoE that fits the 16 GB box
(e.g. OLMoE-1B-7B or Qwen1.5-MoE-A2.7B), then run the *same code* on Qwen3-30B-A3B Q4_K_M.
This is cheaper than debugging a 18 GB download.

### 8.3 Components (4 modules, 1 test — keep it small)

```
src/llm_lab/moe/
  meta.py       # read GGUF/HF config → n_layer, n_expert, n_expert_used,
                # per-expert byte size, total expert bytes. No guesses: report what the file says.
  baseline.py   # run one real generation, greedy, deterministic → RunRecord
  trace.py      # routing-trace writer/reader (JSONL schema below)
  oracle.py     # RunRecord A vs B → identical? first divergence index?
tests/test_moe_trace.py   # assert-based self-check: schema round-trip + union math on a tiny trace
```

No abstract base classes, no plugin registry, no config framework. Wire into the existing
`llm_lab` CLI as subcommands.

### 8.4 Two numeric paths — stated explicitly, never conflated

| Path | Purpose | Gives |
|---|---|---|
| **GGUF Q4_K_M via `llama-cpp-python`** (already a dependency) | Speed, RSS, determinism, token IDs | The performance baseline we must later beat |
| **HF `transformers` with `output_router_logits=True`** | Exact top-k expert IDs per layer per token | The routing trace M2 consumes |

These are **different numeric paths**. M1 does **not** claim they route identically. Whether Q4
routing matches BF16 routing is itself an M2 finding. Getting expert IDs out of llama.cpp needs a
C++ patch — deferred; the HF router-logits path is one config flag.

The 30B BF16 routing run needs ~61 GB and must run on a rented/large machine; the GGUF baseline
runs on the 16 GB box.

### 8.5 Trace schema (JSONL — M2–M7 all consume this, so get it right now)

Header record, once:

```json
{"kind":"header","model":"Qwen3-30B-A3B","path":"...","quant":"Q4_K_M",
 "n_layer":48,"n_expert":128,"n_expert_used":8,
 "expert_bytes":2777088,"total_expert_bytes":17064402944,
 "tokenizer_sha":"...","prompt_sha":"...","seed":0,"decode":"greedy",
 "backend":"transformers|llama-cpp","threads":6,"host":"..."}
```

Then one record per generated position:

```json
{"kind":"step","pos":0,"tok_id":9906,"layer_experts":[[3,17,44,...],[...]]}
```

`layer_experts[l]` is the selected expert IDs for layer `l`, in router order. Fields that cannot be
measured are written `null` — never estimated.

### 8.6 RunRecord (the artifact the oracle compares)

```json
{"config_fingerprint":"sha256 of model+quant+seed+prompt+decode+threads+backend",
 "token_ids":[...],"n_prompt":123,"n_generated":256,
 "wall_s":12.34,"tok_per_s":20.7,"peak_rss_bytes":1234567890,
 "host":{"cpu":"...","ram_bytes":...,"os":"..."}}
```

### 8.7 The exactness oracle

Compares **token ID sequences**, not decoded text. The old repo compared
`base_text.strip() == spec_text.strip()`, which hides divergence and tokenizer artifacts.

```
oracle(A, B) -> {"identical": bool, "first_divergence": int|null,
                 "n_compared": int, "fingerprints_match": bool}
```

If fingerprints differ, the runs are not comparable and the oracle **refuses** rather than reporting
a similarity score.

### 8.8 Data flow

```
checkpoint ──► meta.py ──► architecture facts + expert byte sizes
     │
     ├──► baseline.py (llama-cpp, GGUF Q4) ──► RunRecord  ──┐
     │                                                      ├──► oracle.py ──► identity verdict
     ├──► baseline.py (run twice, same config) ─────────────┘
     │
     └──► baseline.py (transformers, router logits) ──► trace.jsonl ──► (M2 analysis)
```

### 8.9 Error handling

- Missing model file → error naming the exact expected path, non-zero exit.
- **Nondeterminism detected → fail loudly** and record it as a finding. Never average it away;
  without determinism there is no oracle and M1 is not done.
- OOM → report and abort. Never silently fall back to a smaller config or shorter context.
- Any metric that cannot be measured → `null` in the JSON, absent from prose.

### 8.10 Testing

- `tests/test_moe_trace.py` — asserts trace schema round-trip, and union/unique-expert math on a
  hand-built 3-position × 2-layer trace with known answers. This is the kill-test: if the union
  arithmetic breaks, M2's entire conclusion breaks.
- Determinism check: run `baseline.py` twice with the same config, assert identical token IDs.

### 8.11 M1 definition of done

1. A real generation runs on Qwen3-30B-A3B Q4_K_M and emits a RunRecord with measured tok/s and RSS.
2. Two identical-config runs produce **identical token IDs** (or nondeterminism is documented as a finding).
3. ≥2000 generated positions traced across ≥3 prompt domains (prose, code, factual recall).
4. `meta.py` output confirms or corrects the architecture numbers assumed in §3.
5. Every number in the M1 report traces to a command and its saved output in the repo.

---

## 9. Hardware plan

| Priority | Action | Why |
|---|---|---|
| 1 | **16 → 32 GB DDR4-3200 (~$40–60)** | Doubles the expert cache; likely ~2× tok/s. Best dollar-per-token available. |
| 2 | **Do not buy an SSD** | 5600G is Cezanne = PCIe 3.0; measured 2.62 GB/s is already near the platform ceiling. A Gen4 drive is wasted money. |
| 3 | Rented ~128 GB machine, hours at a time | Needed to hold 30B BF16 for the routing trace and to establish the fully-resident reference. |
| 4 | 24 GB GPU — later, M4 only | For the VRAM-tier path. Out of M1 scope. |

The 16 GB 5600G box stays the **showcase / worst-case target**. Keep it at 16 GB for exactly that reason.

---

## 10. Reporting rules (anti-fabrication)

Adopted because the current repo violated all five.

1. No number in a README, report, or commit message unless it came from a run whose **command and
   output are saved in the repo**.
2. Baselines are **measured on the same machine, same session, same checkpoint**. Never a literature
   constant presented as "baseline" (this is how `0.08` and `108×` happened).
3. Projections are labeled `PROJECTED` and shown with their formula and inputs.
4. Output identity is **token-ID equality**, never `.strip()` text comparison.
5. "Lossless" always names its reference checkpoint, quantization, decode mode, and seed.

---

## 11. Open questions (for M1/M2 to answer, not to guess)

1. Do Q4_K_M and BF16 select the **same** experts for the same prompt? (Decides whether the
   HF-derived trace is valid for the GGUF path.)
2. How large is the non-expert resident core (attn + embed + norms + router) at Q4_K_M?
   Does it fit 4 GB? 8 GB?
3. Is llama-cpp-python decode bit-deterministic on this box across runs and thread counts?
4. What is the real reuse distance / co-activation structure of Qwen3-30B-A3B routing?
5. Can expert IDs be extracted from llama.cpp without a C++ patch?

## 12. Artifacts index

Research evidence from the 2026-09-10 session lives in `docs/research/raw/`:

- `novelty-research.md` — the broad prior-art survey (gap inventory, references)
- `unweight.txt`, `backslash.txt` — full-text extractions of the two key compression papers
- `s*.json`, `a*.json`, `h*.json`, `novelty-*.json`, `cert1.json`, `proc1.json`, `route1.json`,
  `compression-training.json`, `entropy-train2.json`, `page-locality.json` — raw Tavily search results
