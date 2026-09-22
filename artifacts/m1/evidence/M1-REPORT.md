# M1 Reference Pipeline — Measured Results

Milestone 1 built a deterministic MoE reference: a real baseline, a determinism proof, and
expert-routing traces. This report records **only** measurements that live in committed
evidence under `artifacts/m1/evidence/`. Every number below names the file it came from. No
number here was projected, estimated, or carried over from literature.

Phases gate in order: **M1a** (tiny, no download) → **M1b** (small real OLMoE, both numeric
paths) → **M1c** (Qwen3-30B-A3B, the milestone target). A later phase runs only after the
earlier gate passes.

---

## Two numeric paths — never asserted equal

The pipeline exercises two genuinely different executions (spec §8.4):

- **GGUF Q4_K_M via `llama-cpp-python`** — speed, RSS, determinism, and token IDs. Runs on
  the 16 GB Windows box.
- **HF `transformers` with `output_router_logits=True`** — the exact top-k expert IDs per
  layer per token (the routing trace M2 will consume). The 30B BF16 routing run needs far
  more RAM and was executed on a rented Linux host.

These are **different numeric paths**. M1 does **not** assert that Q4_K_M decoding and BF16
routing select the same experts or emit the same tokens. Whether Q4 routing matches BF16
routing is an **open M2 question** (spec §11 open-Q1). Nothing in this report claims token
identity between a Q4 path and a BF16 path.

## Exactness phrasing

Where this report speaks of exactness for the local target, it means precisely:

> bit-identical to Qwen3-30B-A3B-Q4_K_M under llama.cpp greedy decoding, seed 0, 6 threads,
> on the recorded host

Exactness always names checkpoint + quantization + decode + seed + host. It is **never**
"identical to full precision" or "identical to BF16". Q4_K_M is itself lossy versus higher
precision; the reference is the Q4_K_M checkpoint under a fixed decode on a recorded machine,
nothing wider.

## Output identity is token-ID equality

The oracle compares two runs by their integer token-ID sequences and their config
fingerprints. "Identical" means the token-ID lists match; it is never a `.strip()` text
comparison. The oracle **refuses** to compare two runs whose config fingerprints differ
(raising rather than returning a verdict), so an "identical" result can only be reported for
runs of the same measured configuration.

## Host split (disclosed)

- **M1a, M1b, and M1c-local** ran on Windows (AMD Ryzen 5 5600G, `AMD64 Family 25 Model 80`,
  ~16 GB RAM — `Windows-10-10.0.26200-SP0`; `total_ram_bytes` 16886743040 ≈ 15.7 GiB per host
  blocks in `m1b-gguf-run-a.json`, `m1c-run-a.json`).
- **M1c rented BF16** ran on Linux (`Linux-5.15.0-191-generic-x86_64-with-glibc2.35`,
  `total_ram_bytes` 135059517440 ≈ 135 GB, per host block in
  `m1c-trace-code-summary.json`). This is a Vultr Ubuntu 22.04 rental.

## SSD energy caveat (spec §3)

Any public performance discussion of the eventual low-memory design must carry this: SSD
expert offloading costs roughly **4.9× the per-token energy of HBM** and **3.1× versus CPU
DRAM**, with SSD access about 80% of per-token energy (arXiv 2508.06978). The M1 tok/s
figures below are baselines to later beat, not endorsements of an energy profile.

---

## M1a — tiny deterministic baseline (Windows, no download)

A randomly-initialized tiny OLMoE-shaped model, FP32, greedy, seed 0. Proves the oracle,
the trace schema, and determinism without touching the network.

- Two independent builds produced **identical** token IDs: `identical: true`,
  `first_divergence: null`, `n_compared: 32`, `fingerprints_match: true`
  (`artifacts/m1/evidence/m1a-oracle.json`).
- Trace: `n_layer` 2, `n_expert` 8, `n_expert_used` 2, `n_prompt` 3, `n_generated` 32,
  `n_steps` 32, `decode` greedy, `stop_at_eos` true, `precision` FP32
  (`artifacts/m1/evidence/m1a-trace-summary.json`).
- File hashes for the raw trace and logits are committed in
  `artifacts/m1/evidence/m1a-files.sha256`; the full test transcript is
  `artifacts/m1/evidence/m1a-test-output.txt`.

## M1b — small real OLMoE, both numeric paths (Windows)

Model: `allenai/OLMoE-1B-7B-0924` family. Both paths ran the prompt "The capital of France
is" under greedy decoding.

**Metadata (measured, GGUF Q4_K_M):** `n_layer` 16, `n_expert` 64, `n_expert_used` 8,
`quant` Q4_K_M, `expert_bytes` null (mixed per-layer quant), `total_expert_bytes`
3900702720 (`artifacts/m1/evidence/m1b-gguf-meta.json`).

**Determinism (each path against itself):**

- GGUF Q4_K_M: `identical: true`, `n_compared: 64`, `fingerprints_match: true`
  (`artifacts/m1/evidence/m1b-gguf-oracle.json`).
- HF int8: `identical: true`, `n_compared: 64`, `fingerprints_match: true`
  (`artifacts/m1/evidence/m1b-hf-oracle.json`).

**Distinct engine profiles (same prompt, different executions):**

- GGUF Q4_K_M: `tok_per_s` 36.33, `peak_rss_bytes` 7475109888, `n_generated` 64
  (`artifacts/m1/evidence/m1b-gguf-run-a.json`).
- HF int8: `tok_per_s` 2.00, `peak_rss_bytes` 12522942464, `n_generated` 64
  (`artifacts/m1/evidence/m1b-hf-run-a.json`).

**Observed output convergence — reported honestly, not as a result.** On this one
strongly-determined greedy prompt, the GGUF Q4 run and the HF int8 run happened to emit
identical token-ID lists (compare the `token_ids` arrays of `m1b-gguf-run-a.json` and
`m1b-hf-run-a.json`). This is **observed output convergence on a single greedy prompt**, not
routing identity, not decode identity, and not a cross-path exactness claim. The two runs
have different config fingerprints and very different engine profiles (≈36 tok/s vs ≈2 tok/s
above) — they are genuinely different executions that agreed on this prompt's tokens. Nothing
in M1 asserts they are equal in general; the oracle itself would refuse to compare them
because their fingerprints differ.

**Traces:** 5 traces × 100 steps = 500 positions across prose/code/factual domains, each hash
committed (`artifacts/m1/evidence/m1b-trace-summary.json`).

**Cache replay is observational.** Replaying the committed trace against one immutable
content-addressed logits digest at two capacities changes only hits/misses, never the tokens
or the logits hash:

- capacity 64: `hits` 0, `misses` 12800, `requests` 12800
  (`artifacts/m1/evidence/m1b-cache-64.json`).
- capacity 512: `hits` 9969, `misses` 2831, `requests` 12800
  (`artifacts/m1/evidence/m1b-cache-512.json`).

Both share the same `logits_sha256`
(`626606ea12a38005b778d0c8955f0d0a96123571cde53fde456a58346e4cb317`) and `token_ids`, so the
replay is arithmetic over one recorded trace, never a second model execution.

**Gate:** `gguf_deterministic`, `hf_int8_deterministic`, `positions_at_least_500`,
`cache_observational_only`, `raw_hashes_present` all true; `passed: true`
(`artifacts/m1/evidence/m1b-gate.json`).

## M1c — Qwen3-30B-A3B, the milestone target

**Local Q4_K_M (Windows, `llama-cpp-python`, greedy, seed 0, 6 threads).**

- Metadata (measured): `model` qwen3moe, `n_layer` 48, `n_expert` 128, `n_expert_used` 8,
  `quant` Q4_K_M, `expert_bytes` null (mixed per-layer quant), `total_expert_bytes`
  17553162240 (`artifacts/m1/evidence/m1c-meta.json`). These match spec §3
  (48 layers, 128 experts, 8 used).
- Determinism: two local runs `identical: true`, `first_divergence: null`, `n_compared: 256`,
  `fingerprints_match: true` (`artifacts/m1/evidence/m1c-oracle.json`). Both runs carry the
  same `config_fingerprint`
  (`ec15e85cba3747e91d791f3b830926731dac750dff00de7e9327fbdd2267f080`) in
  `m1c-run-a.json` and `m1c-run-b.json`.
- Baseline profile: `tok_per_s` 6.02, `peak_rss_bytes` 13759725568, `n_generated` 256
  (`artifacts/m1/evidence/m1c-run-a.json`).
- Local gate: `local_generation_real`, `q4_deterministic`, `measured_metadata`,
  `passed_local` all true (`artifacts/m1/evidence/m1c-local-gate.json`).

This local target is **bit-identical to Qwen3-30B-A3B-Q4_K_M under llama.cpp greedy decoding,
seed 0, 6 threads, on the recorded Windows host** — the two runs above prove reproducibility
of exactly that configuration. No wider exactness (versus BF16 or full precision) is claimed.

**Rented BF16 routing traces (Linux, `transformers`, greedy, seed 0, 6 threads).**

- Checkpoint `Qwen/Qwen3-30B-A3B` at revision
  `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, `precision` BF16, `quant` torch.bfloat16
  (`artifacts/m1/evidence/m1c-trace-code-summary.json` and the other three summaries).
- 4 traces × 500 steps = 2000 positions across code / factual / prose domains, each hash
  committed (`artifacts/m1/evidence/m1c-trace-summary.json`).
- Metadata on the trace path also measures `n_layer` 48, `n_expert` 128, `n_expert_used` 8.
- The rented host does not commit; its transcript is
  `artifacts/m1/evidence/m1c-rented-output.txt` and raw-file hashes are in
  `artifacts/m1/evidence/m1c-files.sha256`.

**Gate:** `local_generation_real`, `q4_deterministic`, `positions_2000`, `three_domains`,
`bf16_precision`, `measured_metadata`, `evidence_hashes_complete` all true;
`total_trace_steps` 2000; `passed: true` (`artifacts/m1/evidence/m1c-gate.json`).

**The Q4 baseline (local) and the BF16 traces (rented) are different numeric paths and were
not asserted equal.** The Q4 run gives the performance baseline; the BF16 traces give the
routing record for M2. Whether Q4 routing equals BF16 routing is the M2 question, not an M1
claim.

---

## Trace semantics (for M2 consumers)

- `tok_id` in a trace step is the **processed** token — the token whose hidden state selected
  the recorded experts at that position — for generated positions only.
- Cache replay is **observational**: it reads one immutable trace plus one content-addressed
  logits digest and computes hits/misses at a chosen capacity. It never runs the model a
  second time.

## Artifact policy

Raw traces, logits tensors, and model checkpoints are gitignored
(`/artifacts/m1/raw/`, `/models/m1/`). What is committed under
`artifacts/m1/evidence/` is the compact evidence: JSON summaries, oracle/gate verdicts,
command transcripts, and `*.sha256` hash manifests that bind each committed summary back to
the raw file it describes.
