# Milestone 1 MoE Reference Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an honest, deterministic MoE metadata/baseline/trace/oracle pipeline, prove it first on a download-free tiny OLMoE, then rehearse both numeric paths on OLMoE-1B-7B before any Qwen3-30B-A3B download, and finally capture the gated Qwen3 M1 evidence.

**Architecture:** Keep the approved four-module boundary. `meta.py` reports only checkpoint facts; `baseline.py` owns deterministic HF and llama.cpp execution plus `RunRecord`; `trace.py` owns JSONL routing records and observational cache replay; `oracle.py` compares token IDs only. The existing argparse entry point exposes these as `llm-lab moe meta|run|trace|compare|replay-cache`; every expensive phase is gated by artifacts from the phase before it.

**Tech Stack:** Python 3.11; stdlib `argparse`, `dataclasses`, `hashlib`, `json`, `pathlib`, `platform`, `shutil`, `time`; PyTorch 2.14 CPU; Transformers 5.16.1; llama-cpp-python 0.3.35; gguf 0.19.0; psutil; bitsandbytes 0.50.2 and Accelerate 1.15.0 for Windows CPU int8 HF loading. Tests are one directly executable assert-based script; no test framework is added.

**Spec:** `docs/superpowers/specs/2026-09-11-moe-exact-runtime-design.md` (§8; approved for M1 only)

## Global Constraints

- Work only on M1: no custom streaming engine, speedup claim, PageCC code, GPU work, or M2 analysis.
- Phase order is normative: M1a tiny random OLMoE, then M1b OLMoE-1B-7B, then M1c Qwen3-30B-A3B.
- Do not download Qwen3 until every M1b gate in Task 7 passes.
- The constrained host is Windows 11, Ryzen 5 5600G, 16 GB RAM; use `./.venv/Scripts/python.exe` and the CPU PyTorch index already configured in `pyproject.toml`.
- Decode is greedy with an explicit seed and thread count. Baseline exactness always means token-ID equality within the same checkpoint, quantization, backend, decode mode, prompt, seed, and thread count. Fixed-length trace collection may continue after EOS solely to meet position-count gates and records `stop_at_eos=false` in its summary.
- The GGUF Q4_K_M llama.cpp path measures token IDs, wall time, decode tok/s, and peak RSS; the HF path captures router logits/top-k experts. Never claim the two numeric paths route or decode identically.
- `tok_id` in a trace step is the processed token whose hidden state selected `layer_experts`, not the next token emitted from its logits. M1 traces generated positions only; prompt positions are excluded.
- JSON fields that cannot be measured are `null`; nothing is estimated.
- Missing files, insufficient disk, nondeterminism, incomparable fingerprints, and OOM are loud errors with non-zero CLI exits. Never shorten context or substitute a model silently.
- Large raw traces and per-position logits stay under ignored `artifacts/m1/raw/`. Commit exact command scripts, captured stdout/stderr transcripts, compact JSON summaries, and SHA-256/byte-count manifests under `artifacts/m1/evidence/`.
- No reported number is valid unless the command and output that produced it are committed. Projections must say `PROJECTED` and include formulas and inputs.
- OLMoE M1b uses official revisions: HF `allenai/OLMoE-1B-7B-0924@6d84c48581ece794365f2b8e9cfb043c68ade9c5` in CPU int8 and GGUF `allenai/OLMoE-1B-7B-0924-GGUF@70df85ed7132bf21b5acbcb33817f58e7d3cb949`, file `olmoe-1b-7b-0924-q4_k_m.gguf` (4,213,511,776 bytes).
- Qwen3 M1c uses official revisions: GGUF `Qwen/Qwen3-30B-A3B-GGUF@e4d4bafdfb96a411a163846265362aceb0b9c63a`, file `Qwen3-30B-A3B-Q4_K_M.gguf` (18,556,685,824 bytes), and HF BF16 `Qwen/Qwen3-30B-A3B@ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`.
- Do not commit model weights or raw traces.

## File Map

- Create `src/llm_lab/moe/__init__.py`: package marker only.
- Create `src/llm_lab/moe/meta.py`: `ModelMeta`, strict HF/GGUF fact readers, and expert-byte accounting.
- Create `src/llm_lab/moe/baseline.py`: `RunConfig`, `RunRecord`, canonical fingerprints, host/RSS measurement, deterministic tiny/HF/llama.cpp execution.
- Create `src/llm_lab/moe/trace.py`: JSONL header/step validation, generated-position HF routing capture, logits manifest, union math, and observational LRU cache replay.
- Create `src/llm_lab/moe/oracle.py`: comparable-run refusal and first-divergence token-ID oracle.
- Modify `src/llm_lab/__init__.py`: add the `moe` argparse tree and dispatch only.
- Modify `pyproject.toml` and `uv.lock`: pin only the two dependencies proven necessary for local int8 HF loading.
- Modify `.gitignore`: ignore `models/m1/` and `artifacts/m1/raw/`, while leaving `artifacts/m1/evidence/` trackable.
- Modify `README.md`: document M1 commands, phase gates, artifact policy, and honest exactness wording after the pipeline works.
- Create `tests/test_moe_trace.py`: one stdlib/assert-driven kill-test script covering all M1a contracts and CLI missing-file behavior.
- Create `artifacts/m1/evidence/*` during Tasks 6–8: committed commands, outputs, summaries, and content manifests; never hand-written measurements.

---

### Task 1: Lock Core Records, Fingerprints, and Oracle

**Files:**
- Create: `src/llm_lab/moe/__init__.py`
- Create: `src/llm_lab/moe/baseline.py`
- Create: `src/llm_lab/moe/oracle.py`
- Create: `tests/test_moe_trace.py`

**Interfaces:**
- Consumes: stdlib plus `psutil` already installed.
- Produces:
  - `RunConfig(model: str, path: str, quant: str | None, seed: int, decode: str, threads: int, backend: str, prompt_sha: str, tokenizer_sha: str | None)`
  - `RunRecord(config_fingerprint: str, token_ids: tuple[int, ...], n_prompt: int, n_generated: int, wall_s: float | None, tok_per_s: float | None, peak_rss_bytes: int | None, host: dict[str, object])`
  - `sha256_bytes(data: bytes) -> str`, `sha256_file(path: Path) -> str`, and `sha256_token_ids(token_ids: Sequence[int]) -> str`, where token IDs are hashed as canonical compact JSON.
  - `RunRecord.read(path: Path) -> RunRecord`, `RunRecord.write(path: Path) -> None`
  - `fingerprint_config(config: RunConfig) -> str`
  - `compare_runs(a: RunRecord, b: RunRecord) -> dict[str, bool | int | None]`

- [ ] **Step 1: Write the failing RunRecord and oracle tests**

Append these tests to `tests/test_moe_trace.py` before creating implementation modules:

```python
from pathlib import Path
import tempfile

from llm_lab.moe.baseline import RunConfig, RunRecord, fingerprint_config
from llm_lab.moe.oracle import compare_runs


def sample_config() -> RunConfig:
    return RunConfig(
        model="tiny-olmoe",
        path="local-random",
        quant="FP32",
        seed=0,
        decode="greedy",
        threads=1,
        backend="transformers",
        prompt_sha="a" * 64,
        tokenizer_sha=None,
    )


def sample_record(token_ids=(7, 8, 9), *, fingerprint=None) -> RunRecord:
    fp = fingerprint or fingerprint_config(sample_config())
    return RunRecord(fp, token_ids, 3, len(token_ids), 1.5, 2.0, 1234, {"os": "test"})


def test_run_record_round_trip_and_stable_fingerprint():
    record = sample_record()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "run.json"
        record.write(path)
        assert RunRecord.read(path) == record
    assert fingerprint_config(sample_config()) == fingerprint_config(sample_config())


def test_oracle_reports_identity_and_first_divergence():
    assert compare_runs(sample_record(), sample_record()) == {
        "identical": True,
        "first_divergence": None,
        "n_compared": 3,
        "fingerprints_match": True,
    }
    assert compare_runs(sample_record(), sample_record((7, 5, 9))) == {
        "identical": False,
        "first_divergence": 1,
        "n_compared": 2,
        "fingerprints_match": True,
    }


def test_oracle_refuses_different_fingerprints():
    try:
        compare_runs(sample_record(), sample_record(fingerprint="b" * 64))
    except ValueError as exc:
        assert "fingerprints differ" in str(exc)
    else:
        raise AssertionError("oracle compared incompatible runs")
```

- [ ] **Step 2: Run the focused tests and verify the import failure**

Run:

```powershell
./.venv/Scripts/python.exe tests/test_moe_trace.py
```

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'llm_lab.moe'`.

- [ ] **Step 3: Implement immutable records and canonical JSON**

In `baseline.py`, implement frozen dataclasses with explicit `to_dict`/`from_dict` methods. Fingerprint exactly this UTF-8 payload:

```python
payload = json.dumps(
    config.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
).encode("utf-8")
return hashlib.sha256(payload).hexdigest()
```

Validate non-negative counts/metrics, `decode == "greedy"`, and 64-character SHA fields. Write JSON via a sibling temporary file followed by `Path.replace()` so interrupted runs do not leave plausible partial records. `host_info()` returns only measured values: `platform.processor()`, `psutil.virtual_memory().total`, and `platform.platform()`.

In `oracle.py`, reject different fingerprints before looking at tokens. Define `first_divergence` as the zero-based generated-token index; if one sequence is a strict prefix, it is `min(len(a), len(b))`. Set `n_compared` to the number of positions inspected through the verdict: `len(a)` when identical; `first_divergence + 1` when both runs contain the divergent position; or the shorter length for a strict-prefix divergence.

- [ ] **Step 4: Run the Task 1 tests**

Run:

```powershell
./.venv/Scripts/python.exe tests/test_moe_trace.py
```

Expected: PASS for all Task 1 tests.

- [ ] **Step 5: Commit Task 1**

```powershell
git add src/llm_lab/moe/__init__.py src/llm_lab/moe/baseline.py src/llm_lab/moe/oracle.py tests/test_moe_trace.py
git commit -m @'
feat(moe): add run records and exactness oracle

Co-Authored-By: Claude Code <noreply@anthropic.com>
'@
```

### Task 2: Add Strict HF/GGUF Metadata Readers

**Files:**
- Create: `src/llm_lab/moe/meta.py`
- Modify: `tests/test_moe_trace.py`

**Interfaces:**
- Consumes: stdlib `json`, `gguf.GGUFReader`, GGUF keys `general.architecture`, `{arch}.block_count`, `{arch}.expert_count`, `{arch}.expert_used_count`, `{arch}.expert_feed_forward_length`, and expert tensors `blk.{layer}.ffn_{gate,up,down}_exps.weight`.
- Produces:
  - `ModelMeta(model: str, path: str, quant: str | None, n_layer: int, n_expert: int, n_expert_used: int, expert_bytes: int | None, total_expert_bytes: int | None)`
  - `read_hf_meta(config_path: Path) -> ModelMeta`
  - `read_gguf_meta(model_path: Path) -> ModelMeta`
  - `read_meta(path: Path, backend: str) -> ModelMeta`

- [ ] **Step 1: Add failing local-config and synthetic-GGUF tests**

Add a helper that writes this local `config.json`:

```python
config = {
    "model_type": "olmoe",
    "hidden_size": 64,
    "intermediate_size": 32,
    "num_hidden_layers": 2,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "torch_dtype": "float32",
}
```

Assert `read_hf_meta(config_path)` returns `n_layer=2`, `n_expert=8`, `n_expert_used=2`, `expert_bytes=3 * 64 * 32 * 4`, and `total_expert_bytes=2 * 8 * expert_bytes`.

Create a tiny GGUF with `gguf.GGUFWriter(path, arch="olmoe")`, the same architecture keys, and one `float32` expert tensor per gate/up/down per layer. Assert `read_gguf_meta()` gets layer/expert counts from metadata and computes per-expert/total bytes from each tensor's `n_bytes // n_expert`, not from a guessed bits-per-weight formula.

Also assert a missing required key raises `ValueError` naming that key and a missing path raises `FileNotFoundError` naming the exact path.

- [ ] **Step 2: Run tests and verify missing symbols**

Run:

```powershell
./.venv/Scripts/python.exe tests/test_moe_trace.py
```

Expected: FAIL because `llm_lab.moe.meta` does not exist.

- [ ] **Step 3: Implement metadata readers without defaults**

For HF, read `config.json` with stdlib `json` (the file path is the evidence) and require the five fields used by the formula; do not let `AutoConfig` fetch anything. OLMoE uses `intermediate_size`; Qwen3-MoE uses `moe_intermediate_size`. Map the explicit config dtype string (`float32`, `float16`, or `bfloat16`) to 4 or 2 bytes; if dtype is absent or unsupported, return `expert_bytes=None`. Compute `expert_bytes = 3 * hidden_size * expert_intermediate_size * dtype_bytes` only when every input came from the file.

For GGUF, decode scalar fields from `GGUFReader.fields`, require the architecture/expert keys, find all three expert tensors for each layer, and require equal per-expert byte totals across layers. If tensor layout or divisibility is inconsistent, raise `ValueError`; never substitute spec assumptions. Map integer `general.file_type` through `gguf.LlamaFileType` and strip the `MOSTLY_` prefix (for example, `MOSTLY_Q4_K_M` becomes `Q4_K_M`); otherwise quant is `None`.

- [ ] **Step 4: Run metadata and full M1 tests**

Run:

```powershell
./.venv/Scripts/python.exe tests/test_moe_trace.py
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```powershell
git add src/llm_lab/moe/meta.py tests/test_moe_trace.py
git commit -m @'
feat(moe): read checkpoint architecture facts

Co-Authored-By: Claude Code <noreply@anthropic.com>
'@
```

### Task 3: Implement Trace Schema, Union Math, and Cache Replay

**Files:**
- Create: `src/llm_lab/moe/trace.py`
- Modify: `tests/test_moe_trace.py`

**Interfaces:**
- Consumes: `ModelMeta`, `RunRecord`, stdlib JSONL.
- Produces:
  - `TraceHeader(kind, model, path, quant, n_layer, n_expert, n_expert_used, expert_bytes, total_expert_bytes, tokenizer_sha, prompt_sha, seed, decode, backend, threads, host)`
  - `TraceStep(kind, pos, tok_id, layer_experts)`
  - `write_trace(path: Path, header: TraceHeader, steps: Iterable[TraceStep]) -> None`
  - `read_trace(path: Path) -> tuple[TraceHeader, list[TraceStep]]`
  - `window_unions(steps: Sequence[TraceStep], window: int) -> list[tuple[int, ...]]`
  - `replay_cache(steps: Sequence[TraceStep], capacity_experts: int, logits_sha256: str) -> dict[str, object]`

- [ ] **Step 1: Add the failing schema/union kill tests**

Use exactly this hand-built trace:

```python
steps = [
    TraceStep("step", 0, 10, ((1, 2), (3, 4))),
    TraceStep("step", 1, 11, ((2, 5), (3, 6))),
    TraceStep("step", 2, 12, ((1, 5), (4, 6))),
]
```

Test: header plus steps round-trip exactly; `window_unions(steps, 2)` returns layer-encoded expert sets `((0,1),(0,2),(0,5),(1,3),(1,4),(1,6))` for positions 0–1 and `((0,1),(0,2),(0,5),(1,3),(1,4),(1,6))` for positions 1–2; the full three-position union has six layer/expert pairs. Also assert the per-layer unique-expert counts are `(3, 3)` for either two-position window and for all three positions. Layer identity must be part of every cache key.

Replay the same trace at capacities 2 and 12. Assert both results preserve `token_ids == [10, 11, 12]`, `trace_sha256`, and the supplied `logits_sha256`; assert hit/miss counts differ. Reject duplicate/non-contiguous `pos`, wrong layer count, wrong top-k length, duplicate expert IDs in one layer, or IDs outside `[0, n_expert)`.

- [ ] **Step 2: Run the tests and verify the trace import fails**

Run:

```powershell
./.venv/Scripts/python.exe tests/test_moe_trace.py
```

Expected: FAIL because trace symbols are missing.

- [ ] **Step 3: Implement strict JSONL and observational LRU replay**

Write one compact, sorted JSON object per UTF-8 line. The first line must have `kind="header"`; all remaining lines must have `kind="step"`. Keep nullable fields explicitly present. Do not add fields beyond the approved §8.5 schema; processed-token semantics are documented beside the schema in README/report. Atomic-write the complete file.

`replay_cache` is analysis-only: process each `(layer, expert)` request in router order through `collections.OrderedDict`; capacity is a count of expert blobs, `0` means every request misses. It returns request/hit/miss counts plus hashes and token IDs but never claims to execute model logits. The immutable logits hash is the proof that cache capacity was not an input to model execution.

- [ ] **Step 4: Run the kill tests**

Run:

```powershell
./.venv/Scripts/python.exe tests/test_moe_trace.py
```

Expected: PASS, including exact union answers.

- [ ] **Step 5: Commit Task 3**

```powershell
git add src/llm_lab/moe/trace.py tests/test_moe_trace.py
git commit -m @'
feat(moe): add strict routing trace format

Co-Authored-By: Claude Code <noreply@anthropic.com>
'@
```

### Task 4: Add Deterministic Tiny and HF Execution

**Files:**
- Modify: `src/llm_lab/moe/baseline.py`
- Modify: `src/llm_lab/moe/trace.py`
- Modify: `tests/test_moe_trace.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: `OlmoeConfig`, `OlmoeForCausalLM`, `AutoModelForCausalLM`, `AutoTokenizer`, `BitsAndBytesConfig`; Task 1 records; Task 3 trace writer.
- Produces:
  - `seed_everything(seed: int, threads: int) -> None`
  - `build_tiny_olmoe(seed: int) -> OlmoeForCausalLM`
  - `run_hf(model, input_ids, config: RunConfig, max_new_tokens: int) -> RunRecord`
  - `TraceCapture(run_record: RunRecord, trace_sha256: str, logits_sha256: str)`
  - `trace_hf(model, input_ids, header: TraceHeader, max_new_tokens: int, trace_path: Path, logits_path: Path, stop_at_eos: bool = True) -> TraceCapture`
  - `load_hf_int8(model_id: str, revision: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]`
  - `load_hf_bf16(model_id: str, revision: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]`
  - `sha256_tokenizer(tokenizer) -> str`, hashing tokenizer files listed by `tokenizer.init_kwargs` in sorted path order and refusing when no local tokenizer artifact can be identified.

- [ ] **Step 1: Add failing tiny determinism and trace-alignment tests**

Construct the tiny fixture locally with:

```python
OlmoeConfig(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    num_experts=8,
    num_experts_per_tok=2,
    max_position_embeddings=64,
    bos_token_id=1,
    eos_token_id=127,
    pad_token_id=0,
)
```

Use prompt IDs `[[1, 3, 4]]`, seed `0`, one thread, and three new tokens. Assert two freshly built models produce identical generated token IDs and oracle identity. Assert the trace contains three steps, each `tok_id` equals the generated token processed in a follow-up forward pass, each step has two layers and two experts per layer, and logits SHA-256 is stable across runs. Also compare a forward pass over one generated position with attention KV cache enabled and disabled using `torch.equal`; nondeterminism is a test failure, not a tolerance. This is a decoder-correctness check distinct from expert-cache replay.

- [ ] **Step 2: Run and verify missing execution functions**

Run:

```powershell
./.venv/Scripts/python.exe tests/test_moe_trace.py
```

Expected: FAIL on missing `build_tiny_olmoe`, `run_hf`, or `trace_hf`.

- [ ] **Step 3: Add only the dependencies required for M1b int8 loading**

Run:

```powershell
uv add bitsandbytes==0.50.2 accelerate==1.15.0
```

These versions have a Windows x86-64 CPU wheel/backend and were smoke-tested on the target Ryzen/CPU PyTorch environment. Do not add a test framework; the repository uses directly executable assert-based scripts.

- [ ] **Step 4: Implement deterministic HF generation and processed-token tracing**

Set `torch.manual_seed(seed)`, `torch.set_num_threads(threads)`, deterministic algorithms, and `model.eval()`. Generate greedily with `do_sample=False` and `use_cache=True`. For tracing, use one KV-cached decode loop:

1. Evaluate the full prompt to obtain the first generated token.
2. Feed that generated token with `past_key_values`; its router logits belong to that processed token.
3. Record `tok_id`, ordered `torch.topk(..., sorted=True).indices`, and raw next-token logits.
4. Repeat until `max_new_tokens` generated tokens have each been processed and traced; stop at EOS by default, but honor explicit `stop_at_eos=False` for fixed-length research traces and record that decode setting in the trace summary.

Write raw logits as a Torch tensor file under `artifacts/m1/raw/`, then hash its bytes. Measure `peak_rss_bytes` from Windows `psutil.Process().memory_info().peak_wset`; use `None` if the platform exposes no peak counter. Convert CUDA/allocator OOM exceptions to a clear `RuntimeError("OOM while loading/running <model>; no fallback attempted")`.

`load_hf_int8` must pass the pinned revision, `BitsAndBytesConfig(load_in_8bit=True)`, and `device_map="cpu"`; after loading, verify at least one `Linear8bitLt` exists or fail rather than mislabelelling precision. `load_hf_bf16` must pass the pinned revision, `dtype=torch.bfloat16`, and `device_map="cpu"`; it is the rented-host path and has no local/int8 fallback.

- [ ] **Step 5: Run M1a tests in the repository's supported style**

Run:

```powershell
./.venv/Scripts/python.exe tests/test_moe_trace.py
```

Expected: PASS in seconds, with no model download.

- [ ] **Step 6: Commit Task 4**

```powershell
git add pyproject.toml uv.lock src/llm_lab/moe/baseline.py src/llm_lab/moe/trace.py tests/test_moe_trace.py
git commit -m @'
feat(moe): trace deterministic HF routing

Co-Authored-By: Claude Code <noreply@anthropic.com>
'@
```

### Task 5: Add llama.cpp Baseline and Lean CLI

**Files:**
- Modify: `src/llm_lab/moe/baseline.py`
- Modify: `src/llm_lab/__init__.py`
- Modify: `tests/test_moe_trace.py`

**Interfaces:**
- Consumes: `llama_cpp.Llama`, Task 1 records/oracle, Task 2 metadata, Task 3 trace/replay.
- Produces:
  - `run_llama_cpp(model_path: Path, prompt: str, config: RunConfig, max_new_tokens: int, n_ctx: int) -> RunRecord`
  - CLI:
    - `llm-lab moe meta --backend hf|llama-cpp --model-path PATH --output FILE`
    - `llm-lab moe run --backend tiny|hf-int8|llama-cpp --model-path PATH --model-id ID --revision SHA --prompt TEXT --prompt-token-ids CSV --seed N --threads N --max-new-tokens N --n-ctx N --output FILE`
    - `llm-lab moe trace --backend tiny|hf-int8|hf-bf16 --model-id ID --revision SHA --prompt TEXT --prompt-token-ids CSV --seed N --threads N --max-new-tokens N --continue-after-eos --trace FILE --logits FILE --run-record FILE --summary FILE`
    - `llm-lab moe compare RUN_A RUN_B --output FILE`
    - `llm-lab moe replay-cache TRACE --logits FILE --capacity-experts N --output FILE`

For `tiny`, `--prompt-token-ids` is required and text/model arguments are rejected. For HF/llama.cpp, `--prompt` is required and token IDs are rejected; llama.cpp requires `--model-path`, HF requires `--model-id` plus a 40-character revision. `--n-ctx` is llama.cpp-only. `--continue-after-eos` is trace-only and must be reflected as `stop_at_eos: false` in the compact summary; it does not change the approved header schema. All output paths are required.

- [ ] **Step 1: Add failing CLI and mocked llama.cpp tests**

Replace the module-level `Llama` factory temporarily in the test with a fake exposing `tokenize`, `generate`, `detokenize`, and metadata; restore it in `finally`. Assert `run_llama_cpp` passes `n_gpu_layers=0`, `seed=0`, `n_threads=1`, `verbose=False`, and greedy generator parameters (`temp=0.0`, sampling filters disabled, penalties neutral). Assert it stores the exact integer IDs yielded by `generate`, not IDs recovered by re-tokenizing decoded text.

Call `main([...])` directly for `moe compare` and `moe replay-cache`; assert JSON outputs match library calls. Call `main(["moe", "run", "--backend", "llama-cpp", "--model-path", missing])`; assert `SystemExit` is non-zero and stderr contains the exact missing path.

- [ ] **Step 2: Run and verify failures**

Run:

```powershell
./.venv/Scripts/python.exe tests/test_moe_trace.py
```

Expected: FAIL because llama.cpp runner and CLI arguments are absent.

- [ ] **Step 3: Implement direct token-ID generation and measured telemetry**

Instantiate `Llama` with CPU-only settings, tokenize the UTF-8 prompt once, then consume `Llama.generate(prompt_ids, temp=0.0, top_k=1, top_p=1.0, min_p=0.0, typical_p=1.0, repeat_penalty=1.0, frequency_penalty=0.0, presence_penalty=0.0)`. Stop at EOS or the requested count. Time only autoregressive decode, and compute `tok_per_s = n_generated / wall_s`; do not count prompt tokens. Peak RSS uses `peak_wset` as in HF.

Build the fingerprint from exact model path, quant read by `meta.py`, SHA-256 of canonical compact JSON prompt token IDs, tokenizer artifact hash, seed, greedy mode, threads, and backend. The CLI prints the same JSON it writes, so PowerShell `Tee-Object` can preserve exact output.

Change `main()` to `main(argv: list[str] | None = None)` and call `parser.parse_args(argv)` so CLI tests need no subprocess. Catch only anticipated file/config/OOM/comparability errors at the top level, print one error, and exit non-zero; unexpected exceptions retain tracebacks.

- [ ] **Step 4: Run M1 tests and the CLI smoke checks**

Run:

```powershell
./.venv/Scripts/python.exe tests/test_moe_trace.py
./.venv/Scripts/llm-lab.exe moe --help
```

Expected: tests PASS; help lists exactly `meta`, `run`, `trace`, `compare`, and `replay-cache`.

- [ ] **Step 5: Commit Task 5**

```powershell
git add src/llm_lab/moe/baseline.py src/llm_lab/__init__.py tests/test_moe_trace.py
git commit -m @'
feat(moe): expose deterministic reference CLI

Co-Authored-By: Claude Code <noreply@anthropic.com>
'@
```

### Task 6: Prove M1a and Establish Artifact Discipline

**Files:**
- Modify: `.gitignore`
- Modify: `src/llm_lab/__init__.py`
- Modify: `tests/test_moe_trace.py`
- Create: `artifacts/m1/evidence/m1a-commands.ps1`
- Create: generated `artifacts/m1/evidence/m1a-test-output.txt`
- Create: generated `artifacts/m1/evidence/m1a-run-a.json`
- Create: generated `artifacts/m1/evidence/m1a-run-b.json`
- Create: generated `artifacts/m1/evidence/m1a-oracle.json`
- Create: generated `artifacts/m1/evidence/m1a-trace-summary.json`
- Create: generated `artifacts/m1/evidence/m1a-files.sha256`

**Interfaces:**
- Consumes: Tasks 1–5 CLI.
- Produces: committed, rerunnable evidence that satisfies M1a definition of done.

- [ ] **Step 1: Add failing artifact-policy tests**

Read `.gitignore` in the test and assert it contains exact lines `/models/m1/` and `/artifacts/m1/raw/`, but does not ignore `/artifacts/m1/evidence/`. Add a CLI test that refuses to overwrite an existing evidence file unless `--force` is passed; this prevents accidental replacement of measurements.

- [ ] **Step 2: Run and verify policy tests fail**

Run:

```powershell
./.venv/Scripts/python.exe tests/test_moe_trace.py
```

Expected: FAIL because ignore rules and overwrite guard do not exist.

- [ ] **Step 3: Implement ignore rules and overwrite protection**

Append:

```gitignore
# M1 local checkpoints and large raw evidence
/models/m1/
/artifacts/m1/raw/
```

All output-writing CLI commands reject existing paths by default. `--force` is explicit and is recorded in the command transcript when used. Phase scripts may instead delete only their own enumerated prior outputs before execution; they never clear an artifact directory wholesale.

- [ ] **Step 4: Write the exact M1a evidence script**

`m1a-commands.ps1` must use `$ErrorActionPreference = "Stop"`, remove only its own prior generated outputs so reruns are explicit and deterministic, create only the two artifact directories, then run:

```powershell
./.venv/Scripts/python.exe tests/test_moe_trace.py 2>&1 |
  Tee-Object artifacts/m1/evidence/m1a-test-output.txt
./.venv/Scripts/llm-lab.exe moe run --backend tiny --seed 0 --threads 1 `
  --prompt-token-ids 1,3,4 --max-new-tokens 32 `
  --output artifacts/m1/evidence/m1a-run-a.json
./.venv/Scripts/llm-lab.exe moe run --backend tiny --seed 0 --threads 1 `
  --prompt-token-ids 1,3,4 --max-new-tokens 32 `
  --output artifacts/m1/evidence/m1a-run-b.json
./.venv/Scripts/llm-lab.exe moe compare `
  artifacts/m1/evidence/m1a-run-a.json artifacts/m1/evidence/m1a-run-b.json `
  --output artifacts/m1/evidence/m1a-oracle.json
./.venv/Scripts/llm-lab.exe moe trace --backend tiny --seed 0 --threads 1 `
  --prompt-token-ids 1,3,4 --max-new-tokens 32 `
  --trace artifacts/m1/raw/m1a-trace.jsonl `
  --logits artifacts/m1/raw/m1a-logits.pt `
  --run-record artifacts/m1/raw/m1a-trace-run.json `
  --summary artifacts/m1/evidence/m1a-trace-summary.json
Get-FileHash artifacts/m1/raw/m1a-trace.jsonl,artifacts/m1/raw/m1a-logits.pt `
  -Algorithm SHA256 | Format-Table -AutoSize | Out-File `
  artifacts/m1/evidence/m1a-files.sha256 -Encoding utf8
```

- [ ] **Step 5: Execute M1a and inspect generated facts**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File artifacts/m1/evidence/m1a-commands.ps1
```

Expected: test output ends in PASS; oracle has `identical: true`, `first_divergence: null`, `fingerprints_match: true`; summary reports 32 generated/traced positions, two layers, eight experts, top-2.

- [ ] **Step 6: Commit M1a evidence**

```powershell
git add .gitignore src/llm_lab/__init__.py tests/test_moe_trace.py artifacts/m1/evidence/m1a-commands.ps1 artifacts/m1/evidence/m1a-test-output.txt artifacts/m1/evidence/m1a-run-a.json artifacts/m1/evidence/m1a-run-b.json artifacts/m1/evidence/m1a-oracle.json artifacts/m1/evidence/m1a-trace-summary.json artifacts/m1/evidence/m1a-files.sha256
git commit -m @'
test(moe): prove M1a deterministic plumbing

Co-Authored-By: Claude Code <noreply@anthropic.com>
'@
```

### Task 7: Run the OLMoE End-to-End Gate Before Qwen3

**Files:**
- Create: `artifacts/m1/evidence/m1b-commands.ps1`
- Create: generated `artifacts/m1/evidence/m1b-preflight.txt`
- Create: generated `artifacts/m1/evidence/m1b-output.txt`
- Create: generated `artifacts/m1/evidence/m1b-gguf-meta.json`
- Create: generated `artifacts/m1/evidence/m1b-gguf-run-a.json`
- Create: generated `artifacts/m1/evidence/m1b-gguf-run-b.json`
- Create: generated `artifacts/m1/evidence/m1b-gguf-oracle.json`
- Create: generated `artifacts/m1/evidence/m1b-hf-run-a.json`
- Create: generated `artifacts/m1/evidence/m1b-hf-run-b.json`
- Create: generated `artifacts/m1/evidence/m1b-hf-oracle.json`
- Create: generated `artifacts/m1/evidence/m1b-trace-prose-a-summary.json`
- Create: generated `artifacts/m1/evidence/m1b-trace-prose-b-summary.json`
- Create: generated `artifacts/m1/evidence/m1b-trace-code-a-summary.json`
- Create: generated `artifacts/m1/evidence/m1b-trace-code-b-summary.json`
- Create: generated `artifacts/m1/evidence/m1b-trace-factual-summary.json`
- Create: generated `artifacts/m1/evidence/m1b-trace-summary.json`
- Create: generated `artifacts/m1/evidence/m1b-cache-64.json`
- Create: generated `artifacts/m1/evidence/m1b-cache-512.json`
- Create: generated `artifacts/m1/evidence/m1b-files.sha256`
- Create: generated `artifacts/m1/evidence/m1b-gate.json`

**Interfaces:**
- Consumes: official OLMoE GGUF and HF revisions; all CLI commands.
- Produces: `m1b-gate.json` with booleans `gguf_deterministic`, `hf_int8_deterministic`, `positions_at_least_500`, `cache_observational_only`, `raw_hashes_present`, and `passed`; Task 8 may proceed only if `passed` is true.

- [ ] **Step 1: Write preflight and download commands with exact revisions**

The script first records `Get-PSDrive`, `psutil.virtual_memory()`, package versions, CPU name, and free disk. Calculate the exact requirement before any transfer: official GGUF bytes + official HF repository bytes + 4 GiB working headroom, minus already-present file bytes. Abort while naming required and available bytes when the requirement is not met. Download only:

```powershell
./.venv/Scripts/hf.exe download allenai/OLMoE-1B-7B-0924-GGUF `
  olmoe-1b-7b-0924-q4_k_m.gguf `
  --revision 70df85ed7132bf21b5acbcb33817f58e7d3cb949 `
  --local-dir models/m1/olmoe-gguf
```

The HF int8 loader downloads the pinned BF16 shards into the Hugging Face cache before quantizing in memory. The script obtains exact remote byte totals with `hf download ... --dry-run --json`, saves that JSON in the preflight transcript, and uses it in the disk calculation; no rounded size becomes a gate input. Never download Qwen in this task.

- [ ] **Step 2: Run the OLMoE GGUF metadata and deterministic baseline**

Use only the baseline prompt `"The capital of France is"`, `seed=0`, `threads=6`, `n_ctx=1024`, and `max_new_tokens=64`. Run `moe meta`, run `moe run` twice, then `moe compare`. Expected: measured GGUF metadata; two token-identical Q4_K_M records; oracle identity true. Do not compare these token IDs with HF int8 output.

- [ ] **Step 3: Run the OLMoE HF-int8 determinism check**

Use the same baseline prompt twice through `--backend hf-int8`, pinned model/revision, `seed=0`, `threads=6`, and `max_new_tokens=64`. Expected: two comparable int8 records and oracle identity true. Header/config must say `INT8_BITSANDBYTES`, never BF16.

- [ ] **Step 4: Capture exactly 500 processed generated positions**

Use five literal prompts embedded in the script: two prose, two code, and one factual-recall prompt. Request exactly 100 generated positions per prompt with `eos_token_id=None` supplied only for trace capture, so the fixed workload is 500 positions without padding or data-dependent extra prompts. Each trace file has its own header/prompt SHA, raw logits file, and compact per-trace summary. The script derives `m1b-trace-summary.json` by summing exactly five 100-position summaries and listing each source-summary hash without merging unlike trace headers. If any trace contains other than 100 steps, the gate fails.

- [ ] **Step 5: Replay two expert-cache capacities observationally**

Replay each immutable trace with capacities 64 and 512 expert blobs. For every replay pair, assert identical trace SHA, logits SHA, and token IDs while hit/miss counts differ for at least one trace. The evidence wording must say:

```text
Cache replay did not execute or alter model logits. Identical logits means both
capacity analyses reference the same content-addressed logits artifact.
```

- [ ] **Step 6: Generate and enforce the M1b gate**

Have the script derive `m1b-gate.json` from the outputs, never from hardcoded `true` values. Run the script while capturing the complete stream:

```powershell
powershell -ExecutionPolicy Bypass -File artifacts/m1/evidence/m1b-commands.ps1 2>&1 |
  Tee-Object artifacts/m1/evidence/m1b-output.txt
```

Expected: `passed: true`; exactly 500 positions; both within-path oracles identical; cache replay hashes stable. If int8 load OOMs, preserve `m1b-output.txt` and preflight as a finding and stop—do not download Qwen or substitute Qwen1.5-MoE-A2.7B.

- [ ] **Step 7: Run regression tests after the real-model gate**

Run:

```powershell
./.venv/Scripts/python.exe tests/test_moe_trace.py
```

Expected: PASS.

- [ ] **Step 8: Commit M1b commands and compact evidence**

```powershell
git add artifacts/m1/evidence/m1b-commands.ps1 artifacts/m1/evidence/m1b-preflight.txt artifacts/m1/evidence/m1b-output.txt artifacts/m1/evidence/m1b-gguf-meta.json artifacts/m1/evidence/m1b-gguf-run-a.json artifacts/m1/evidence/m1b-gguf-run-b.json artifacts/m1/evidence/m1b-gguf-oracle.json artifacts/m1/evidence/m1b-hf-run-a.json artifacts/m1/evidence/m1b-hf-run-b.json artifacts/m1/evidence/m1b-hf-oracle.json artifacts/m1/evidence/m1b-trace-prose-a-summary.json artifacts/m1/evidence/m1b-trace-prose-b-summary.json artifacts/m1/evidence/m1b-trace-code-a-summary.json artifacts/m1/evidence/m1b-trace-code-b-summary.json artifacts/m1/evidence/m1b-trace-factual-summary.json artifacts/m1/evidence/m1b-trace-summary.json artifacts/m1/evidence/m1b-cache-64.json artifacts/m1/evidence/m1b-cache-512.json artifacts/m1/evidence/m1b-files.sha256 artifacts/m1/evidence/m1b-gate.json
git commit -m @'
test(moe): validate M1b on OLMoE

Co-Authored-By: Claude Code <noreply@anthropic.com>
'@
```

### Task 8: Execute Gated Qwen3 M1c on Local and Rented Hosts

**Files:**
- Create: `artifacts/m1/evidence/m1c-local-commands.ps1`
- Create: `artifacts/m1/evidence/m1c-rented-commands.ps1`
- Create: generated `artifacts/m1/evidence/m1c-local-preflight.txt`
- Create: generated `artifacts/m1/evidence/m1c-local-output.txt`
- Create: generated `artifacts/m1/evidence/m1c-rented-output.txt`
- Create: generated `artifacts/m1/evidence/m1c-meta.json`
- Create: generated `artifacts/m1/evidence/m1c-run-a.json`
- Create: generated `artifacts/m1/evidence/m1c-run-b.json`
- Create: generated `artifacts/m1/evidence/m1c-oracle.json`
- Create: generated `artifacts/m1/evidence/m1c-trace-prose-a-summary.json`
- Create: generated `artifacts/m1/evidence/m1c-trace-prose-b-summary.json`
- Create: generated `artifacts/m1/evidence/m1c-trace-code-summary.json`
- Create: generated `artifacts/m1/evidence/m1c-trace-factual-summary.json`
- Create: generated `artifacts/m1/evidence/m1c-trace-summary.json`
- Create: generated `artifacts/m1/evidence/m1c-files.sha256`
- Create: generated `artifacts/m1/evidence/m1c-gate.json`

**Interfaces:**
- Consumes: `artifacts/m1/evidence/m1b-gate.json` with `passed=true`; pinned Qwen revisions.
- Produces: local Q4 baseline evidence, rented BF16 route evidence, and an M1c gate covering all spec §8.11 criteria.

- [ ] **Step 1: Make the local script refuse to run unless M1b passed**

Read and validate `m1b-gate.json` first. Preflight exact free disk with `shutil.disk_usage`; require at least `18,556,685,824 + 10 * 2**30` free bytes before download. If not, exit non-zero naming required and available bytes.

- [ ] **Step 2: Download only the approved Q4_K_M file**

```powershell
./.venv/Scripts/hf.exe download Qwen/Qwen3-30B-A3B-GGUF `
  Qwen3-30B-A3B-Q4_K_M.gguf `
  --revision e4d4bafdfb96a411a163846265362aceb0b9c63a `
  --local-dir models/m1/qwen3-gguf
```

Hash the downloaded file and record its exact byte count before loading it. Abort on mismatch.

- [ ] **Step 3: Measure metadata and deterministic local Q4 baseline**

Run `moe meta` and two `moe run` invocations using the same literal prompt, `seed=0`, `threads=6`, `n_ctx=2048`, and `max_new_tokens=256`; compare them. Expected: a real generation, measured wall time/tok/s/peak RSS, oracle identity, and metadata confirming or correcting layer/expert/expert-byte facts. Never copy §3 estimates into the result.

- [ ] **Step 4: Prepare the rented-host BF16 script without local execution**

The rented script runs `hf download ... --dry-run --json` first, records the exact remote byte total, and requires `remote_bytes + 16 * 2**30` available disk plus `80 * 2**30` available RAM before downloading HF revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`. It installs the project from the committed lock, records host/package details, and runs `--backend hf-bf16`; no int8 fallback is permitted.

Embed exactly four literal prompts—two prose, one code, one factual recall—and trace exactly 500 processed generated positions per prompt with EOS stopping disabled only for trace capture. Save raw traces/logits outside Git. Emit one compact summary per trace plus an aggregate `m1c-trace-summary.json` containing their hashes and an exact summed count of 2000; any other count fails the gate.

- [ ] **Step 5: Run the local script, then the rented script**

Run locally:

```powershell
powershell -ExecutionPolicy Bypass -File artifacts/m1/evidence/m1c-local-commands.ps1 2>&1 |
  Tee-Object artifacts/m1/evidence/m1c-local-output.txt
```

Copy the repository commit to the rented Windows host and run:

```powershell
powershell -ExecutionPolicy Bypass -File artifacts/m1/evidence/m1c-rented-commands.ps1 2>&1 |
  Tee-Object artifacts/m1/evidence/m1c-rented-output.txt
```

Expected: local Q4 oracle identity true; rented trace aggregate has exactly 2000 positions across three domains and explicitly records `BF16`.

- [ ] **Step 6: Return only compact rented evidence and verify hashes**

Copy back the generated trace summary, command output, and raw-file SHA/byte manifest. Do not copy raw traces into Git. Verify copied summary/manifests against the rented transcript, then derive `m1c-gate.json` with booleans for real local generation, Q4 determinism, 2000 positions, three domains, BF16 precision, measured metadata, and complete evidence hashes.

- [ ] **Step 7: Commit M1c evidence only when all gates pass**

```powershell
git add artifacts/m1/evidence/m1c-local-commands.ps1 artifacts/m1/evidence/m1c-rented-commands.ps1 artifacts/m1/evidence/m1c-local-preflight.txt artifacts/m1/evidence/m1c-local-output.txt artifacts/m1/evidence/m1c-rented-output.txt artifacts/m1/evidence/m1c-meta.json artifacts/m1/evidence/m1c-run-a.json artifacts/m1/evidence/m1c-run-b.json artifacts/m1/evidence/m1c-oracle.json artifacts/m1/evidence/m1c-trace-prose-a-summary.json artifacts/m1/evidence/m1c-trace-prose-b-summary.json artifacts/m1/evidence/m1c-trace-code-summary.json artifacts/m1/evidence/m1c-trace-factual-summary.json artifacts/m1/evidence/m1c-trace-summary.json artifacts/m1/evidence/m1c-files.sha256 artifacts/m1/evidence/m1c-gate.json
git commit -m @'
test(moe): capture Qwen3 M1c reference evidence

Co-Authored-By: Claude Code <noreply@anthropic.com>
'@
```

If any gate fails, commit only the truthful failure transcript in a separately named evidence file and leave M1c incomplete.

### Task 9: Document the Measured M1 Result and Run Final Verification

**Files:**
- Modify: `README.md`
- Modify: `src/llm_lab/__init__.py`
- Modify: `tests/test_moe_trace.py`
- Create: `artifacts/m1/evidence/M1-REPORT.md`

**Interfaces:**
- Consumes: generated M1a/M1b/M1c gate files and evidence only.
- Produces: public commands and an M1 report with no unsupported numbers.

- [ ] **Step 1: Add a failing CLI inventory assertion**

Capture `main(["moe", "--help"])` output and assert it names all five M1 commands and does not contain `speedup`, `PageCC`, `GPU`, or `lossless to BF16`. This protects the public scope boundary.

- [ ] **Step 2: Run and verify the documentation-facing test fails if help is incomplete**

Run:

```powershell
./.venv/Scripts/python.exe tests/test_moe_trace.py
```

Expected: FAIL only if command help or forbidden wording needs correction.

- [ ] **Step 3: Write README and M1 report from evidence**

Document phase order, exact model/revision/precision/backend names, command examples, raw-versus-committed artifact policy, processed-token trace semantics, and the oracle's fingerprint refusal. Populate every measurement by copying from committed JSON/transcript evidence and cite the exact artifact path beside it. Required exactness phrasing for the local target:

```text
bit-identical to Qwen3-30B-A3B-Q4_K_M under llama.cpp greedy decoding,
seed 0, 6 threads, on the recorded host
```

State explicitly that HF BF16 routing and GGUF Q4 decoding are different numeric paths and were not asserted equal. Include the SSD energy caveat from spec §3 in any public performance discussion.

- [ ] **Step 4: Run complete validation**

Run:

```powershell
uv sync --frozen
./.venv/Scripts/python.exe tests/test_moe_trace.py
./.venv/Scripts/llm-lab.exe moe --help
./.venv/Scripts/llm-lab.exe moe compare `
  artifacts/m1/evidence/m1c-run-a.json artifacts/m1/evidence/m1c-run-b.json
```

Expected: frozen sync succeeds; tests pass; help lists five commands; final oracle says `identical: true` and `fingerprints_match: true`.

- [ ] **Step 5: Audit reporting invariants**

Search the changed README/report/evidence for `PROJECTED`, `speedup`, `lossless`, `.strip()`, `Q4`, `BF16`, `null`, and numeric units. For every measurement, point to its command transcript and output; remove any untraceable number. Confirm `git status --short` contains no model files or raw traces.

- [ ] **Step 6: Commit documentation and final verification**

```powershell
git add README.md src/llm_lab/__init__.py tests/test_moe_trace.py artifacts/m1/evidence/M1-REPORT.md
git commit -m @'
docs(moe): report measured M1 reference results

Co-Authored-By: Claude Code <noreply@anthropic.com>
'@
```

## Self-Review Results

- **Spec coverage:** Tasks 1–6 cover all M1a components, RunRecord, oracle, schema, union kill-test, deterministic tiny generation, error handling, and CLI wiring. Task 7 adds the approved small real MoE and validates both numeric paths before Qwen. Task 8 covers every M1c §8.11 criterion. Task 9 enforces reporting rules and public wording.
- **Approved adjustments:** OLMoE—not Qwen1.5-MoE-A2.7B—is used because it is smaller. Its official 3.92 GiB Q4_K_M GGUF exercises llama.cpp locally, while the pinned BF16 repository is loaded through verified Windows CPU int8 support for routing. No cross-quantization exactness is claimed.
- **Resolved ambiguities:** `tok_id` is the processed generated token; cache replay is explicitly observational and binds both capacities to one content-addressed logits artifact; large raw data is ignored while summaries/transcripts/hashes are committed.
- **Scope:** One M1 plan is appropriate because each phase gates the next and all phases reuse the same four modules. M2+ remains excluded.
