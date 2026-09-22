"""
Unit Tests for the MoE reference pipeline (M1).
Validates:
1. RunRecord JSON round-trip and stable config fingerprints.
2. Record/config validation: every rejection path is a loud ValueError.
3. Canonical hashing of token IDs and files.
4. The exactness oracle: identity, first divergence, and fingerprint refusal.
"""

from pathlib import Path
import tempfile

from llm_lab.moe.baseline import RunConfig, RunRecord, fingerprint_config
from llm_lab.moe.oracle import compare_runs

from dataclasses import replace
from llm_lab.moe.baseline import sha256_bytes, sha256_file, sha256_token_ids


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


def assert_rejects(build, field: str) -> None:
    """A rejected construction must raise ValueError naming the offending field."""
    try:
        build()
    except ValueError as exc:
        assert field in str(exc), f"error for {field} did not name it: {exc}"
    else:
        raise AssertionError(f"accepted invalid {field}")


def test_run_config_rejects_invalid_values():
    assert_rejects(lambda: replace(sample_config(), decode="sampling"), "decode")
    assert_rejects(lambda: replace(sample_config(), seed=-1), "seed")
    assert_rejects(lambda: replace(sample_config(), threads=-1), "threads")
    assert_rejects(lambda: replace(sample_config(), prompt_sha="a" * 63), "prompt_sha")
    assert_rejects(lambda: replace(sample_config(), prompt_sha=None), "prompt_sha")
    assert_rejects(lambda: replace(sample_config(), tokenizer_sha="short"), "tokenizer_sha")


def test_run_record_rejects_invalid_values():
    assert_rejects(
        lambda: replace(sample_record(), config_fingerprint="b" * 63), "config_fingerprint"
    )
    assert_rejects(lambda: replace(sample_record(), n_prompt=-1), "n_prompt")
    assert_rejects(lambda: replace(sample_record(), n_generated=-1), "n_generated")
    assert_rejects(lambda: replace(sample_record(), n_generated=None), "n_generated")
    assert_rejects(lambda: replace(sample_record(), wall_s=-0.5), "wall_s")
    assert_rejects(lambda: replace(sample_record(), tok_per_s=-1.0), "tok_per_s")
    assert_rejects(lambda: replace(sample_record(), peak_rss_bytes=-1), "peak_rss_bytes")


def test_run_record_normalizes_token_ids_to_tuple():
    record = sample_record([7, 8, 9])
    assert record.token_ids == (7, 8, 9)
    assert record == sample_record((7, 8, 9))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "run.json"
        record.write(path)
        assert RunRecord.read(path) == record


def test_token_id_and_file_hashes_are_canonical():
    assert sha256_token_ids([7, 8, 9]) == sha256_bytes(b"[7,8,9]")
    assert sha256_token_ids((7, 8, 9)) == sha256_token_ids([7, 8, 9])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "blob.bin"
        path.write_bytes(b"abc")
        assert sha256_file(path) == sha256_bytes(b"abc")


import json

import gguf
import numpy as np

from llm_lab.moe.meta import ModelMeta, read_gguf_meta, read_hf_meta, read_meta

# The brief's local OLMoE-shaped config; every number below is derived from these.
HF_CONFIG = {
    "model_type": "olmoe",
    "hidden_size": 64,
    "intermediate_size": 32,
    "num_hidden_layers": 2,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "torch_dtype": "float32",
}

# Matching GGUF metadata. Keys without a dot get the architecture prefix when written.
GGUF_KV = {
    "block_count": 2,
    "expert_count": 8,
    "expert_used_count": 2,
    "expert_feed_forward_length": 32,
    "general.file_type": int(gguf.LlamaFileType.ALL_F32),
}


def write_hf_config(directory, **overrides) -> Path:
    """Write a local config.json. An override of None drops that key, to test strictness."""
    config = {**HF_CONFIG, **overrides}
    path = Path(directory) / "config.json"
    payload = {k: v for k, v in config.items() if v is not None}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def expert_tensors(n_layer: int, n_expert: int, hidden: int, ffn: int) -> dict:
    """float32 gate/up/down expert stacks per layer, shaped as llama.cpp exports them."""
    shapes = {"gate": (ffn, hidden), "up": (ffn, hidden), "down": (hidden, ffn)}
    return {
        f"blk.{layer}.ffn_{part}_exps.weight": np.zeros((n_expert, *shape), dtype=np.float32)
        for layer in range(n_layer)
        for part, shape in shapes.items()
    }


def write_gguf(path: Path, kv: dict, tensors: dict, *, arch="olmoe", raw_dtype=None) -> Path:
    """Write a GGUF holding exactly the given scalars and tensors, so omissions are testable."""
    writer = gguf.GGUFWriter(path, arch=arch)
    for key, value in kv.items():
        writer.add_uint32(key if "." in key else f"{arch}.{key}", value)
    for name, array in tensors.items():
        writer.add_tensor(name, array, raw_dtype=raw_dtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def assert_missing_path(build, path) -> None:
    """A missing checkpoint must raise FileNotFoundError naming the exact path."""
    try:
        build()
    except FileNotFoundError as exc:
        assert str(path) in str(exc), f"error did not name {path}: {exc}"
    else:
        raise AssertionError(f"accepted missing path {path}")


def test_hf_meta_reads_only_config_facts():
    with tempfile.TemporaryDirectory() as tmp:
        config_path = write_hf_config(tmp)
        meta = read_hf_meta(config_path)
        assert meta.model == "olmoe"
        assert meta.path == str(config_path)
        assert meta.quant == "float32"
        assert (meta.n_layer, meta.n_expert, meta.n_expert_used) == (2, 8, 2)
        assert meta.expert_bytes == 3 * 64 * 32 * 4
        assert meta.total_expert_bytes == 2 * 8 * meta.expert_bytes
        # Unmeasurable fields must survive as JSON null, so the trace header stays serializable.
        assert json.loads(json.dumps(meta.to_dict())) == meta.to_dict()


def test_hf_meta_prefers_the_moe_expert_width():
    # Qwen3-MoE carries both keys and only moe_intermediate_size sizes an expert.
    with tempfile.TemporaryDirectory() as tmp:
        meta = read_hf_meta(
            write_hf_config(tmp, model_type="qwen3_moe", moe_intermediate_size=16)
        )
        assert meta.model == "qwen3_moe"
        assert meta.expert_bytes == 3 * 64 * 16 * 4


def test_hf_meta_sizes_experts_only_from_a_known_dtype():
    with tempfile.TemporaryDirectory() as tmp:
        assert read_hf_meta(write_hf_config(tmp, torch_dtype="bfloat16")).expert_bytes == (
            3 * 64 * 32 * 2
        )
        assert read_hf_meta(write_hf_config(tmp, torch_dtype="float16")).expert_bytes == (
            3 * 64 * 32 * 2
        )
        for dtype in (None, "float8_e4m3fn"):
            meta = read_hf_meta(write_hf_config(tmp, torch_dtype=dtype))
            assert meta.expert_bytes is None, f"invented expert_bytes for dtype {dtype!r}"
            assert meta.total_expert_bytes is None
            assert meta.quant == dtype
            # Counts still come straight from the file; only the byte formula goes silent.
            assert (meta.n_layer, meta.n_expert, meta.n_expert_used) == (2, 8, 2)


def test_hf_meta_rejects_missing_keys_and_paths():
    with tempfile.TemporaryDirectory() as tmp:
        for key in (
            "model_type",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_experts",
            "num_experts_per_tok",
        ):
            assert_rejects(
                lambda k=key: read_hf_meta(write_hf_config(tmp, **{k: None})), key
            )
        missing = Path(tmp) / "absent" / "config.json"
        assert_missing_path(lambda: read_hf_meta(missing), missing)
        # A JSON `true` must not pass as 1 for a count field (bool is an int subclass).
        for key in ("num_hidden_layers", "num_experts", "num_experts_per_tok"):
            assert_rejects(
                lambda k=key: read_hf_meta(write_hf_config(tmp, **{k: True})), key
            )
        # The error names the alias that actually held the bad value, not just the last key.
        assert_rejects(
            lambda: read_hf_meta(write_hf_config(tmp, moe_intermediate_size="x")),
            "moe_intermediate_size",
        )


def test_gguf_meta_measures_expert_bytes_from_real_tensor_sizes():
    with tempfile.TemporaryDirectory() as tmp:
        path = write_gguf(Path(tmp) / "olmoe.gguf", GGUF_KV, expert_tensors(2, 8, 64, 32))
        meta = read_gguf_meta(path)
        assert meta.model == "olmoe"
        assert meta.path == str(path)
        assert meta.quant == "ALL_F32"
        assert (meta.n_layer, meta.n_expert, meta.n_expert_used) == (2, 8, 2)
        # Three tensors of 8*32*64*4 bytes, each holding all 8 experts.
        assert meta.expert_bytes == 3 * (8 * 32 * 64 * 4 // 8)
        assert meta.total_expert_bytes == 2 * 8 * meta.expert_bytes
        assert json.loads(json.dumps(meta.to_dict())) == meta.to_dict()


def test_gguf_meta_reads_without_expert_feed_forward_length():
    # The real OLMoE Q4_K_M GGUF omits expert_feed_forward_length; expert bytes come from tensors.
    kv = {k: v for k, v in GGUF_KV.items() if k != "expert_feed_forward_length"}
    with tempfile.TemporaryDirectory() as tmp:
        path = write_gguf(Path(tmp) / "no_ffn.gguf", kv, expert_tensors(2, 8, 64, 32))
        meta = read_gguf_meta(path)
        assert (meta.n_layer, meta.n_expert, meta.n_expert_used) == (2, 8, 2)
        assert meta.expert_bytes == 3 * (8 * 32 * 64 * 4 // 8)
        assert meta.total_expert_bytes == 2 * 8 * meta.expert_bytes


def test_gguf_meta_ignores_nominal_bits_per_weight():
    # Q8_0 packs 32 weights into a 34-byte block, so an 8-bit-per-weight estimate under-counts.
    n_expert, rows, block_bytes, block_weights = 4, 2, 34, 32
    kv = {
        "block_count": 1,
        "expert_count": n_expert,
        "expert_used_count": 2,
        "expert_feed_forward_length": 32,
        "general.file_type": int(gguf.LlamaFileType.MOSTLY_Q8_0),
    }
    tensors = {
        f"blk.0.ffn_{part}_exps.weight": np.zeros(
            (n_expert, rows, block_bytes), dtype=np.uint8
        )
        for part in ("gate", "up", "down")
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = write_gguf(
            Path(tmp) / "q8.gguf", kv, tensors, raw_dtype=gguf.GGMLQuantizationType.Q8_0
        )
        meta = read_gguf_meta(path)
        assert meta.quant == "Q8_0"
        measured = 3 * (n_expert * rows * block_bytes // n_expert)
        nominal = 3 * (n_expert * rows * block_weights // n_expert)
        assert measured != nominal, "fixture cannot tell measured bytes from a bpw estimate"
        assert meta.expert_bytes == measured
        assert meta.total_expert_bytes == 1 * n_expert * measured


def test_gguf_meta_rejects_missing_keys_bad_layout_and_paths():
    with tempfile.TemporaryDirectory() as tmp:
        experts = expert_tensors(2, 8, 64, 32)
        for key in (
            "block_count",
            "expert_count",
            "expert_used_count",
        ):
            partial = {k: v for k, v in GGUF_KV.items() if k != key}
            assert_rejects(
                lambda kv=partial, k=key: read_gguf_meta(
                    write_gguf(Path(tmp) / f"no_{k}.gguf", kv, experts)
                ),
                key,
            )

        truncated = {n: a for n, a in experts.items() if n != "blk.1.ffn_down_exps.weight"}
        assert_rejects(
            lambda: read_gguf_meta(write_gguf(Path(tmp) / "short.gguf", GGUF_KV, truncated)),
            "blk.1.ffn_down_exps.weight",
        )

        indivisible = {
            f"blk.0.ffn_{part}_exps.weight": np.zeros((2, 2, 2), dtype=np.float32)
            for part in ("gate", "up", "down")
        }
        assert_rejects(
            lambda: read_gguf_meta(
                write_gguf(
                    Path(tmp) / "indivisible.gguf",
                    {**GGUF_KV, "block_count": 1, "expert_count": 3},
                    indivisible,
                )
            ),
            "not divisible",
        )

        missing = Path(tmp) / "absent.gguf"
        assert_missing_path(lambda: read_gguf_meta(missing), missing)


def test_read_meta_dispatches_on_backend():
    with tempfile.TemporaryDirectory() as tmp:
        config_path = write_hf_config(tmp)
        model_path = write_gguf(Path(tmp) / "olmoe.gguf", GGUF_KV, expert_tensors(2, 8, 64, 32))
        assert read_meta(config_path, "hf") == read_hf_meta(config_path)
        assert read_meta(model_path, "llama-cpp") == read_gguf_meta(model_path)
        assert_rejects(lambda: read_meta(config_path, "vllm"), "backend")


def test_model_meta_rejects_impossible_counts():
    meta = ModelMeta("olmoe", "x.gguf", None, 2, 8, 2, 96, 1536)
    assert_rejects(lambda: replace(meta, n_layer=0), "n_layer")
    assert_rejects(lambda: replace(meta, n_expert=0), "n_expert")
    assert_rejects(lambda: replace(meta, n_expert_used=-1), "n_expert_used")
    assert_rejects(lambda: replace(meta, expert_bytes=-1), "expert_bytes")
    assert_rejects(lambda: replace(meta, total_expert_bytes=-1), "total_expert_bytes")


from llm_lab.moe.trace import (
    TraceHeader,
    TraceStep,
    read_trace,
    replay_cache,
    window_unions,
    write_trace,
)


def sample_header(**overrides) -> TraceHeader:
    fields = dict(
        kind="header",
        model="olmoe",
        path="local-random",
        quant="float32",
        n_layer=2,
        n_expert=8,
        n_expert_used=2,
        expert_bytes=96,
        total_expert_bytes=1536,
        tokenizer_sha=None,
        prompt_sha="a" * 64,
        seed=0,
        decode="greedy",
        backend="transformers",
        threads=1,
        host={"os": "test"},
    )
    fields.update(overrides)
    return TraceHeader(**fields)


def sample_steps() -> list:
    # The brief's exact hand-built trace. Every union answer below is derived from these three.
    return [
        TraceStep("step", 0, 10, ((1, 2), (3, 4))),
        TraceStep("step", 1, 11, ((2, 5), (3, 6))),
        TraceStep("step", 2, 12, ((1, 5), (4, 6))),
    ]


def test_trace_round_trips_header_and_steps_exactly():
    header, steps = sample_header(), sample_steps()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trace.jsonl"
        write_trace(path, header, steps)
        got_header, got_steps = read_trace(path)
        assert got_header == header
        assert got_steps == steps
        # First line is the header, every other line is a step; nullable fields stay present.
        lines = path.read_text(encoding="utf-8").splitlines()
        assert json.loads(lines[0])["kind"] == "header"
        assert json.loads(lines[0])["tokenizer_sha"] is None
        assert [json.loads(line)["kind"] for line in lines[1:]] == ["step", "step", "step"]


def test_window_unions_are_layer_encoded_and_exact():
    steps = sample_steps()
    expected = ((0, 1), (0, 2), (0, 5), (1, 3), (1, 4), (1, 6))

    pairwise = window_unions(steps, 2)
    assert pairwise == [expected, expected]

    full = window_unions(steps, 3)
    assert full == [expected]
    assert len(full[0]) == 6

    def per_layer_counts(union):
        return tuple(
            sum(1 for layer, _ in union if layer == which) for which in (0, 1)
        )

    assert per_layer_counts(pairwise[0]) == (3, 3)
    assert per_layer_counts(pairwise[1]) == (3, 3)
    assert per_layer_counts(full[0]) == (3, 3)


def test_replay_cache_capacity_changes_hits_never_the_logits():
    steps = sample_steps()
    logits_sha = "c" * 64
    small = replay_cache(steps, 2, logits_sha)
    large = replay_cache(steps, 12, logits_sha)

    # Cache capacity moves hits/misses but never the trace, the logits, or the tokens.
    assert small["token_ids"] == [10, 11, 12]
    assert large["token_ids"] == [10, 11, 12]
    assert small["trace_sha256"] == large["trace_sha256"]
    assert small["logits_sha256"] == large["logits_sha256"] == logits_sha
    assert small["requests"] == large["requests"] == 12
    assert (small["hits"], small["misses"]) == (0, 12)
    assert (large["hits"], large["misses"]) == (6, 6)
    assert small["hits"] != large["hits"]

    # Capacity 0 is a real cache that holds nothing: every request misses.
    empty = replay_cache(steps, 0, logits_sha)
    assert (empty["hits"], empty["misses"]) == (0, 12)


def test_trace_rejects_malformed_steps():
    header = sample_header()

    def write(steps):
        with tempfile.TemporaryDirectory() as tmp:
            write_trace(Path(tmp) / "bad.jsonl", header, steps)

    # Non-contiguous / duplicate pos.
    assert_rejects(
        lambda: write([TraceStep("step", 0, 10, ((1, 2), (3, 4))),
                       TraceStep("step", 2, 11, ((2, 5), (3, 6)))]),
        "pos",
    )
    assert_rejects(
        lambda: write([TraceStep("step", 0, 10, ((1, 2), (3, 4))),
                       TraceStep("step", 0, 11, ((2, 5), (3, 6)))]),
        "pos",
    )
    # Wrong layer count (header says 2 layers).
    assert_rejects(lambda: write([TraceStep("step", 0, 10, ((1, 2),))]), "layer")
    # Wrong top-k length (header says 2 experts per layer).
    assert_rejects(lambda: write([TraceStep("step", 0, 10, ((1, 2, 3), (3, 4)))]), "top-k")
    # Duplicate expert IDs within one layer.
    assert_rejects(lambda: write([TraceStep("step", 0, 10, ((1, 1), (3, 4)))]), "duplicate expert")
    # Expert ID outside [0, n_expert).
    assert_rejects(lambda: write([TraceStep("step", 0, 10, ((1, 8), (3, 4)))]), "outside")


import torch
from transformers import OlmoeConfig

from llm_lab.moe.baseline import (
    RunConfig,
    build_tiny_olmoe,
    run_hf,
    seed_everything,
)
from llm_lab.moe.trace import TraceCapture, trace_hf

TINY_PROMPT = [[1, 3, 4]]
TINY_SEED = 0
TINY_THREADS = 1
TINY_NEW_TOKENS = 3


def tiny_run_config() -> RunConfig:
    return RunConfig(
        model="tiny-olmoe",
        path="local-random",
        quant="float32",
        seed=TINY_SEED,
        decode="greedy",
        threads=TINY_THREADS,
        backend="transformers",
        prompt_sha=sha256_token_ids(TINY_PROMPT[0]),
        tokenizer_sha=None,
    )


def tiny_trace_header(**overrides) -> TraceHeader:
    # The header must describe the tiny fixture exactly, or _validate_steps rejects the trace.
    return sample_header(
        model="tiny-olmoe",
        n_layer=2,
        n_expert=8,
        n_expert_used=2,
        expert_bytes=None,
        total_expert_bytes=None,
        prompt_sha=sha256_token_ids(TINY_PROMPT[0]),
        seed=TINY_SEED,
        threads=TINY_THREADS,
        **overrides,
    )


def test_tiny_olmoe_two_builds_generate_identically():
    # Oracle identity: same seed, two fresh models, byte-identical token IDs.
    config = tiny_run_config()
    record_a = run_hf(build_tiny_olmoe(TINY_SEED), TINY_PROMPT, config, TINY_NEW_TOKENS)
    record_b = run_hf(build_tiny_olmoe(TINY_SEED), TINY_PROMPT, config, TINY_NEW_TOKENS)
    assert record_a.n_prompt == 3
    assert record_a.n_generated == 3
    assert compare_runs(record_a, record_b) == {
        "identical": True,
        "first_divergence": None,
        "n_compared": 3,
        "fingerprints_match": True,
    }


def test_trace_hf_records_processed_tokens_and_stable_logits():
    header = tiny_trace_header()
    with tempfile.TemporaryDirectory() as tmp:
        trace_path = Path(tmp) / "trace.jsonl"
        logits_a = Path(tmp) / "logits_a.pt"
        logits_b = Path(tmp) / "logits_b.pt"

        capture_a = trace_hf(
            build_tiny_olmoe(TINY_SEED), TINY_PROMPT, header,
            TINY_NEW_TOKENS, trace_path, logits_a,
        )
        header_out, steps = read_trace(trace_path)
        assert header_out == header
        assert len(steps) == 3
        # Each step: two layers, two experts per layer (matches the header shape).
        for step in steps:
            assert len(step.layer_experts) == 2
            assert all(len(layer) == 2 for layer in step.layer_experts)

        # tok_id is the PROCESSED token: it must equal the token generated at that position
        # by an independent greedy decode, not the next-token prediction.
        traced_tokens = [step.tok_id for step in steps]
        model = build_tiny_olmoe(TINY_SEED).eval()
        with torch.no_grad():
            generated = model.generate(
                torch.tensor(TINY_PROMPT, dtype=torch.long),
                max_new_tokens=3, min_new_tokens=3, do_sample=False, use_cache=True,
            )
        assert traced_tokens == generated[0, 3:].tolist()
        assert capture_a.run_record.token_ids == tuple(traced_tokens)

        # Logits hash is stable across a second identical capture.
        capture_b = trace_hf(
            build_tiny_olmoe(TINY_SEED), TINY_PROMPT, header,
            TINY_NEW_TOKENS, Path(tmp) / "trace_b.jsonl", logits_b,
        )
        assert capture_a.logits_sha256 == capture_b.logits_sha256
        assert capture_a.trace_sha256 == capture_b.trace_sha256
        # The default capture records the decode setting it ran under.
        assert capture_a.stop_at_eos is True
        assert capture_a.to_dict()["stop_at_eos"] is True


def test_trace_hf_fixed_length_ignores_eos():
    # stop_at_eos=False must always yield the full requested length and record the setting.
    header = tiny_trace_header()
    with tempfile.TemporaryDirectory() as tmp:
        capture = trace_hf(
            build_tiny_olmoe(TINY_SEED), TINY_PROMPT, header,
            TINY_NEW_TOKENS, Path(tmp) / "t.jsonl", Path(tmp) / "l.pt",
            stop_at_eos=False,
        )
        _, steps = read_trace(Path(tmp) / "t.jsonl")
        assert len(steps) == 3
        assert capture.run_record.n_generated == 3
        # The setting must be recorded, not just implied by length: a fixed-length capture is
        # distinguishable after the fact from a greedy-with-EOS one.
        assert capture.stop_at_eos is False
        assert capture.to_dict()["stop_at_eos"] is False


def test_incremental_kv_decode_is_bit_repeatable():
    # Determinism: the SAME incremental KV-cached decode over one position, run twice, must be
    # bit-identical. This is where torch.equal belongs and holds -- a repeated identical path is
    # exactly reproducible or the oracle is meaningless. "Not a tolerance" is literal here.
    prompt = torch.tensor(TINY_PROMPT, dtype=torch.long)

    def decode_one():
        seed_everything(TINY_SEED, TINY_THREADS)
        model = build_tiny_olmoe(TINY_SEED).eval()
        with torch.no_grad():
            primed = model(prompt, use_cache=True)
            first = primed.logits[:, -1, :].argmax(-1, keepdim=True)
            step = model(first, past_key_values=primed.past_key_values, use_cache=True)
        return step.logits[:, -1, :]

    assert torch.equal(decode_one(), decode_one())


def test_kv_cache_on_vs_off_agree_within_float_tolerance():
    # Decoder correctness: incremental cached decode (1 query row) vs full use_cache=False recompute
    # (N query rows) over the same generated position. argmax must match exactly; raw logits agree
    # only within float tolerance. Bit-identity across these two paths is IMPOSSIBLE in float32:
    # a 1-row cached step and an N-row full recompute order the matmul reductions differently. That
    # is mathematical equivalence (identical argmax), NOT nondeterminism -- each path is on its own
    # bit-repeatable (see test_incremental_kv_decode_is_bit_repeatable). Observed max-abs-diff on
    # this fixture is ~1.49e-7 (~2x float32 eps), comfortably under atol=1e-6.
    seed_everything(TINY_SEED, TINY_THREADS)
    model = build_tiny_olmoe(TINY_SEED).eval()
    prompt = torch.tensor(TINY_PROMPT, dtype=torch.long)
    with torch.no_grad():
        primed = model(prompt, use_cache=True)
        first_tok = primed.logits[:, -1, :].argmax(-1, keepdim=True)
        cached = model(first_tok, past_key_values=primed.past_key_values, use_cache=True)
        full = model(torch.cat([prompt, first_tok], dim=1), use_cache=False)
    a, b = cached.logits[:, -1, :], full.logits[:, -1, :]
    assert a.argmax(-1).item() == b.argmax(-1).item()
    assert torch.allclose(a, b, atol=1e-6, rtol=1e-5)


import io
import contextlib

import llm_lab.moe.baseline as baseline
from llm_lab import main


class _FakeLlama:
    """A no-download stand-in for llama_cpp.Llama.

    Records its constructor and generate kwargs so a test can assert the exact CPU/greedy contract,
    and yields token IDs whose detokenized bytes tokenize back to DIFFERENT IDs, so an implementation
    that stored re-tokenized text instead of the yielded IDs would be caught red-handed.
    """

    last_init = None
    last_generate = None

    # generate yields these; EOS(999) stops the loop; [5, 6, 7] is what re-tokenizing "zzz" gives.
    YIELD = [100, 200, 300, 400, 999, 500]
    EOS = 999
    RETOKENIZE = [5, 6, 7]

    def __init__(self, **kwargs):
        _FakeLlama.last_init = kwargs

    def tokenize(self, data: bytes):
        # Prompt "hi" -> [1, 2]; the sentinel "zzz" (what detokenize returns) -> [5, 6, 7].
        return list(_FakeLlama.RETOKENIZE) if data == b"zzz" else [1, 2]

    def token_eos(self):
        return _FakeLlama.EOS

    def detokenize(self, ids) -> bytes:
        return b"zzz"

    def generate(self, prompt_ids, **kwargs):
        _FakeLlama.last_generate = {"prompt_ids": list(prompt_ids), **kwargs}
        yield from _FakeLlama.YIELD


def llama_cpp_config(prompt_ids) -> RunConfig:
    return RunConfig(
        model="olmoe", path="fake.gguf", quant="Q4_K_M", seed=0, decode="greedy",
        threads=1, backend="llama-cpp", prompt_sha=sha256_token_ids(prompt_ids), tokenizer_sha=None,
    )


def test_run_llama_cpp_uses_cpu_greedy_params_and_stores_exact_ids():
    real = baseline.Llama
    baseline.Llama = _FakeLlama
    try:
        config = llama_cpp_config([1, 2])
        record = baseline.run_llama_cpp(Path("fake.gguf"), "hi", config, max_new_tokens=3, n_ctx=256)
    finally:
        baseline.Llama = real

    # CPU-only construction, exact seed/threads/ctx, no verbosity.
    assert _FakeLlama.last_init == {
        "model_path": "fake.gguf", "n_gpu_layers": 0, "seed": 0,
        "n_threads": 1, "n_ctx": 256, "verbose": False,
    }
    # Exact greedy generator parameters: temperature zero, sampling filters off, penalties neutral.
    gen = _FakeLlama.last_generate
    assert gen["prompt_ids"] == [1, 2]
    assert gen["temp"] == 0.0
    assert (gen["top_k"], gen["top_p"], gen["min_p"], gen["typical_p"]) == (1, 1.0, 0.0, 1.0)
    assert (gen["repeat_penalty"], gen["frequency_penalty"], gen["presence_penalty"]) == (1.0, 0.0, 0.0)

    # Stores the exact integer IDs yielded (capped at max_new_tokens), NOT re-tokenized text.
    assert record.token_ids == (100, 200, 300)
    assert record.token_ids != tuple(_FakeLlama.RETOKENIZE), "stored re-tokenized text, not yielded IDs"
    assert record.n_prompt == 2
    assert record.n_generated == 3
    assert record.config_fingerprint == fingerprint_config(llama_cpp_config([1, 2]))


def test_run_llama_cpp_stops_at_eos_without_storing_it():
    real = baseline.Llama
    baseline.Llama = _FakeLlama
    try:
        record = baseline.run_llama_cpp(
            Path("fake.gguf"), "hi", llama_cpp_config([1, 2]), max_new_tokens=10, n_ctx=256
        )
    finally:
        baseline.Llama = real
    # generate yields 100,200,300,400,EOS,... -> stop at EOS, EOS never stored.
    assert record.token_ids == (100, 200, 300, 400)
    assert _FakeLlama.EOS not in record.token_ids


def test_cli_compare_matches_library_call():
    with tempfile.TemporaryDirectory() as tmp:
        a = Path(tmp) / "a.json"
        b = Path(tmp) / "b.json"
        out = Path(tmp) / "cmp.json"
        sample_record((7, 8, 9)).write(a)
        sample_record((7, 5, 9)).write(b)
        main(["moe", "compare", str(a), str(b), "--output", str(out)])
        assert json.loads(out.read_text(encoding="utf-8")) == compare_runs(
            sample_record((7, 8, 9)), sample_record((7, 5, 9))
        )


def test_cli_replay_cache_reads_trace_and_hashes_logits():
    header, steps = sample_header(), sample_steps()
    with tempfile.TemporaryDirectory() as tmp:
        trace_path = Path(tmp) / "trace.jsonl"
        logits_path = Path(tmp) / "logits.pt"
        out = Path(tmp) / "replay.json"
        write_trace(trace_path, header, steps)
        logits_path.write_bytes(b"raw-logits-bytes")
        main(["moe", "replay-cache", str(trace_path), "--logits", str(logits_path),
              "--capacity-experts", "12", "--output", str(out)])
        # Ruling A: the CLI hashed the logits FILE and passed that digest to replay_cache.
        expected = replay_cache(steps, 12, sha256_file(logits_path))
        assert json.loads(out.read_text(encoding="utf-8")) == expected
        assert expected["hits"] == 6 and expected["misses"] == 6


def test_cli_replay_cache_rejects_malformed_trace_before_replay():
    # Ruling B: a trace that would fail _validate_steps must be rejected by read_trace, never replayed.
    header = sample_header()
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad.jsonl"
        logits_path = Path(tmp) / "l.pt"
        out = Path(tmp) / "replay.json"
        logits_path.write_bytes(b"x")
        # A step with the wrong layer count (header says 2) hand-written past write_trace's guard.
        lines = [
            json.dumps(header.to_dict(), sort_keys=True, separators=(",", ":")),
            json.dumps(TraceStep("step", 0, 10, ((1, 2),)).to_dict(), sort_keys=True, separators=(",", ":")),
        ]
        bad.write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:
            main(["moe", "replay-cache", str(bad), "--logits", str(logits_path),
                  "--capacity-experts", "4", "--output", str(out)])
        except SystemExit as exc:
            assert exc.code != 0
        else:
            raise AssertionError("replayed a malformed trace")
        assert not out.exists(), "wrote replay output for a malformed trace"


def test_cli_run_missing_llama_cpp_model_names_the_path():
    with tempfile.TemporaryDirectory() as tmp:
        missing = str(Path(tmp) / "absent.gguf")
        out = Path(tmp) / "run.json"
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err):
                main(["moe", "run", "--backend", "llama-cpp", "--model-path", missing,
                      "--prompt", "hi", "--output", str(out)])
        except SystemExit as exc:
            assert exc.code != 0
        else:
            raise AssertionError("accepted a missing model path")
        assert missing in err.getvalue(), f"stderr did not name the missing path: {err.getvalue()!r}"


def test_cli_run_tiny_is_deterministic_without_download():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "run.json"
        main(["moe", "run", "--backend", "tiny", "--prompt-token-ids", "1,3,4",
              "--seed", "0", "--max-new-tokens", "3", "--output", str(out)])
        record = RunRecord.read(out)
        assert record.n_prompt == 3
        assert record.n_generated == 3
        expected_cfg = replace(
            tiny_run_config(), backend="tiny", prompt_sha=sha256_token_ids([1, 3, 4])
        )
        assert record.config_fingerprint == fingerprint_config(expected_cfg)


def test_cli_trace_tiny_writes_the_fixed_summary_schema():
    with tempfile.TemporaryDirectory() as tmp:
        trace_p = Path(tmp) / "trace.jsonl"
        logits_p = Path(tmp) / "logits.pt"
        record_p = Path(tmp) / "run.json"
        summary_p = Path(tmp) / "summary.json"
        main(["moe", "trace", "--backend", "tiny", "--prompt-token-ids", "1,3,4",
              "--seed", "0", "--max-new-tokens", "3", "--trace", str(trace_p),
              "--logits", str(logits_p), "--run-record", str(record_p), "--summary", str(summary_p)])

        summary = json.loads(summary_p.read_text(encoding="utf-8"))
        assert set(summary) == {
            "model", "path", "quant", "backend", "precision", "n_layer", "n_expert",
            "n_expert_used", "n_prompt", "n_generated", "n_steps", "stop_at_eos", "seed",
            "decode", "threads", "prompt_sha", "tokenizer_sha", "trace_sha256", "logits_sha256",
            "trace_path", "logits_path", "host",
        }
        assert summary["backend"] == "tiny"
        assert summary["precision"] == "FP32"
        assert summary["decode"] == "greedy"
        assert summary["stop_at_eos"] is True
        assert summary["n_steps"] == summary["n_generated"] == 3
        assert (summary["n_layer"], summary["n_expert"], summary["n_expert_used"]) == (2, 8, 2)
        assert summary["prompt_sha"] == sha256_token_ids([1, 3, 4])
        assert summary["tokenizer_sha"] is None
        # Hashes match what was actually written.
        _, steps = read_trace(trace_p)
        assert len(steps) == 3
        assert summary["logits_sha256"] == sha256_file(logits_p)
        assert RunRecord.read(record_p).n_generated == 3


def test_cli_trace_continue_after_eos_flips_stop_at_eos():
    with tempfile.TemporaryDirectory() as tmp:
        summary_p = Path(tmp) / "summary.json"
        main(["moe", "trace", "--backend", "tiny", "--prompt-token-ids", "1,3,4",
              "--seed", "0", "--max-new-tokens", "3", "--continue-after-eos",
              "--trace", str(Path(tmp) / "t.jsonl"), "--logits", str(Path(tmp) / "l.pt"),
              "--run-record", str(Path(tmp) / "r.json"), "--summary", str(summary_p)])
        assert json.loads(summary_p.read_text(encoding="utf-8"))["stop_at_eos"] is False


def test_cli_moe_argument_matrix_rejections():
    """Every backend's forbidden/required argument is a SystemExit, before any model work."""
    def rejected(argv):
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                main(argv)
        except SystemExit as exc:
            assert exc.code != 0
        else:
            raise AssertionError(f"accepted invalid argv: {argv}")

    # tiny run requires token IDs, rejects text/model/n-ctx.
    rejected(["moe", "run", "--backend", "tiny", "--output", "o.json"])
    rejected(["moe", "run", "--backend", "tiny", "--prompt-token-ids", "1", "--prompt", "hi", "--output", "o.json"])
    rejected(["moe", "run", "--backend", "tiny", "--prompt-token-ids", "1", "--n-ctx", "8", "--output", "o.json"])
    # hf-int8 run requires prompt + model-id + 40-char revision, rejects token IDs and n-ctx.
    rejected(["moe", "run", "--backend", "hf-int8", "--prompt", "hi", "--output", "o.json"])
    rejected(["moe", "run", "--backend", "hf-int8", "--prompt", "hi", "--model-id", "m",
              "--revision", "short", "--output", "o.json"])
    rejected(["moe", "run", "--backend", "hf-int8", "--prompt", "hi", "--model-id", "m",
              "--revision", "a" * 40, "--prompt-token-ids", "1", "--output", "o.json"])
    rejected(["moe", "run", "--backend", "hf-int8", "--prompt", "hi", "--model-id", "m",
              "--revision", "a" * 40, "--n-ctx", "8", "--output", "o.json"])
    rejected(["moe", "run", "--backend", "hf-int8", "--prompt", "hi", "--model-id", "m",
              "--revision", "a" * 40, "--model-path", "m.gguf", "--output", "o.json"])
    # llama-cpp run rejects token IDs / model-id.
    rejected(["moe", "run", "--backend", "llama-cpp", "--model-path", "m.gguf",
              "--prompt-token-ids", "1", "--output", "o.json"])
    rejected(["moe", "run", "--backend", "llama-cpp", "--prompt", "hi", "--model-id", "m",
              "--model-path", "m.gguf", "--output", "o.json"])
    # trace: n-ctx is not even a trace argument (argparse rejects it); tiny rejects prompt.
    rejected(["moe", "trace", "--backend", "tiny", "--prompt-token-ids", "1", "--n-ctx", "8",
              "--trace", "t", "--logits", "l", "--run-record", "r", "--summary", "s"])
    rejected(["moe", "trace", "--backend", "tiny", "--prompt-token-ids", "1", "--prompt", "hi",
              "--trace", "t", "--logits", "l", "--run-record", "r", "--summary", "s"])
    rejected(["moe", "trace", "--backend", "hf-bf16", "--prompt", "hi", "--model-id", "m",
              "--revision", "short", "--trace", "t", "--logits", "l", "--run-record", "r", "--summary", "s"])


import subprocess


def test_gitignore_isolates_raw_from_committed_evidence():
    """Raw traces/logits are gitignored; the evidence directory stays tracked."""
    repo = Path(__file__).resolve().parent.parent
    lines = (repo / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "/models/m1/" in lines, "missing /models/m1/ ignore rule"
    assert "/artifacts/m1/raw/" in lines, "missing /artifacts/m1/raw/ ignore rule"
    assert "/artifacts/m1/evidence/" not in lines, "evidence must not be ignored"

    def ignored(rel: str) -> bool:
        # git check-ignore exits 0 when the path is ignored, 1 when it is tracked.
        return subprocess.run(
            ["git", "check-ignore", "-q", rel], cwd=repo
        ).returncode == 0

    assert ignored("artifacts/m1/raw/m1a-trace.jsonl"), "raw trace not ignored"
    assert ignored("models/m1/olmoe.gguf"), "model checkpoint not ignored"
    assert not ignored("artifacts/m1/evidence/m1a-commands.ps1"), "evidence wrongly ignored"


def test_cli_refuses_to_overwrite_output_without_force():
    """An output-writing command rejects an existing path unless --force is passed."""
    with tempfile.TemporaryDirectory() as tmp:
        a = Path(tmp) / "a.json"
        b = Path(tmp) / "b.json"
        out = Path(tmp) / "cmp.json"
        sample_record((7, 8, 9)).write(a)
        sample_record((7, 8, 9)).write(b)
        out.write_text("do-not-clobber", encoding="utf-8")

        try:
            with contextlib.redirect_stderr(io.StringIO()):
                main(["moe", "compare", str(a), str(b), "--output", str(out)])
        except SystemExit as exc:
            assert exc.code != 0
        else:
            raise AssertionError("overwrote an existing output without --force")
        assert out.read_text(encoding="utf-8") == "do-not-clobber", "clobbered without --force"

        # --force is the explicit override.
        main(["moe", "compare", str(a), str(b), "--output", str(out), "--force"])
        assert json.loads(out.read_text(encoding="utf-8")) == compare_runs(
            sample_record((7, 8, 9)), sample_record((7, 8, 9))
        )


from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from llm_lab.moe.baseline import sha256_tokenizer


def _write_fast_tokenizer(directory, vocab) -> "PreTrainedTokenizerFast":
    """A tiny local fast (tokenizer.json) tokenizer, built offline from an explicit vocab."""
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tok.save(str(Path(directory) / "tokenizer.json"))
    return PreTrainedTokenizerFast(tokenizer_file=str(Path(directory) / "tokenizer.json"))


def test_sha256_tokenizer_hashes_local_files_deterministically():
    # A fast tokenizer records no file paths in init_kwargs, so the digest must come from its
    # serialized files. Same files -> same sha across independent loads; changed bytes -> changed.
    vocab = {"[UNK]": 0, "a": 1, "b": 2}
    with tempfile.TemporaryDirectory() as tmp:
        one, two, other = Path(tmp) / "one", Path(tmp) / "two", Path(tmp) / "other"
        for d in (one, two, other):
            d.mkdir()
        sha_one = sha256_tokenizer(_write_fast_tokenizer(one, vocab))
        sha_two = sha256_tokenizer(_write_fast_tokenizer(two, vocab))
        sha_other = sha256_tokenizer(_write_fast_tokenizer(other, {**vocab, "c": 3}))

        assert len(sha_one) == 64
        assert sha_one == sha_two, "same tokenizer files must hash identically across loads"
        assert sha_one != sha_other, "different tokenizer bytes must change the digest"


def test_sha256_tokenizer_refuses_when_no_artifact():
    assert_rejects(lambda: sha256_tokenizer(object()), "artifact")


def mixed_quant_expert_tensors(hidden: int, ffn: int) -> dict:
    """Two-layer OLMoE experts where layer 1 uses a wider ffn, so its per-expert bytes differ.

    A real Q4_K_M GGUF gets differing layer sizes from mixing quant types; a plain byte-size
    difference exercises the same code path without needing two quant encoders in a fixture.
    """
    return {
        **expert_tensors(1, 8, hidden, ffn),
        **{
            name.replace("blk.0.", "blk.1."): arr
            for name, arr in expert_tensors(1, 8, hidden, ffn * 2).items()
        },
    }


def test_gguf_meta_allows_mixed_quant_per_layer_sizes():
    # Real Q4_K_M mixes quant types across layers, so per-expert bytes differ layer to layer.
    with tempfile.TemporaryDirectory() as tmp:
        tensors = mixed_quant_expert_tensors(64, 32)
        path = write_gguf(Path(tmp) / "mixed.gguf", GGUF_KV, tensors)
        meta = read_gguf_meta(path)
        assert meta.expert_bytes is None, "no single per-expert value under mixed sizes"
        expected = sum(int(np.asarray(a).nbytes) for a in tensors.values())
        assert meta.total_expert_bytes == expected, "total must be the measured sum over all layers"

        # A uniform-size checkpoint still yields a non-null per-expert value.
        uniform = read_gguf_meta(
            write_gguf(Path(tmp) / "uniform.gguf", GGUF_KV, expert_tensors(2, 8, 64, 32))
        )
        assert uniform.expert_bytes == 3 * (8 * 32 * 64 * 4 // 8)


def test_moe_help_names_five_commands_and_no_marketing():
    """The public `moe --help` names exactly the five reference commands and stays claim-free.

    This is the documentation scope boundary: help text is the one place a user reads before
    running anything, so it must inventory the commands and never leak speed/marketing wording.
    """
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            main(["moe", "--help"])
    except SystemExit as exc:
        assert exc.code == 0, f"moe --help exited nonzero: {exc.code}"
    else:
        raise AssertionError("moe --help did not exit")

    text = out.getvalue()
    for command in ("meta", "run", "trace", "compare", "replay-cache"):
        assert command in text, f"moe --help omits the {command!r} command"
    for forbidden in ("speedup", "PageCC", "GPU", "lossless to BF16"):
        assert forbidden not in text, f"moe --help leaked forbidden wording: {forbidden!r}"


if __name__ == "__main__":
    # Auto-discovery, so a test appended by a later task can never be silently skipped.

    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("All MoE Trace Tests Passed!")
