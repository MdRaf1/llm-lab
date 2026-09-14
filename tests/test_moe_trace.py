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


if __name__ == "__main__":
    # Auto-discovery, so a test appended by a later task can never be silently skipped.
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("All MoE Trace Tests Passed!")
