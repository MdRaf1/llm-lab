"""
Unit Tests for the MoE reference pipeline (M1).
Validates:
1. Immutable RunRecord JSON round-trip and stable config fingerprints.
2. The exactness oracle: identity, first divergence, and fingerprint refusal.
"""

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


if __name__ == "__main__":
    test_run_record_round_trip_and_stable_fingerprint()
    test_oracle_reports_identity_and_first_divergence()
    test_oracle_refuses_different_fingerprints()
    print("All MoE Trace Tests Passed!")
