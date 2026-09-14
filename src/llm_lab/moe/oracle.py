"""
Exactness oracle for MoE runs.
Compares token IDs only: never decoded text, never normalized text.
Runs with different config fingerprints are refused, not compared.
"""

from __future__ import annotations

from .baseline import RunRecord


def compare_runs(a: RunRecord, b: RunRecord) -> dict[str, bool | int | None]:
    """
    Compare two runs position by position.

    `first_divergence` is the zero-based generated-token index of the first mismatch,
    or `min(len(a), len(b))` when one sequence is a strict prefix of the other.
    `n_compared` is the number of positions inspected to reach the verdict.
    """
    if a.config_fingerprint != b.config_fingerprint:
        raise ValueError(
            "fingerprints differ: "
            f"{a.config_fingerprint} != {b.config_fingerprint}; runs are not comparable"
        )

    shared = min(len(a.token_ids), len(b.token_ids))
    first_divergence = next(
        (i for i in range(shared) if a.token_ids[i] != b.token_ids[i]), None
    )

    if first_divergence is not None:
        # Both runs contain the divergent position.
        n_compared = first_divergence + 1
    elif len(a.token_ids) != len(b.token_ids):
        # Strict prefix: the divergent position exists in only one run.
        first_divergence = shared
        n_compared = shared
    else:
        n_compared = shared

    return {
        "identical": first_divergence is None,
        "first_divergence": first_divergence,
        "n_compared": n_compared,
        "fingerprints_match": True,
    }
