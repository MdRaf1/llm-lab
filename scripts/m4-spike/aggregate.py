import json, statistics, sys

PER_TOKEN_EXPERT_BYTES = 8 * 48 * 2_654_208  # top-8 × 48 layers × 2.65 MB/expert ≈ 1.02e9 (spec §3)
# miss_frac=0.5 → ≈500 MB/token threshold: squarely in the physical band between the ~2.6 tok/s
# drive floor and the ~10 tok/s interactive line, so below it the OS is clearly caching most of the
# model. It is a regime boundary, not a bright line — raw numbers + spreads are committed so a
# near-boundary result reads as a visible judgment call, never a hidden flip.


REASON_DETAIL = {
    "go": "warm miss/token above the streaming anchor and warm tok/s below the conservative ceiling: decode is per-token expert-scatter-I/O-bound with headroom for a tiered loader",
    "model_fits_cache_moot": "warm miss/token below the streaming anchor: decode is not per-token expert-scatter-I/O-bound on this host (the working set faults in ~once per process and is reused across tokens even when the model exceeds RAM); not a streaming regime the M4 loader would improve",
    "stock_saturates_no_headroom": "streaming regime but warm tok/s already at/above the conservative I/O ceiling: the drive is saturated, leaving no headroom for a tiered loader to beat",
}


def compute_ceiling(scatter_bps, miss_bytes_per_token):
    return scatter_bps / miss_bytes_per_token if miss_bytes_per_token > 0 else float("inf")


def verdict(warm_tok_s_max, warm_ceiling_min_tok_s, warm_miss_bytes_per_token_median,
            per_token_expert_bytes, miss_frac=0.5):
    streaming_regime = warm_miss_bytes_per_token_median > miss_frac * per_token_expert_bytes
    headroom = warm_tok_s_max < warm_ceiling_min_tok_s   # spread separation, not a fixed margin
    go = streaming_regime and headroom
    if go:
        reason = "go"
    elif not streaming_regime:
        reason = "model_fits_cache_moot"
    else:
        reason = "stock_saturates_no_headroom"
    return {"verdict": "go" if go else "no_go",
            "branch": "proceed_patch" if go else "escalate",
            "streaming_regime": streaming_regime, "headroom": headroom, "reason": reason}


def _stats(xs):
    return {"median": statistics.median(xs), "min": min(xs), "max": max(xs)}


def build_gate(runs, scatter_bps, scatter_provenance, per_token_expert_bytes, miss_frac=0.5):
    w_tok, w_mbpt = _stats(runs["warm_tok_s"]), _stats(runs["warm_miss_bytes_per_token"])
    c_tok, c_mbpt = _stats(runs["cold_tok_s"]), _stats(runs["cold_miss_bytes_per_token"])
    warm_ceiling = compute_ceiling(scatter_bps, w_mbpt["median"])       # headline
    warm_ceiling_min = compute_ceiling(scatter_bps, w_mbpt["max"])      # conservative (most misses)
    v = verdict(w_tok["max"], warm_ceiling_min, w_mbpt["median"], per_token_expert_bytes, miss_frac)
    gate = {"phase": "baseline_spike", "gated_on": "warm_steady_state",
            "ncpumoe_note": "flag ~inert on CPU-only box; baseline is OS-paged mmap",
            "miss_bytes_source": "physical PhysicalDisk read counter (bytes the drive served), not cache-size arithmetic",
            "miss_frac": miss_frac, "miss_frac_rationale": "0.5*per_token ≈ 500 MB/token, mid physical band (~2.6 floor..~10 interactive tok/s)",
            "warm_tok_s_median": w_tok["median"], "warm_tok_s_min": w_tok["min"], "warm_tok_s_max": w_tok["max"],
            "warm_miss_bytes_per_token_median": w_mbpt["median"], "warm_miss_bytes_per_token_min": w_mbpt["min"], "warm_miss_bytes_per_token_max": w_mbpt["max"],
            "warm_ceiling_tok_s": warm_ceiling, "warm_ceiling_min_tok_s": warm_ceiling_min,
            "cold_tok_s_median": c_tok["median"], "cold_tok_s_min": c_tok["min"], "cold_tok_s_max": c_tok["max"],
            "cold_miss_bytes_per_token_median": c_mbpt["median"], "cold_ceiling_tok_s": compute_ceiling(scatter_bps, c_mbpt["median"]),
            "per_token_expert_bytes": per_token_expert_bytes, "scatter_bps": scatter_bps, "scatter_provenance": scatter_provenance,
            "spike_verdict": v["verdict"], "branch": v["branch"],
            "streaming_regime": v["streaming_regime"], "headroom": v["headroom"], "reason": v["reason"],
            "reason_detail": REASON_DETAIL[v["reason"]]}
    if v["verdict"] == "no_go":
        gate["passed"] = False
    return gate


def _selfcheck():
    assert abs(compute_ceiling(1.021e9, 1.0e8) - 10.21) < 1e-6
    assert compute_ceiling(1.021e9, 0) == float("inf")
    # streaming (8e8 > 500MB) AND separated (fastest warm 0.85 < pessimistic ceiling 1.10) -> go
    assert verdict(0.85, 1.10, 8.0e8, 1.0e9)["verdict"] == "go"
    # streaming but fastest warm overlaps the ceiling -> no headroom
    r = verdict(1.15, 1.10, 8.0e8, 1.0e9)
    assert r["verdict"] == "no_go" and r["reason"] == "stock_saturates_no_headroom"
    # warm barely misses (< 500MB/token) -> model fits cache -> moot
    r = verdict(0.85, 1.10, 1.0e8, 1.0e9)
    assert r["verdict"] == "no_go" and r["reason"] == "model_fits_cache_moot"
    g = build_gate({"cold_tok_s": [0.5], "cold_miss_bytes_per_token": [9.8e8],
                    "warm_tok_s": [0.80, 0.82, 0.78], "warm_miss_bytes_per_token": [7.9e8, 8.0e8, 8.1e8]},
                   1.021e9, "M3 microbench (same host)", 1.0e9)
    assert g["spike_verdict"] == "go" and g["branch"] == "proceed_patch"
    assert g["reason_detail"].startswith("warm miss/token above")
    assert g["warm_tok_s_max"] == 0.82 and g["warm_miss_bytes_per_token_max"] == 8.1e8  # spreads committed
    g2 = build_gate({"cold_tok_s": [0.5], "cold_miss_bytes_per_token": [9.8e8],
                     "warm_tok_s": [6.0], "warm_miss_bytes_per_token": [1.0e8]},
                    1.021e9, "M3 microbench (same host)", 1.0e9)
    assert g2["passed"] is False and g2["reason"] == "model_fits_cache_moot"
    assert "not per-token expert-scatter" in g2["reason_detail"]
    print("aggregate self-check OK")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
