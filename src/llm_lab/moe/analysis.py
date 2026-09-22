"""Analysis-only derivations over routing traces. Never executes model logits."""
from __future__ import annotations

from collections import OrderedDict
from itertools import combinations
from typing import Sequence

from llm_lab.moe.trace import TraceStep, replay_cache


def _requests(steps: Sequence[TraceStep]) -> list[tuple[int, int]]:
    """Every (layer, expert) routing fact in router order, matching trace.replay_cache."""
    return [
        (layer, expert)
        for step in steps
        for layer, experts in enumerate(step.layer_experts)
        for expert in experts
    ]


def stack_distances(steps: Sequence[TraceStep]) -> list[int | None]:
    """LRU stack distance per request: distinct blobs seen since this blob's last request."""
    recency: "OrderedDict[tuple[int, int], None]" = OrderedDict()  # LRU: oldest first
    out: list[int | None] = []
    for key in _requests(steps):
        if key in recency:
            # Distinct blobs more recently used than `key` = entries after it in the order.
            keys = list(recency)
            out.append(len(keys) - 1 - keys.index(key))
            recency.move_to_end(key)
        else:
            out.append(None)
            recency[key] = None
    return out


def reuse_distance_histogram(steps: Sequence[TraceStep]) -> dict[str, object]:
    distances = stack_distances(steps)
    histogram: dict[str, int] = {}
    cold = 0
    for d in distances:
        if d is None:
            cold += 1
        else:
            histogram[str(d)] = histogram.get(str(d), 0) + 1
    return {"histogram": histogram, "cold": cold, "total_requests": len(distances)}


def coactivation_pairs(steps: Sequence[TraceStep]) -> dict[int, dict[tuple[int, int], int]]:
    """Per-layer count of unordered expert pairs co-selected at the same position."""
    out: dict[int, dict[tuple[int, int], int]] = {}
    for step in steps:
        for layer, experts in enumerate(step.layer_experts):
            layer_counts = out.setdefault(layer, {})
            for lo, hi in combinations(sorted(set(experts)), 2):
                layer_counts[(lo, hi)] = layer_counts.get((lo, hi), 0) + 1
    return out


def coactivation_to_json(pairs: dict[int, dict[tuple[int, int], int]]) -> dict:
    return {
        str(layer): {f"{lo},{hi}": c for (lo, hi), c in counts.items()}
        for layer, counts in pairs.items()
    }


def hit_rate_curve(steps, fractions, n_expert_total, logits_sha256) -> list[dict]:
    """Hit-rate vs working-set fraction f; capacity = round(f * n_expert_total)."""
    if n_expert_total <= 0:
        raise ValueError(f"n_expert_total must be positive, got {n_expert_total}")
    rows = []
    for f in sorted(set(fractions)):
        capacity = round(f * n_expert_total)
        r = replay_cache(steps, capacity, logits_sha256)
        requests = r["requests"]
        hit_rate = r["hits"] / requests if requests else 0.0
        rows.append({
            "fraction": f,
            "capacity_experts": capacity,
            "hits": r["hits"],
            "misses": r["misses"],
            "requests": requests,
            "hit_rate": hit_rate,
            "miss_rate": 1.0 - hit_rate if requests else 0.0,
        })
    return rows


import bisect


def interp_miss_rate(curve, f: float) -> float:
    """Linear interpolation of miss_rate at fraction f, clamped to the curve endpoints."""
    pts = sorted((row["fraction"], row["miss_rate"]) for row in curve)
    xs = [x for x, _ in pts]
    if f <= xs[0]:
        return pts[0][1]
    if f >= xs[-1]:
        return pts[-1][1]
    i = bisect.bisect_right(xs, f)
    x0, y0 = pts[i - 1]
    x1, y1 = pts[i]
    return y0 + (y1 - y0) * (f - x0) / (x1 - x0)


def project_tier(ram_bytes, nonexpert_bytes, reserve_bytes, avg_expert_bytes,
                 n_expert_total, n_layer, n_expert_used, curve, b_cold_bytes_s) -> dict:
    expert_cache_bytes = ram_bytes - nonexpert_bytes - reserve_bytes
    if expert_cache_bytes <= 0:
        return {"ram_bytes": ram_bytes, "core_fits": False, "capacity_experts": None,
                "fraction": None, "miss_rate": None, "miss_bytes_per_token": None, "tok_s": None}
    capacity = min(n_expert_total, expert_cache_bytes // avg_expert_bytes)
    f = capacity / n_expert_total
    miss_rate = interp_miss_rate(curve, f)
    miss_per_token = n_layer * n_expert_used * miss_rate
    cold_bytes_per_token = miss_per_token * avg_expert_bytes
    tok_s = b_cold_bytes_s / cold_bytes_per_token if cold_bytes_per_token else float("inf")
    return {"ram_bytes": ram_bytes, "core_fits": True, "capacity_experts": int(capacity),
            "fraction": f, "miss_rate": miss_rate,
            "miss_bytes_per_token": cold_bytes_per_token, "tok_s": tok_s}


def gate_branch(tok_s_16gb: float, spreads_straddle: bool) -> str:
    if spreads_straddle or 5.0 < tok_s_16gb < 10.0:
        return "re_rent"
    if tok_s_16gb >= 10.0:
        return "build_m3m4"
    return "escalate_m5m6"
