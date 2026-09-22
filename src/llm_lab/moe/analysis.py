"""Analysis-only derivations over routing traces. Never executes model logits."""
from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

from llm_lab.moe.trace import TraceStep


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
