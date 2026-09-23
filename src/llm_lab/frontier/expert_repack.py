# src/llm_lab/frontier/expert_repack.py
"""M3 expert-contiguous repack: slice, plan, write, verify. Layout-only, output-exact."""
from __future__ import annotations

EXPERT_PARTS = ("gate", "up", "down")
ALIGN = 4096
HEADROOM_BYTES = 10 * (2 ** 30)
PACKED_NAME = "qwen3-experts.packed"
MANIFEST_NAME = "manifest.json"


def expert_tensor_name(layer: int, part: str) -> str:
    return f"blk.{layer}.ffn_{part}_exps.weight"


def per_expert_bytes(tensor, n_expert: int) -> int:
    name = tensor.name
    if len(tensor.shape) != 3:
        raise ValueError(f"{name}: expected 3-D stacked expert tensor, got shape {list(tensor.shape)}")
    if int(tensor.shape[-1]) != n_expert:
        raise ValueError(f"{name}: outer dim {int(tensor.shape[-1])} != expert_count {n_expert}")
    if tensor.n_bytes % n_expert:
        raise ValueError(f"{name}: {tensor.n_bytes} bytes not divisible by expert_count {n_expert}")
    return tensor.n_bytes // n_expert


def expert_slice(tensor, e: int, n_expert: int) -> bytes:
    per_expert_bytes(tensor, n_expert)  # guards
    return tensor.data[e].tobytes()
