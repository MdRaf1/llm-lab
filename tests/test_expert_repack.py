# tests/test_expert_repack.py
import sys
from pathlib import Path

import gguf
import numpy as np

from test_moe_trace import write_gguf, expert_tensors  # reused fixture builders
from llm_lab.frontier.expert_repack import (
    EXPERT_PARTS, expert_tensor_name, per_expert_bytes, expert_slice,
)


def _tiny_gguf(tmp: Path, n_layer=2, n_expert=4, d_model=32, d_ff=64) -> Path:
    # d_model/d_ff are multiples of 32 so the innermost dim is Q8_0 block-aligned
    # (controller ruling: original 8x8 dims are invalid for Q8_0's 32-wide blocks).
    path = tmp / "tiny.gguf"
    tensors = {}
    for l in range(n_layer):
        for part in EXPERT_PARTS:
            # expert on numpy axis 0; Q8_0 so raw bytes are deterministic and block-aligned
            arr = (np.arange(n_expert * d_model * d_ff, dtype=np.float32)
                   .reshape(n_expert, d_ff, d_model) + l)
            tensors[expert_tensor_name(l, part)] = arr
    write_gguf(path, {"qwen3moe.block_count": n_layer,
                      "qwen3moe.expert_count": n_expert,
                      "qwen3moe.expert_used_count": 2},
               tensors, arch="qwen3moe", raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    return path


def test_expert_slice_matches_numpy_axis0(tmp_path_factory=None):
    tmp = Path("_m3_tmp"); tmp.mkdir(exist_ok=True)
    path = _tiny_gguf(tmp)
    reader = gguf.GGUFReader(str(path))
    tmap = {t.name: t for t in reader.tensors}
    t = tmap[expert_tensor_name(0, "gate")]
    per = per_expert_bytes(t, 4)
    assert per == t.n_bytes // 4
    for e in range(4):
        assert expert_slice(t, e, 4) == t.data[e].tobytes()


def test_per_expert_bytes_guards():
    tmp = Path("_m3_tmp"); tmp.mkdir(exist_ok=True)
    path = _tiny_gguf(tmp)
    reader = gguf.GGUFReader(str(path))
    tmap = {t.name: t for t in reader.tensors}
    t = tmap[expert_tensor_name(0, "gate")]
    # wrong n_expert -> not divisible / axis mismatch must raise naming the tensor
    try:
        per_expert_bytes(t, 3)
    except ValueError as exc:
        assert "gate" in str(exc)
    else:
        raise AssertionError("accepted n_expert that mismatches shape[-1]")


def test_plan_layout_aligned_and_contiguous():
    from llm_lab.frontier.expert_repack import plan_layout, align_up, ALIGN
    tmp = Path("_m3_tmp"); tmp.mkdir(exist_ok=True)
    path = _tiny_gguf(tmp)
    reader = gguf.GGUFReader(str(path))
    blobs, expected = plan_layout(reader, 2, 4)
    assert len(blobs) == 2 * 4
    assert [(b["layer"], b["expert"]) for b in blobs[:5]] == [(0,0),(0,1),(0,2),(0,3),(1,0)]
    for b in blobs:
        assert b["offset"] % ALIGN == 0, "blob not 4K-aligned"
        r = b["roles"]
        assert r["gate"]["sub_offset"] == 0
        assert r["up"]["sub_offset"] == r["gate"]["length"]
        assert r["down"]["sub_offset"] == r["gate"]["length"] + r["up"]["length"]
        assert b["length"] == sum(r[p]["length"] for p in ("gate", "up", "down"))
    last = blobs[-1]
    assert expected == align_up(last["offset"] + last["length"])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("All Expert Repack Tests Passed!")
