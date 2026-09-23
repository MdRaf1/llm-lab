# tests/test_bench_read.py
import os
import sys
from pathlib import Path

import numpy as np

from llm_lab.frontier.bench_read import read_range_cold, SECTOR


def _tmpfile(tmp: Path, nbytes: int) -> Path:
    p = tmp / "blob.bin"
    data = (np.arange(nbytes, dtype=np.uint64).astype(np.uint8)).tobytes()
    p.write_bytes(data)
    return p, data


def test_cold_read_aligned_unaligned_and_eof():
    tmp = Path("_m3b_tmp"); tmp.mkdir(exist_ok=True)
    size = 3 * SECTOR + 100          # deliberately NOT a sector multiple (EOF case)
    p, data = _tmpfile(tmp, size)
    # aligned range
    assert read_range_cold(str(p), SECTOR, SECTOR, size) == data[SECTOR:2 * SECTOR]
    # unaligned offset + length (sector-rounding must still return exact requested bytes)
    assert read_range_cold(str(p), SECTOR + 100, 50, size) == data[SECTOR + 100:SECTOR + 150]
    # read that straddles EOF: request 2*SECTOR starting near the end -> short, no error
    got = read_range_cold(str(p), 3 * SECTOR, 2 * SECTOR, size)
    assert got == data[3 * SECTOR:size]


def test_buffer_base_is_sector_aligned():
    # FILE_FLAG_NO_BUFFERING requires the buffer's BASE ADDRESS to be sector-aligned,
    # not merely offset/length. A read that returns correct bytes above already proves
    # the aligned buffer works end-to-end against a real file; this pins the base directly.
    from llm_lab.frontier.bench_read import aligned_buffer
    buf = aligned_buffer(3 * SECTOR)
    assert buf.ctypes.data % SECTOR == 0, "buffer base not sector-aligned; NO_BUFFERING will fail on device"
    assert buf.nbytes == 3 * SECTOR


def test_choose_experts_deterministic_and_identical():
    from llm_lab.frontier.bench_read import choose_experts
    a = choose_experts(1234, 48, 128, 8)
    b = choose_experts(1234, 48, 128, 8)
    assert a == b, "same seed must give identical expert sets for both arms"
    assert set(a.keys()) == set(range(48))
    assert all(len(v) == 8 and len(set(v)) == 8 for v in a.values())


def test_stock_offset_matches_library_slice():
    # Closes the ne[2]=experts caveat NON-circularly: a raw file read at the computed
    # absolute offset must equal gguf's own axis-0 per-expert slice.
    import gguf
    from test_expert_repack import _tiny_gguf
    from llm_lab.frontier.bench_read import stock_ranges, choose_experts
    from llm_lab.frontier.expert_repack import per_expert_bytes, expert_tensor_name
    tmp = Path("_m3b_tmp"); tmp.mkdir(exist_ok=True)
    path = _tiny_gguf(tmp, n_layer=2, n_expert=4)
    reader = gguf.GGUFReader(str(path))
    tmap = {t.name: t for t in reader.tensors}
    chosen = choose_experts(7, 2, 4, 2)
    ranges = stock_ranges(reader, chosen, 4)
    L0, es0 = sorted(chosen.items())[0]
    e0 = es0[0]
    t = tmap[expert_tensor_name(L0, "gate")]
    per = per_expert_bytes(t, 4)
    off = t.data_offset + e0 * per
    with open(path, "rb") as fh:
        fh.seek(off)
        raw = fh.read(per)
    assert raw == t.data[e0].tobytes()
    assert (off, per) in ranges


def test_run_arm_stats_shape():
    from llm_lab.frontier.bench_read import run_arm
    tmp = Path("_m3b_tmp"); tmp.mkdir(exist_ok=True)
    p = tmp / "arm.bin"; p.write_bytes(b"\xAB" * (16 * SECTOR))
    size = p.stat().st_size
    ranges = [(i * SECTOR, SECTOR) for i in range(16)]
    stats = run_arm(str(p), ranges, working_set_bytes=16 * SECTOR, file_size=size, threads=8, runs=5)
    assert set(stats) >= {"min", "median", "max", "seconds"}
    assert stats["min"] <= stats["median"] <= stats["max"]
    assert len(stats["seconds"]) == 5


def test_repacked_arm_is_scattered_not_a_run():
    # Scatter guard: the repacked arm must read the routed experts at their true, non-adjacent
    # packed-file offsets — NOT an 8-adjacent contiguous run that would fake a sequential win.
    import json
    from test_expert_repack import _tiny_gguf
    from llm_lab.frontier.expert_repack import repack_model, MANIFEST_NAME
    from llm_lab.frontier.bench_read import repacked_ranges, choose_experts
    tmp = Path("_m3b_tmp"); tmp.mkdir(exist_ok=True)
    path = _tiny_gguf(tmp, n_layer=2, n_expert=8)
    out = tmp / "scatter_repack"
    repack_model(str(path), str(out))
    manifest = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))
    chosen = choose_experts(3, 2, 8, 4)             # 4 of 8 per layer -> guaranteed gaps
    chosen_offs = sorted(o for o, _ in repacked_ranges(manifest, chosen))
    all_offs = sorted(b["offset"] for b in manifest["blobs"])
    chosen_set = set(chosen_offs)
    between = [o for o in all_offs if chosen_offs[0] < o < chosen_offs[-1] and o not in chosen_set]
    assert between, "repacked arm reads a contiguous run; scatter guard violated"


def test_derive_verdict_and_gate_logic():
    from llm_lab.frontier.bench_read import derive_verdict, derive_m3_gate
    assert derive_verdict({"median": 10, "min": 9, "max": 11},
                          {"median": 5, "min": 3, "max": 8}) == "separated_win"
    assert derive_verdict({"median": 10, "min": 6, "max": 12},
                          {"median": 5, "min": 3, "max": 8}) == "overlap_inconclusive"
    oe = {"all_equal": True, "n_slices": 24, "rollup_sha256": "x" * 64}
    g = derive_m3_gate(oe, {"throughput_verdict": "separated_win"})
    assert g["passed"] is True and g["branch"] == "proceed_m4"
    g2 = derive_m3_gate({**oe, "all_equal": False}, {"throughput_verdict": "separated_win"})
    assert g2["passed"] is False and g2["branch"] == "fix_output_exact"
    g3 = derive_m3_gate(oe, {"throughput_verdict": "overlap_inconclusive"})
    assert g3["passed"] is False and g3["branch"] == "remeasure"


def test_bench_read_model_end_to_end():
    from test_expert_repack import _tiny_gguf
    from llm_lab.frontier.expert_repack import repack_model
    from llm_lab.frontier.bench_read import bench_read_model
    tmp = Path("_m3b_tmp"); tmp.mkdir(exist_ok=True)
    path = _tiny_gguf(tmp, n_layer=2, n_expert=4)
    out = tmp / "bench_repack"
    repack_model(str(path), str(out))
    gate = bench_read_model(str(path), str(out), seed=7, experts_per_layer=2, runs=3, threads=4)
    assert gate["passed"] in (True, False)
    assert gate["throughput_verdict"] in ("separated_win", "overlap_inconclusive")
    assert "qd16_ratio" in gate["throughput"] and "nvme_seq_bps" in gate["throughput"]
    assert gate["output_exact"]["all_equal"] is True


if __name__ == "__main__":
    import shutil
    shutil.rmtree("_m3b_tmp", ignore_errors=True)  # idempotent: reset scratch between runs
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("All Bench Read Tests Passed!")
