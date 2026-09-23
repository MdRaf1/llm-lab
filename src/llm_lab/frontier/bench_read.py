# src/llm_lab/frontier/bench_read.py
"""M3 cold read benchmark: unbuffered scattered reads, both arms, NVMe re-measure, gate."""
from __future__ import annotations

import ctypes
import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from llm_lab.frontier.expert_repack import (
    EXPERT_PARTS, MANIFEST_NAME, PACKED_NAME, expert_tensor_name, per_expert_bytes,
)

SECTOR = 4096

if sys.platform == "win32":
    from ctypes import wintypes
    _k = ctypes.WinDLL("kernel32", use_last_error=True)
    _GENERIC_READ = 0x80000000
    _FILE_SHARE_READ = 0x00000001
    _OPEN_EXISTING = 3
    _FILE_FLAG_NO_BUFFERING = 0x20000000
    _INVALID = ctypes.c_void_p(-1).value
    _k.CreateFileW.restype = wintypes.HANDLE
    _k.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                               wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    _k.SetFilePointerEx.restype = wintypes.BOOL
    _k.SetFilePointerEx.argtypes = [wintypes.HANDLE, wintypes.LARGE_INTEGER,
                                    ctypes.POINTER(wintypes.LARGE_INTEGER), wintypes.DWORD]
    _k.ReadFile.restype = wintypes.BOOL
    _k.ReadFile.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                            ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
    _k.CloseHandle.restype = wintypes.BOOL
    _k.CloseHandle.argtypes = [wintypes.HANDLE]


def aligned_buffer(nbytes: int, align: int = SECTOR):
    raw = np.empty(nbytes + align, dtype=np.uint8)
    off = (-raw.ctypes.data) % align
    return raw[off:off + nbytes]


def read_range_cold(path: str, offset: int, length: int, file_size: int) -> bytes:
    aligned_off = (offset // SECTOR) * SECTOR
    delta = offset - aligned_off
    extent = ((delta + length + SECTOR - 1) // SECTOR) * SECTOR
    buf = aligned_buffer(extent)
    h = _k.CreateFileW(path, _GENERIC_READ, _FILE_SHARE_READ, None,
                       _OPEN_EXISTING, _FILE_FLAG_NO_BUFFERING, None)
    if h == _INVALID:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not _k.SetFilePointerEx(h, wintypes.LARGE_INTEGER(aligned_off), None, 0):  # 0 = FILE_BEGIN
            raise ctypes.WinError(ctypes.get_last_error())
        nread = wintypes.DWORD(0)
        ok = _k.ReadFile(h, buf.ctypes.data, extent, ctypes.byref(nread), None)
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        _k.CloseHandle(h)
    end = min(delta + length, int(nread.value))
    return buf[delta:end].tobytes() if end > delta else b""


def choose_experts(seed: int, n_layer: int, n_expert: int, k: int) -> dict[int, list[int]]:
    rng = random.Random(seed)
    return {L: sorted(rng.sample(range(n_expert), k)) for L in range(n_layer)}


def repacked_ranges(manifest: dict, chosen: dict[int, list[int]]) -> list[tuple[int, int]]:
    by_le = {(b["layer"], b["expert"]): b for b in manifest["blobs"]}
    out = []
    for L, es in chosen.items():
        for e in es:
            b = by_le[(L, e)]
            out.append((b["offset"], b["length"]))
    return out


def stock_ranges(reader, chosen: dict[int, list[int]], n_expert: int) -> list[tuple[int, int]]:
    tmap = {t.name: t for t in reader.tensors}
    out = []
    for L, es in chosen.items():
        for part in EXPERT_PARTS:
            t = tmap[expert_tensor_name(L, part)]
            per = per_expert_bytes(t, n_expert)
            for e in es:
                out.append((t.data_offset + e * per, per))
    return out


def working_set_bytes(reader, chosen: dict[int, list[int]], n_expert: int) -> int:
    tmap = {t.name: t for t in reader.tensors}
    total = 0
    for L, es in chosen.items():
        per_expert = sum(per_expert_bytes(tmap[expert_tensor_name(L, p)], n_expert) for p in EXPERT_PARTS)
        total += per_expert * len(es)
    return total


def run_arm(path, ranges, working_set_bytes, file_size, threads, runs) -> dict:
    seconds = []
    for _ in range(runs):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=threads) as pool:
            list(pool.map(lambda r: read_range_cold(path, r[0], r[1], file_size), ranges))
        seconds.append(time.perf_counter() - t0)
    kept = seconds[1:] if len(seconds) > 1 else seconds   # drop warm-up run 0
    bps = [working_set_bytes / s for s in kept]
    return {"min": min(bps), "median": statistics.median(bps), "max": max(bps),
            "seconds": seconds}


def derive_verdict(repacked: dict, stock: dict) -> str:
    if repacked["median"] > stock["median"] and repacked["min"] >= stock["max"]:
        return "separated_win"
    return "overlap_inconclusive"


def derive_m3_gate(output_exact: dict, throughput: dict) -> dict:
    verdict = throughput["throughput_verdict"]
    all_equal = bool(output_exact["all_equal"])
    passed = all_equal and verdict == "separated_win"
    branch = ("fix_output_exact" if not all_equal
              else "proceed_m4" if verdict == "separated_win" else "remeasure")
    return {"output_exact": output_exact, "throughput": throughput,
            "throughput_verdict": verdict, "branch": branch, "passed": passed}


def measure_nvme_seq(path, nbytes, file_size, chunk=8 * 1024 * 1024) -> float:
    total = min(nbytes, file_size)
    read = 0
    off = 0
    t0 = time.perf_counter()
    while off < total:
        got = read_range_cold(path, off, min(chunk, total - off), file_size)
        if not got:
            break
        read += len(got)
        off += chunk
    dt = time.perf_counter() - t0
    return read / dt if dt else 0.0


def bench_read_model(input_path, repack_dir, *, seed, experts_per_layer, runs, threads) -> dict:
    import gguf
    from pathlib import Path
    repack = Path(repack_dir)
    manifest = json.loads((repack / MANIFEST_NAME).read_text(encoding="utf-8"))
    packed = str(repack / PACKED_NAME)
    packed_size = os.path.getsize(packed)
    src_size = os.path.getsize(input_path)
    reader = gguf.GGUFReader(str(input_path))
    n_layer, n_expert = manifest["num_layers"], manifest["num_experts"]
    chosen = choose_experts(seed, n_layer, n_expert, experts_per_layer)
    ws = working_set_bytes(reader, chosen, n_expert)
    rep_ranges = repacked_ranges(manifest, chosen)
    stk_ranges = stock_ranges(reader, chosen, n_expert)
    rep16 = run_arm(packed, rep_ranges, ws, packed_size, threads, runs)
    stk16 = run_arm(input_path, stk_ranges, ws, src_size, threads, runs)
    rep1 = run_arm(packed, rep_ranges, ws, packed_size, 1, runs)
    stk1 = run_arm(input_path, stk_ranges, ws, src_size, 1, runs)
    nvme = measure_nvme_seq(input_path, 4 * (2 ** 30), src_size)
    verdict = derive_verdict(rep16, stk16)
    throughput = {
        "working_set_bytes": ws, "experts_per_layer": experts_per_layer, "seed": seed,
        "runs": runs, "qd_proxy_threads": threads,
        "repacked_bps": {k: rep16[k] for k in ("min", "median", "max")},
        "stock_bps": {k: stk16[k] for k in ("min", "median", "max")},
        "qd16_ratio": (rep16["median"] / stk16["median"]) if stk16["median"] else None,
        "qd1_ratio": (rep1["median"] / stk1["median"]) if stk1["median"] else None,
        "nvme_seq_bps": nvme, "throughput_verdict": verdict,
        "note": "QD is a thread-count proxy applied identically to both arms, not a measured device queue depth.",
    }
    return derive_m3_gate(manifest["output_exact"], throughput)
