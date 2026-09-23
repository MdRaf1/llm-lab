# src/llm_lab/frontier/bench_read.py
"""M3 cold read benchmark: unbuffered scattered reads, both arms, NVMe re-measure, gate."""
from __future__ import annotations

import ctypes
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from llm_lab.frontier.expert_repack import EXPERT_PARTS, expert_tensor_name, per_expert_bytes

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
        _k.SetFilePointerEx(h, wintypes.LARGE_INTEGER(aligned_off), None, 0)  # 0 = FILE_BEGIN
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
