# src/llm_lab/frontier/bench_read.py
"""M3 cold read benchmark: unbuffered scattered reads, both arms, NVMe re-measure, gate."""
from __future__ import annotations

import ctypes
import sys

import numpy as np

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
