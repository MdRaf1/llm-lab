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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("All Bench Read Tests Passed!")
