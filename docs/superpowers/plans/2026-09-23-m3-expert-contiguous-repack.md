# M3 — Expert-Contiguous Repack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repack the Qwen3-30B-A3B Q4_K_M GGUF into a single packed file of 4 KB-aligned, per-`(layer,expert)` contiguous blobs + a manifest, prove it byte-identical to the source, and measure that reading a routed expert working-set from it beats reading the same experts from the stock GGUF.

**Architecture:** Two new argparse subcommands under the existing `moe` group in `src/llm_lab/__init__.py`, backed by two modules in `src/llm_lab/frontier/`. `moe repack` slices experts (numpy axis 0 of each stacked `ffn_*_exps` tensor), streams them to `models/m3/qwen3-repack/qwen3-experts.packed` with a `manifest.json`, and discharges the output-exact gate by reading blobs back and comparing sha256 to source slices. `moe bench-read` reads a seeded synthetic routed subset cold (Windows `FILE_FLAG_NO_BUFFERING` via ctypes, ≥16-thread pool) from both the repack and the stock GGUF, re-measures NVMe, and emits a derived `m3-gate.json`. No loader, cache, async, or inference — that is M4.

**Tech Stack:** Python 3.11 (`./.venv/Scripts/python.exe`), stdlib `argparse` + `ctypes`/`ctypes.wintypes` + `concurrent.futures` + `hashlib` + `shutil`, `numpy>=2.0.0`, `gguf>=0.19.0`. No pytest, no click/typer, no new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-11-moe-exact-runtime-design.md` (§7 M3 row; §2 existing-code audit; §10 honesty rules; §11-Q1 open threat)

## Global Constraints

- Python floor: `requires-python = ">=3.11,<3.12"`. Runtime is `sys.platform == "win32"`, python 3.11.15.
- Dependencies: `numpy>=2.0.0`, `gguf>=0.19.0` only. No new deps. `pywin32` is NOT installed — unbuffered I/O MUST use `ctypes.windll.kernel32`.
- Invoke everything as `./.venv/Scripts/python.exe -m llm_lab moe <cmd> ...`. Bare `python`/`pytest` are not on PATH.
- Tests are directly-executable assert scripts (no pytest): module-level `test_*` fns, bare `assert`, auto-discovered by a `__main__` loop, run via `./.venv/Scripts/python.exe tests/<file>.py`, ending `print("All <X> Tests Passed!")`.
- Target model (source, opened READ-ONLY): `models/m1/qwen3-gguf/Qwen3-30B-A3B-Q4_K_M.gguf` — arch `qwen3moe`, 48 layers, 128 experts, top-8; `ffn_{gate,up}_exps` Q4_K, `ffn_down_exps` Q6_K; router `ffn_gate_inp` F32; no shared expert.
- Every reported number ties to a committed command+output: `_emit` prints JSON identical to what it writes; `m3-commands.ps1` tees each run's stdout. NVMe bandwidth is RE-MEASURED in-session — never quote §3's 2.62 GB/s as a gate input.
- Output identity is byte equality (sha256), never text/`.strip()`. Q4_K / Q6_K / F32 are distinct numeric paths, never cross-asserted. §11-Q1 stays open; M3 deliberately does not depend on it.
- Gate verdicts are DERIVED in the builder (`passed = all(criteria)`), never written as literals — mirror `moe/analysis.py::gate_branch`.
- Scope is layout-only / output-exact. OUT OF SCOPE (the M4 line): no loader, cache, tiering, demand-load, async/IOCP, eviction, GPU, or llama.cpp patch, and no speedup claim beyond measured read throughput. The benchmark thread pool is an instrument, not a loader.

## Review Focus

- **Wrong-axis GGUF** (experts not on the last `ne` dim): `t.shape[-1] != n_expert` must abort loudly, not silently mis-slice. → Task 1 test.
- **Indivisible expert bytes** (`t.n_bytes % n_expert != 0`, e.g. a ragged/mixed tensor): must raise naming the tensor, never floor-divide silently. → Task 1 test.
- **Unbuffered read past EOF** (sector-rounding the last expert's slice runs beyond file end): `ReadFile` returns a short count — accept it, clamp to file size, never error or hang. → Task 7 test.
- **Disk exhaustion mid-write despite preflight** (free drifts between check and write): catch `OSError`, delete the `.tmp`, raise loud — never leave a partial `.packed`. → Task 4 test.
- **Re-run over an existing output dir**: repack's two files are covered by the no-clobber guard and refused without `--force`. → Task 6 test.

## File Structure

- `src/llm_lab/frontier/expert_repack.py` — **create.** Geometry read, per-expert slicing, layout planning, 4 KB alignment, streaming writer, byte-identity verification. Owns the output-exact half. Constants: `EXPERT_PARTS = ("gate", "up", "down")`, `ALIGN = 4096`, `HEADROOM_BYTES = 10 * (2**30)`, `PACKED_NAME = "qwen3-experts.packed"`, `MANIFEST_NAME = "manifest.json"`.
- `src/llm_lab/frontier/bench_read.py` — **create.** ctypes unbuffered cold reader, seeded synthetic scatter, both-arm benchmark, NVMe re-measure, gate derivation. Owns the throughput half + the final `m3-gate.json` builder.
- `src/llm_lab/__init__.py` — **modify.** Register `repack` + `bench-read` in `_add_moe_subparsers`; add dispatch branches in `_run_moe`; add a `repack` entry to `_moe_output_paths` (it writes two files).
- `tests/test_expert_repack.py` — **create.** Direct-run assert script for the repack module.
- `tests/test_bench_read.py` — **create.** Direct-run assert script for the benchmark module.
- `artifacts/m3/evidence/` — **create (Task 10).** `M3-REPORT.md`, `m3-commands.ps1`, `m3-gate.json`, `m3-files.sha256`, plus tee'd `m3-*-output.txt`.
- `models/m3/qwen3-repack/` — **create at runtime (gitignored).** `qwen3-experts.packed` + `manifest.json`.

Both test files reuse the fixture builders already in `tests/test_moe_trace.py` — `write_gguf(path, kv, tensors, *, arch=..., raw_dtype=...)` and `expert_tensors(...)` (which builds stacked experts as numpy axis 0, matching the real layout). Import them with `from test_moe_trace import write_gguf, expert_tensors` (same `tests/` dir; `sys.path[0]` is `tests/` when run directly).

---

### Task 1: Geometry read + per-expert slicing

**Files:**
- Create: `src/llm_lab/frontier/expert_repack.py`
- Test: `tests/test_expert_repack.py`

**Interfaces:**
- Consumes: `gguf.GGUFReader`; `llm_lab.moe.meta.read_gguf_meta(path) -> ModelMeta` (fields `n_layer, n_expert, n_expert_used, ...`).
- Produces:
  - `EXPERT_PARTS = ("gate", "up", "down")`
  - `expert_tensor_name(layer: int, part: str) -> str` → `f"blk.{layer}.ffn_{part}_exps.weight"`
  - `per_expert_bytes(tensor, n_expert: int) -> int` — asserts `len(tensor.shape) == 3`, `int(tensor.shape[-1]) == n_expert`, `tensor.n_bytes % n_expert == 0`; returns `tensor.n_bytes // n_expert`. Raises `ValueError` naming the tensor on any violation.
  - `expert_slice(tensor, e: int, n_expert: int) -> bytes` — returns `tensor.data[e].tobytes()` (numpy axis 0 = expert), after the same guards.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_expert_repack.py
import sys
from pathlib import Path

import gguf
import numpy as np

from test_moe_trace import write_gguf, expert_tensors  # reused fixture builders
from llm_lab.frontier.expert_repack import (
    EXPERT_PARTS, expert_tensor_name, per_expert_bytes, expert_slice,
)


def _tiny_gguf(tmp: Path, n_layer=2, n_expert=4, d_model=8, d_ff=8) -> Path:
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


```python
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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("All Expert Repack Tests Passed!")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe tests/test_expert_repack.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_lab.frontier.expert_repack'`.

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe tests/test_expert_repack.py`
Expected: PASS — `All Expert Repack Tests Passed!`

- [ ] **Step 5: Commit**

```bash
git add src/llm_lab/frontier/expert_repack.py tests/test_expert_repack.py
git commit -m "feat(m3): expert tensor slicing with axis and divisibility guards"
```

### Task 2: Layout plan + 4 KB alignment

**Files:**
- Modify: `src/llm_lab/frontier/expert_repack.py`
- Test: `tests/test_expert_repack.py`

**Interfaces:**
- Consumes: `gguf.GGUFReader`, `per_expert_bytes`, `expert_tensor_name`, `EXPERT_PARTS`, `ALIGN`.
- Produces:
  - `align_up(n: int, a: int = ALIGN) -> int` → `-(-n // a) * a`.
  - `plan_layout(reader, n_layer: int, n_expert: int) -> tuple[list[dict], int]` — returns `(blobs, expected_output_bytes)`. `blobs` is layer-major, expert-ascending; each entry:
    ```python
    {"layer": L, "expert": e,
     "offset": <4KB-aligned packed byte offset>,
     "length": <gate_per + up_per + down_per, unpadded true length>,
     "roles": {"gate": {"sub_offset": 0,               "length": gate_per, "dtype": "Q4_K"},
               "up":   {"sub_offset": gate_per,          "length": up_per,   "dtype": "Q4_K"},
               "down": {"sub_offset": gate_per + up_per, "length": down_per, "dtype": "Q6_K"}}}
    ```
    Each blob starts at `align_up(prev_offset + prev_length)`; `expected_output_bytes = align_up(last.offset + last.length)`.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe tests/test_expert_repack.py`
Expected: FAIL — `ImportError: cannot import name 'plan_layout'`.

- [ ] **Step 3: Write minimal implementation**

```python
def align_up(n: int, a: int = ALIGN) -> int:
    return -(-n // a) * a


def plan_layout(reader, n_layer: int, n_expert: int) -> tuple[list[dict], int]:
    tmap = {t.name: t for t in reader.tensors}
    blobs: list[dict] = []
    offset = 0
    for layer in range(n_layer):
        pers = {}
        for part in EXPERT_PARTS:
            t = tmap.get(expert_tensor_name(layer, part))
            if t is None:
                raise ValueError(f"missing expert tensor {expert_tensor_name(layer, part)}")
            pers[part] = (per_expert_bytes(t, n_expert), t.tensor_type.name)
        for e in range(n_expert):
            sub = 0
            roles = {}
            for part in EXPERT_PARTS:
                length, dtype = pers[part]
                roles[part] = {"sub_offset": sub, "length": length, "dtype": dtype}
                sub += length
            offset = align_up(offset)
            blobs.append({"layer": layer, "expert": e, "offset": offset, "length": sub, "roles": roles})
            offset += sub
    expected = align_up(blobs[-1]["offset"] + blobs[-1]["length"]) if blobs else 0
    return blobs, expected
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe tests/test_expert_repack.py`
Expected: PASS — `All Expert Repack Tests Passed!`

- [ ] **Step 5: Commit**

```bash
git add src/llm_lab/frontier/expert_repack.py tests/test_expert_repack.py
git commit -m "feat(m3): plan 4K-aligned layer-major blob layout with per-role sub-offsets"
```

---

### Task 3: Preflight disk gate (mirror M1c)

**Files:**
- Modify: `src/llm_lab/frontier/expert_repack.py`
- Test: `tests/test_expert_repack.py`

**Interfaces:**
- Consumes: `HEADROOM_BYTES`.
- Produces: `check_disk(expected_output_bytes: int, free_bytes: int, headroom_bytes: int = HEADROOM_BYTES) -> dict` — returns a report dict `{"expected_output_bytes", "headroom_bytes", "required_bytes", "available_free_bytes"}` where `required = expected_output_bytes + headroom_bytes`; raises `RuntimeError(f"PREFLIGHT_FAIL required={required} available={free_bytes}")` when `required > free_bytes`. `free_bytes` is passed in (the CLI supplies `shutil.disk_usage(out_dir).free`) so the check is deterministically testable.

- [ ] **Step 1: Write the failing test**

```python
def test_check_disk_passes_and_fails():
    from llm_lab.frontier.expert_repack import check_disk
    rep = check_disk(1000, free_bytes=10_000, headroom_bytes=2000)
    assert rep["required_bytes"] == 3000 and rep["available_free_bytes"] == 10_000
    try:
        check_disk(1000, free_bytes=2500, headroom_bytes=2000)  # required 3000 > 2500
    except RuntimeError as exc:
        assert "PREFLIGHT_FAIL" in str(exc)
        assert "required=3000" in str(exc) and "available=2500" in str(exc)
    else:
        raise AssertionError("check_disk accepted insufficient free space")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe tests/test_expert_repack.py`
Expected: FAIL — `ImportError: cannot import name 'check_disk'`.

- [ ] **Step 3: Write minimal implementation**

```python
def check_disk(expected_output_bytes: int, free_bytes: int,
               headroom_bytes: int = HEADROOM_BYTES) -> dict:
    required = expected_output_bytes + headroom_bytes
    report = {
        "expected_output_bytes": expected_output_bytes,
        "headroom_bytes": headroom_bytes,
        "required_bytes": required,
        "available_free_bytes": free_bytes,
    }
    if required > free_bytes:
        raise RuntimeError(f"PREFLIGHT_FAIL required={required} available={free_bytes}")
    return report
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe tests/test_expert_repack.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm_lab/frontier/expert_repack.py tests/test_expert_repack.py
git commit -m "feat(m3): mirror M1c preflight disk gate with named headroom"
```

---

### Task 4: Streaming packed-file writer

**Files:**
- Modify: `src/llm_lab/frontier/expert_repack.py`
- Test: `tests/test_expert_repack.py`

**Interfaces:**
- Consumes: `gguf.GGUFReader`, `plan_layout`, `check_disk`, `expert_slice`, `align_up`, `EXPERT_PARTS`, `ALIGN`, `PACKED_NAME`; stdlib `os`, `shutil`, `hashlib`, `pathlib.Path`.
- Produces: `write_packed(reader, blobs, n_expert, out_dir, expected_bytes, *, free_bytes) -> list[dict]` — runs `check_disk(expected_bytes, free_bytes)`, streams every blob to `out_dir/<PACKED_NAME>.tmp` (each blob 4 KB-aligned via zero padding; roles written gate→up→down), records `src_sha256` (hash of the source slice) + `blob_sha256` (hash of the concatenated blob) into each blob's per-role and blob entries, then `os.replace(tmp, out_dir/PACKED_NAME)` atomically. On any `OSError` mid-write it deletes the `.tmp` and re-raises. Returns the mutated `blobs` (now carrying hashes). Never writes weights through more than one expert slice at a time (streaming; source stays a read-only mmap).

- [ ] **Step 1: Write the failing test**

```python
def test_write_packed_size_hashes_and_preflight():
    import hashlib
    from llm_lab.frontier.expert_repack import plan_layout, write_packed, PACKED_NAME
    tmp = Path("_m3_tmp"); tmp.mkdir(exist_ok=True)
    path = _tiny_gguf(tmp)
    reader = gguf.GGUFReader(str(path))
    blobs, expected = plan_layout(reader, 2, 4)
    out = tmp / "repack"
    blobs = write_packed(reader, blobs, 4, out, expected, free_bytes=10 ** 12)
    packed = out / PACKED_NAME
    last = blobs[-1]
    assert packed.stat().st_size == last["offset"] + last["length"]
    raw = packed.read_bytes()
    for b in blobs:
        for part, r in b["roles"].items():
            start = b["offset"] + r["sub_offset"]
            got = raw[start:start + r["length"]]
            assert hashlib.sha256(got).hexdigest() == r["src_sha256"]
    # preflight refuses when free < required, and leaves no packed file behind
    out2 = tmp / "repack_fail"
    try:
        write_packed(reader, blobs, 4, out2, expected, free_bytes=1)
    except RuntimeError as exc:
        assert "PREFLIGHT_FAIL" in str(exc)
    else:
        raise AssertionError("write_packed ignored the disk gate")
    assert not (out2 / PACKED_NAME).exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe tests/test_expert_repack.py`
Expected: FAIL — `ImportError: cannot import name 'write_packed'`.

- [ ] **Step 3: Write minimal implementation**

```python
def write_packed(reader, blobs, n_expert, out_dir, expected_bytes, *, free_bytes) -> list[dict]:
    import hashlib
    import os
    from pathlib import Path
    check_disk(expected_bytes, free_bytes)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / (PACKED_NAME + ".tmp")
    tmap = {t.name: t for t in reader.tensors}
    try:
        with open(tmp, "wb") as f:
            for b in blobs:
                pad = b["offset"] - f.tell()
                if pad > 0:
                    f.write(b"\x00" * pad)
                for part in EXPERT_PARTS:
                    t = tmap[expert_tensor_name(b["layer"], part)]
                    data = expert_slice(t, b["expert"], n_expert)
                    b["roles"][part]["src_sha256"] = hashlib.sha256(data).hexdigest()
                    f.write(data)
        os.replace(tmp, out / PACKED_NAME)
    except OSError:
        if tmp.exists():
            tmp.unlink()
        raise
    return blobs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe tests/test_expert_repack.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm_lab/frontier/expert_repack.py tests/test_expert_repack.py
git commit -m "feat(m3): streaming packed-file writer with atomic rename and preflight"
```

---

### Task 5: Byte-identity verification + output-exact witness

**Files:**
- Modify: `src/llm_lab/frontier/expert_repack.py`
- Test: `tests/test_expert_repack.py`

**Interfaces:**
- Consumes: the packed file + `blobs` (with `src_sha256` per role) from Task 4; stdlib `hashlib`.
- Produces: `verify_repack(packed_path, blobs) -> dict` — reads each blob back from the packed file at `offset+sub_offset`, hashes each role slice to `read_sha256` (stored back into the blob's role entry), `memcmp`s it against `src_sha256`, and returns the output-exact witness `{"n_slices": int, "all_equal": bool, "rollup_sha256": str}`. `rollup_sha256 = sha256("\n".join(sorted(f"{src}:{read}" for every role)))`. `all_equal` is the AND of every per-role `src==read`.

- [ ] **Step 1: Write the failing test**

```python
def test_verify_repack_detects_corruption():
    from llm_lab.frontier.expert_repack import plan_layout, write_packed, verify_repack, PACKED_NAME
    tmp = Path("_m3_tmp"); tmp.mkdir(exist_ok=True)
    path = _tiny_gguf(tmp)
    reader = gguf.GGUFReader(str(path))
    blobs, expected = plan_layout(reader, 2, 4)
    out = tmp / "repack_verify"
    blobs = write_packed(reader, blobs, 4, out, expected, free_bytes=10 ** 12)
    packed = out / PACKED_NAME
    witness = verify_repack(packed, blobs)
    assert witness["all_equal"] is True
    assert witness["n_slices"] == 2 * 4 * 3
    assert len(witness["rollup_sha256"]) == 64
    # flip one byte inside the first blob -> round-trip through the file must catch it
    data = bytearray(packed.read_bytes())
    data[blobs[0]["offset"]] ^= 0xFF
    packed.write_bytes(bytes(data))
    assert verify_repack(packed, blobs)["all_equal"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe tests/test_expert_repack.py`
Expected: FAIL — `ImportError: cannot import name 'verify_repack'`.

- [ ] **Step 3: Write minimal implementation**

```python
def verify_repack(packed_path, blobs) -> dict:
    import hashlib
    from pathlib import Path
    raw = Path(packed_path).read_bytes()
    pairs = []
    all_equal = True
    for b in blobs:
        for part, r in b["roles"].items():
            start = b["offset"] + r["sub_offset"]
            read_sha = hashlib.sha256(raw[start:start + r["length"]]).hexdigest()
            r["read_sha256"] = read_sha
            if read_sha != r["src_sha256"]:
                all_equal = False
            pairs.append(f"{r['src_sha256']}:{read_sha}")
    rollup = hashlib.sha256("\n".join(sorted(pairs)).encode("utf-8")).hexdigest()
    return {"n_slices": len(pairs), "all_equal": all_equal, "rollup_sha256": rollup}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe tests/test_expert_repack.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm_lab/frontier/expert_repack.py tests/test_expert_repack.py
git commit -m "feat(m3): byte-identity verification with rollup witness"
```

---

### Task 6: `moe repack` orchestrator + CLI wiring

**Files:**
- Modify: `src/llm_lab/frontier/expert_repack.py` (add orchestrator)
- Modify: `src/llm_lab/__init__.py` (register subparser, dispatch, `_moe_output_paths`)
- Test: `tests/test_expert_repack.py`

**Interfaces:**
- Consumes: `read_gguf_meta`, `gguf.GGUFReader`, `plan_layout`, `write_packed`, `verify_repack`, `shutil.disk_usage`, `json`.
- Produces:
  - `repack_model(input_path: str, out_dir: str) -> dict` — `meta = read_gguf_meta(Path(input_path))`; open reader; `blobs, expected = plan_layout(reader, meta.n_layer, meta.n_expert)`; `write_packed(reader, blobs, meta.n_expert, out_dir, expected, free_bytes=shutil.disk_usage(out_dir_parent).free)`; `witness = verify_repack(out_dir/PACKED_NAME, blobs)`; assemble `manifest = {"source_file","source_sha256","architecture":meta.model,"num_layers":meta.n_layer,"num_experts":meta.n_expert,"top_k":meta.n_expert_used,"packed_file":PACKED_NAME,"alignment":ALIGN,"output_exact":witness,"blobs":blobs}`; write `out_dir/MANIFEST_NAME` with `json.dump(..., indent=2, sort_keys=True)`; return manifest. `source_sha256` may be `None` (18 GB hash is optional/slow) — compute it only for fixtures, `None` for the real model unless `--hash-source` is passed (skip the flag; leave `None`, the per-slice hashes are the witness).
  - CLI in `_add_moe_subparsers`: `repack` subparser with `--input` (required), `--output` (required, the dir), `--force`.
  - `_run_moe` branch: `from llm_lab.frontier.expert_repack import repack_model, PACKED_NAME, MANIFEST_NAME`; call `repack_model`; `print` a one-line summary (`repacked <n> blobs; output_exact.all_equal=<bool>`). Do NOT use `_emit` (output is a dir, not a JSON payload).
  - `_moe_output_paths`: for `repack`, return `[Path(args.output)/PACKED_NAME, Path(args.output)/MANIFEST_NAME]` so `_guard_outputs` refuses to clobber without `--force`.

- [ ] **Step 1: Write the failing test**

```python
def test_repack_model_and_cli():
    from llm_lab.frontier.expert_repack import repack_model, PACKED_NAME, MANIFEST_NAME
    import llm_lab
    tmp = Path("_m3_tmp"); tmp.mkdir(exist_ok=True)
    path = _tiny_gguf(tmp)
    out = tmp / "repack_cli"
    manifest = repack_model(str(path), str(out))
    assert (out / PACKED_NAME).exists() and (out / MANIFEST_NAME).exists()
    assert manifest["output_exact"]["all_equal"] is True
    assert manifest["num_experts"] == 4 and len(manifest["blobs"]) == 8
    # CLI path + no-clobber guard: second run without --force must exit non-zero
    out2 = tmp / "repack_cli2"
    assert llm_lab.main(["moe", "repack", "--input", str(path), "--output", str(out2)]) in (0, None)
    try:
        llm_lab.main(["moe", "repack", "--input", str(path), "--output", str(out2)])
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("re-run over existing output was not refused")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe tests/test_expert_repack.py`
Expected: FAIL — `ImportError: cannot import name 'repack_model'`.

- [ ] **Step 3: Write minimal implementation**

Add `repack_model` to `expert_repack.py`:

```python
def repack_model(input_path: str, out_dir: str) -> dict:
    import json
    import shutil
    from pathlib import Path
    import gguf
    from llm_lab.moe.meta import read_gguf_meta
    meta = read_gguf_meta(Path(input_path))
    reader = gguf.GGUFReader(str(input_path))
    blobs, expected = plan_layout(reader, meta.n_layer, meta.n_expert)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    blobs = write_packed(reader, blobs, meta.n_expert, out, expected,
                         free_bytes=shutil.disk_usage(out).free)
    witness = verify_repack(out / PACKED_NAME, blobs)
    manifest = {"source_file": Path(input_path).name, "source_sha256": None,
                "architecture": meta.model, "num_layers": meta.n_layer,
                "num_experts": meta.n_expert, "top_k": meta.n_expert_used,
                "packed_file": PACKED_NAME, "alignment": ALIGN,
                "output_exact": witness, "blobs": blobs}
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                     encoding="utf-8")
    return manifest
```

Then wire the CLI in `src/llm_lab/__init__.py` (three edits, following the existing `analyze` pattern):

1. In `_add_moe_subparsers`, after the `analyze_p` block, register the subparser:

```python
    repack_p = moe_sub.add_parser("repack", help="Repack experts into aligned contiguous blobs + manifest")
    repack_p.add_argument("--input", required=True)
    repack_p.add_argument("--output", required=True)   # target directory
    repack_p.add_argument("--force", action="store_true")
```

2. In `_run_moe`, add a dispatch branch (heavy import kept local, like `analyze`):

```python
    if args.moe_command == "repack":
        from llm_lab.frontier.expert_repack import repack_model
        manifest = repack_model(args.input, args.output)
        oe = manifest["output_exact"]
        print(f"repacked {len(manifest['blobs'])} blobs; "
              f"output_exact.all_equal={oe['all_equal']} n_slices={oe['n_slices']}")
        return
```

3. In `_moe_output_paths`, add a branch beside the existing multi-output `trace` case (so `_guard_outputs` refuses to clobber the two files without `--force`); place it above the default `return [args.output]`:

```python
    if args.moe_command == "repack":
        from pathlib import Path
        from llm_lab.frontier.expert_repack import PACKED_NAME, MANIFEST_NAME
        return [str(Path(args.output) / PACKED_NAME), str(Path(args.output) / MANIFEST_NAME)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe tests/test_expert_repack.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm_lab/frontier/expert_repack.py src/llm_lab/__init__.py tests/test_expert_repack.py
git commit -m "feat(m3): moe repack subcommand producing packed file + manifest"
```

---

### Task 7: ctypes unbuffered cold reader

**Files:**
- Create: `src/llm_lab/frontier/bench_read.py`
- Test: `tests/test_bench_read.py`

**Interfaces:**
- Consumes: stdlib `ctypes`, `ctypes.wintypes`, `os`; `numpy`.
- Produces:
  - `SECTOR = 4096`
  - `aligned_buffer(nbytes: int, align: int = SECTOR)` → numpy uint8 view whose base address is `align`-aligned (allocate `nbytes+align`, offset to alignment; caller keeps the returned view which holds its base alive).
  - `read_range_cold(path: str, offset: int, length: int, file_size: int) -> int` — opens `path` with `FILE_FLAG_NO_BUFFERING` via `CreateFileW`, sector-rounds `offset` down and the read extent up to a multiple of `SECTOR` (clamped so it never requests past `file_size` rounded up), `SetFilePointerEx` + synchronous `ReadFile` into an aligned buffer, closes the handle, returns the byte count actually delivered. A short final read at EOF is accepted, never raised. Its own handle per call (no shared-handle race).

Note (Windows/ctypes facts, verified): `pywin32` is absent — use `ctypes.windll.kernel32`. `FILE_FLAG_NO_BUFFERING = 0x20000000`, `GENERIC_READ = 0x80000000`, `FILE_SHARE_READ = 1`, `OPEN_EXISTING = 3`, `INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value`. Set `restype/argtypes` for `CreateFileW`, `SetFilePointerEx` (uses `wintypes.LARGE_INTEGER` for >4 GB offsets), `ReadFile`, `CloseHandle`. Use `ctypes.WinDLL('kernel32', use_last_error=True)` + `ctypes.get_last_error()`.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe tests/test_bench_read.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_lab.frontier.bench_read'`.

- [ ] **Step 3: Write minimal implementation**

```python
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
```

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe tests/test_bench_read.py`
Expected: PASS — `All Bench Read Tests Passed!`

- [ ] **Step 5: Commit**

```bash
git add src/llm_lab/frontier/bench_read.py tests/test_bench_read.py
git commit -m "feat(m3): unbuffered cold reader via ctypes with sector-rounding and EOF tolerance"
```

---

### Task 8: Synthetic scatter + both-arm benchmark

**Files:**
- Modify: `src/llm_lab/frontier/bench_read.py`
- Test: `tests/test_bench_read.py`

**Interfaces:**
- Consumes: `read_range_cold`, `gguf.GGUFReader`, `llm_lab.frontier.expert_repack.{per_expert_bytes, expert_tensor_name, expert_slice, EXPERT_PARTS}`; stdlib `random`, `time`, `statistics`, `concurrent.futures.ThreadPoolExecutor`.
- Produces:
  - `choose_experts(seed: int, n_layer: int, n_expert: int, k: int) -> dict[int, list[int]]` — `random.Random(seed)`; `k` distinct experts per layer, sorted. Deterministic for a seed.
  - `repacked_ranges(manifest: dict, chosen) -> list[tuple[int, int]]` — for each `(L, e)`, the blob's `(offset, length)` from the manifest (whole-expert single read). Scattered by construction (chosen experts are non-adjacent in a layer-major file).
  - `stock_ranges(reader, chosen, n_expert: int) -> list[tuple[int, int]]` — for each `(L, e, part)`, `(tensor.data_offset + e * per, per)` where `per = per_expert_bytes(tensor, n_expert)`. Proven: `tensor.data_offset` is the absolute file offset; expert `e`'s bytes are `[base + e*per, base + (e+1)*per)`.
  - `working_set_bytes(reader, chosen, n_expert) -> int` — Σ over chosen experts of `gate_per + up_per + down_per` (identical logical payload for both arms).
  - `run_arm(path, ranges, working_set_bytes, file_size, threads, runs) -> dict` — runs `runs` timed passes (each reads all `ranges` cold via a `ThreadPoolExecutor(threads)`), drops run 0, returns `{"min","median","max"}` of `bps = working_set_bytes / seconds` over kept runs, plus `"seconds"` list.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe tests/test_bench_read.py`
Expected: FAIL — `ImportError: cannot import name 'choose_experts'`.

- [ ] **Step 3: Write minimal implementation**

```python
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from llm_lab.frontier.expert_repack import EXPERT_PARTS, expert_tensor_name, per_expert_bytes


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
```

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe tests/test_bench_read.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm_lab/frontier/bench_read.py tests/test_bench_read.py
git commit -m "feat(m3): seeded scatter, both-arm ranges, and timed cold-read arm runner"
```

---

### Task 9: Gate derivation + `moe bench-read` CLI

**Files:**
- Modify: `src/llm_lab/frontier/bench_read.py`
- Modify: `src/llm_lab/__init__.py`
- Test: `tests/test_bench_read.py`

**Interfaces:**
- Consumes: everything from Task 8; `json`, `shutil`, `read_range_cold` (for NVMe); `_emit` in `__init__.py`.
- Produces:
  - `derive_verdict(repacked: dict, stock: dict) -> str` — `"separated_win"` iff `repacked["median"] > stock["median"] and repacked["min"] >= stock["max"]` (full band separation), else `"overlap_inconclusive"`.
  - `derive_m3_gate(output_exact: dict, throughput: dict) -> dict` — builds the final gate: copies `output_exact`, embeds `throughput`, sets `throughput_verdict = throughput["throughput_verdict"]`, derives `branch` (`"proceed_m4"` if passed; `"remeasure"` if verdict is `overlap_inconclusive`; `"fix_output_exact"` if `not all_equal`), and `passed = output_exact["all_equal"] and throughput_verdict == "separated_win"`. Nothing hardcoded.
  - `measure_nvme_seq(path, nbytes, file_size) -> float` — cold sequential `bps` via `read_range_cold` over `nbytes` from offset 0 (single-threaded).
  - `bench_read_model(input_path, repack_dir, *, seed, experts_per_layer, runs, threads) -> dict` — loads `manifest.json`, opens the source reader, builds `chosen`/ranges/`working_set_bytes`, runs both arms at `threads` (QD-proxy) and at `1` (QD=1), re-measures NVMe (~4 GiB), assembles the `throughput` block (both arms' min/median/max, `qd16_ratio = repacked.median/stock.median`, `qd1_ratio`, `nvme_seq_bps`, `experts_per_layer`, `seed`, `runs`, `qd_proxy_threads`, and a `note` that QD is a thread-count proxy), then returns `derive_m3_gate(manifest["output_exact"], throughput)`.
  - CLI `bench-read` subparser: `--input`, `--repack`, `--output` (the gate json), `--seed` (int, default 1234), `--experts-per-layer` (int, default 8), `--runs` (int, default 5), `--threads` (int, default 16), `--force`. Dispatch calls `bench_read_model` and `_emit(gate, args.output)` (single output file → no `_moe_output_paths` change).

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe tests/test_bench_read.py`
Expected: FAIL — `ImportError: cannot import name 'derive_verdict'`.

- [ ] **Step 3: Write minimal implementation**

```python
import json
import os

from llm_lab.frontier.expert_repack import PACKED_NAME, MANIFEST_NAME


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
```

```python
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
```

Then wire `bench-read` in `src/llm_lab/__init__.py` (two edits, `analyze` pattern):

1. In `_add_moe_subparsers`, after the `repack_p` block:

```python
    bench_p = moe_sub.add_parser("bench-read", help="Cold-read throughput: repacked vs stock GGUF")
    bench_p.add_argument("--input", required=True)
    bench_p.add_argument("--repack", required=True)   # the repack output directory
    bench_p.add_argument("--output", required=True)   # m3-gate.json
    bench_p.add_argument("--seed", type=int, default=1234)
    bench_p.add_argument("--experts-per-layer", type=int, default=8)
    bench_p.add_argument("--runs", type=int, default=5)
    bench_p.add_argument("--threads", type=int, default=16)
    bench_p.add_argument("--force", action="store_true")
```

2. In `_run_moe`, after the `repack` branch:

```python
    if args.moe_command == "bench-read":
        from llm_lab.frontier.bench_read import bench_read_model
        gate = bench_read_model(args.input, args.repack, seed=args.seed,
                                experts_per_layer=args.experts_per_layer,
                                runs=args.runs, threads=args.threads)
        _emit(gate, args.output)
        return
```

`bench-read` writes only `--output`, so `_guard_outputs`/`_moe_output_paths` already cover it — no further wiring.

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe tests/test_bench_read.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm_lab/frontier/bench_read.py src/llm_lab/__init__.py tests/test_bench_read.py
git commit -m "feat(m3): derived m3 gate and moe bench-read subcommand"
```

---

### Task 10: Measured run on the real model + evidence + housekeeping

This is the milestone deliverable: run the two commands on the real 18.56 GB GGUF, capture committed evidence, and finish the tracker/branch hygiene. No new unit test — the gate object IS the test, and the derived `passed` is its assertion.

**Files:**
- Create: `artifacts/m3/evidence/{M3-REPORT.md, m3-commands.ps1, m3-gate.json, m3-files.sha256, m3-repack-output.txt, m3-bench-output.txt}`
- Runtime (gitignored): `models/m3/qwen3-repack/{qwen3-experts.packed, manifest.json}`
- Confirm: `models/m3/` and `artifacts/m3/evidence/*.txt`-adjacent raw stay untracked; report artifacts committed.

- [ ] **Step 1: Confirm the full unit suites are green**

Run: `./.venv/Scripts/python.exe tests/test_expert_repack.py && ./.venv/Scripts/python.exe tests/test_bench_read.py && ./.venv/Scripts/python.exe tests/test_moe_trace.py`
Expected: three `All ... Tests Passed!` lines. The pre-existing suite must still pass.

- [ ] **Step 2: Write the exact-command transcript** `artifacts/m3/evidence/m3-commands.ps1`

```powershell
# M3 expert-contiguous repack — exact commands run on this machine/session.
# Step A: repack -> packed file + manifest.json; discharges the output-exact (byte-identity) gate.
./.venv/Scripts/python.exe -m llm_lab moe repack `
  --input models/m1/qwen3-gguf/Qwen3-30B-A3B-Q4_K_M.gguf `
  --output models/m3/qwen3-repack `
  2>&1 | Tee-Object artifacts/m3/evidence/m3-repack-output.txt

# Step B: bench-read -> m3-gate.json (stdout is byte-identical to the file, via _emit).
./.venv/Scripts/python.exe -m llm_lab moe bench-read `
  --input models/m1/qwen3-gguf/Qwen3-30B-A3B-Q4_K_M.gguf `
  --repack models/m3/qwen3-repack `
  --output artifacts/m3/evidence/m3-gate.json `
  --seed 1234 --experts-per-layer 8 --runs 5 --threads 16 `
  2>&1 | Tee-Object artifacts/m3/evidence/m3-bench-output.txt

# Step C: content-address every evidence file (except the manifest) + the runtime manifest.json.
./.venv/Scripts/python.exe -c "import hashlib,glob; [print(hashlib.sha256(open(f,'rb').read()).hexdigest(), f) for f in sorted([p for p in glob.glob('artifacts/m3/evidence/*') if not p.endswith('m3-files.sha256')]+['models/m3/qwen3-repack/manifest.json'])]" > artifacts/m3/evidence/m3-files.sha256
```

- [ ] **Step 3: Run the repack on the real model**

Run: Step A of `m3-commands.ps1`.
Expected: preflight prints `required=… available=…` then `PREFLIGHT_OK`; final line `repacked 6144 blobs; output_exact.all_equal=True n_slices=18432`. Confirm `models/m3/qwen3-repack/{qwen3-experts.packed, manifest.json}` exist and the packed size ≈ 16 GB. (Headroom note: free ≈ 37 GiB, output ≈ 15 GiB — comfortable; the earlier "~3 GB" figure double-counted the on-disk source. The preflight uses the live `shutil.disk_usage().free`, so it is self-correcting regardless.)

- [ ] **Step 4: Run the throughput benchmark**

Run: Step B of `m3-commands.ps1`.
Expected: `artifacts/m3/evidence/m3-gate.json` written; inspect `output_exact.all_equal` (must be `true`), `throughput.qd16_ratio`, `throughput_verdict`, `branch`, `passed`. If `throughput_verdict == "overlap_inconclusive"`: per the Q12 ruling, re-run with `--runs 9` to tighten bands before concluding; if still overlapping, STOP and surface to the human whether M4's case rests on layout read-throughput or on its async/cache machinery — do not silently pass or fail.

- [ ] **Step 5: Generate the sha256 manifest**

Run: Step C of `m3-commands.ps1`.
Expected: `m3-files.sha256` lists every evidence file + `manifest.json`; `m3-gate.json` and `m3-bench-output.txt` share a hash (byte-identical, like M2).

- [ ] **Step 6: Confirm `models/m3/` is gitignored, author `M3-REPORT.md`**

Verify `git status --short` does NOT list `models/m3/` (the `models/` tree is gitignored like `models/m1`); if it appears, add `models/` (or `models/m3/`) to `.gitignore`. Then hand-author `artifacts/m3/evidence/M3-REPORT.md` (prose, mirroring `M2-REPORT.md`). It MUST:
- State M3 is MEASURED, not projected; cite `m3-gate.json` fields by name (`output_exact.rollup_sha256`, `throughput.qd16_ratio`, `throughput_verdict`, `passed`).
- Report the **re-measured** `nvme_seq_bps` as the in-session NVMe figure and explicitly say it replaces §3's 2.62 GB/s (never quote §3 as an input).
- Describe the fair comparison in one line: repacked = 8 aligned single-blob reads/token at real scattered offsets; stock = 24 scattered unaligned role-slice reads/token; same experts, same primitive, same QD-proxy; stock sector-rounding is the honest cost of misalignment.
- Label `output_exact` as byte-identity (not inference); note §11-Q1 stays open and M3 does not depend on it.
- Carry the M2 line: "nothing here was hand-tuned toward a verdict" — `passed`/`branch` are derived by `derive_m3_gate`.
- If the verdict was `overlap_inconclusive`, report it honestly as "no beyond-noise read-throughput win demonstrated at QD-proxy ≥16," not "layout useless," and record the decision taken.

- [ ] **Step 7: Create the three missing triage labels**

Run:
```bash
gh label create needs-info --repo MdRaf1/llm-lab --color BFD4F2 --description "Blocked pending information"
gh label create ready-for-agent --repo MdRaf1/llm-lab --color 0E8A16 --description "Ready for an agent to pick up"
gh label create ready-for-human --repo MdRaf1/llm-lab --color 5319E7 --description "Needs a human decision"
```
Expected: three labels created (or "already exists" — idempotent, non-fatal). This makes the tracker match `docs/agents/triage-labels.md`.

- [ ] **Step 8: Commit the evidence**

```bash
git add artifacts/m3/evidence/M3-REPORT.md artifacts/m3/evidence/m3-commands.ps1 artifacts/m3/evidence/m3-gate.json artifacts/m3/evidence/m3-files.sha256 artifacts/m3/evidence/m3-repack-output.txt artifacts/m3/evidence/m3-bench-output.txt
git commit -m "docs(m3): measured expert-contiguous repack evidence and gate"
```

Do NOT `git add models/m3/` (gitignored, ~16 GB). Stage files by name, as above.

---

## Execution Handoff

After the last task, integrate the `m3` branch per `superpowers:finishing-a-development-branch`: PR into `master`, `--no-ff` merge mirroring the M2 merge, then update issue **#1** to reference this plan + the merged evidence and close it. The commit trailer for every M3 commit:

```
Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```

