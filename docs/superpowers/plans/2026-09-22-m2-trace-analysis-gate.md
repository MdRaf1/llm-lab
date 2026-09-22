# M2 — Trace Analysis → Decision Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the surviving MoE routing traces into cache-hit curves, union growth, reuse distance, and co-activation structure, then project a per-memory-tier tok/s ceiling for Qwen3-30B-A3B and emit a machine-checkable decision-gate verdict.

**Architecture:** A new analysis-only module (`src/llm_lab/moe/analysis.py`) builds on the existing `window_unions`/`replay_cache` in `trace.py`. All measured numbers come from the 500 surviving **int8-OLMoE** positions; every 30B number is a `PROJECTED` value obtained by the **normalized working-set-fraction** scheme — measure hit-rate as a function of f (cache capacity ÷ total experts) on OLMoE, apply that curve to 30B geometry. No loader, no repack, no llama.cpp patch.

**Tech Stack:** Python 3, stdlib only for analysis (`collections.OrderedDict`, `bisect`, `json`, `hashlib`); `gguf` (already a dep) for the exact core measurement; the project's assert-based `tests/test_moe_trace.py` self-check (no pytest).

**Spec:** `docs/superpowers/specs/2026-09-11-moe-exact-runtime-design.md` (§7 milestone + gate, §8 trace schema, §10 honesty rules, §11 open questions)

## Global Constraints

- **Python entry point:** `./.venv/Scripts/python.exe` — bare `python`/`hf` are not on PATH. Tools live under `.venv/Scripts/*.exe`. No pytest.
- **Run the suite with:** `./.venv/Scripts/python.exe tests/test_moe_trace.py` (expects the trailing line `All MoE Trace Tests Passed!`). Tests are auto-discovered: any top-level `def test_*` runs.
- **Honesty rule #1:** No number in a report, `m2-gate.json`, or commit message unless its command and output are saved in the repo.
- **Honesty rule #2:** Baselines measured on the same machine/session/checkpoint — never a literature constant.
- **Honesty rule #3 (central to M2):** Every projected number is labeled `PROJECTED` with its formula and inputs shown. M2 carries **two** labeled projection hops: int8→Q4 precision, and OLMoE→30B family.
- **Honesty rule #4:** Output identity is token-ID equality, never `.strip()` text comparison.
- **Honesty rule #5:** Q4 and BF16/int8 are different numeric paths — never assert cross-path routing/token identity.
- **Threat-to-validity stamp:** every projected 30B number carries "valid for the Q4 runtime only if routing is precision-stable — §11-Q1 UNRESOLVED".
- **Cold bandwidth (measured, §3):** `B_cold = 2.62 GB/s` NVMe sequential. Decimal GB (÷10⁹) throughout the gate arithmetic; byte fields stay raw bytes.
- **Gate arithmetic:** `tok/s = B_cold ÷ cold_bytes_per_token`; `cold_bytes_per_token = miss_requests_per_token × avg_expert_bytes`.
- **Gate thresholds (§7):** ceiling **≥ 10 tok/s** → build M3/M4 on projection; **≤ 5 tok/s** (miss > 524 MB/token) → escalate M5/M6 on projection; **∈ (5,10) tok/s**, OR inter-domain spread across the 4 traces straddles a gate line → **re-rent** the 30B BF16 trace. No point estimate without its band.
- **Tier ladder:** RAM ∈ {4, 8, 16, 32} GB; 16 GB is *the* gate tier. Reserve a **flat ~2.5 GB on every tier** for OS+KV+activations — OS/runtime/KV overhead is roughly constant, not proportional to total RAM, so the same reserve applies to each tier (the 16 GB gate tier is invariant to this choice). Expert cache = RAM − resident weight core − reserve. GPU/VRAM tier is **out of M2 scope** (deferred to M4).
- **Out of scope:** open-Q1 (Q4-vs-BF16 routing identity), any loader/streaming/repack code, any llama.cpp C++ patch, any speedup claim.

---

### Task 1: Exact resident weight core in `meta.py`

The M2 "core fits a tier" gate clause needs the non-expert resident weight bytes (attn + embed + output + norms + router). Today `meta.py` measures only `total_expert_bytes`; the ~1 GB core figure is `file_size − total_expert_bytes` done outside the module, which includes GGUF metadata/padding and is therefore an **upper bound** that *overstates* the core — the wrong direction for a "publicly drop this tier" decision. `read_gguf_meta` already iterates `reader.tensors`, so sum the non-expert tensors for an exact figure. The HF path cannot see tensor bytes → `nonexpert_bytes=None`.

**Files:**
- Modify: `src/llm_lab/moe/meta.py` — `ModelMeta` dataclass (add field + `to_dict`), `read_gguf_meta` (measure), `read_hf_meta` (pass `None`)
- Test: `tests/test_moe_trace.py` (append; reuses existing `write_gguf`, `expert_tensors` helpers)

**Interfaces:**
- Consumes: existing `gguf.GGUFReader`, `_EXPERT_TENSOR_PARTS` (= gate/up/down), `ModelMeta`.
- Produces: `ModelMeta.nonexpert_bytes: int | None` — exact summed bytes of every tensor whose name does **not** end in `_exps.weight` (GGUF), else `None` (HF). `to_dict()` emits key `"nonexpert_bytes"`.

- [ ] **Step 1: Write the failing test**

```python
def test_gguf_meta_measures_nonexpert_core_exactly():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "m.gguf"
        # 2 layers, 4 experts, hidden 8, ffn 8; expert_tensors() emits the ffn_*_exps.weight set.
        tensors = expert_tensors(n_layer=2, n_expert=4, hidden=8, ffn=8)
        # Two non-expert tensors of known size: token_embd + one attn matrix.
        tensors["token_embd.weight"] = np.zeros((8, 8), dtype=np.float32)      # 256 B
        tensors["blk.0.attn_q.weight"] = np.zeros((8, 8), dtype=np.float32)    # 256 B
        write_gguf(path, {"olmoe.block_count": 2, "olmoe.expert_count": 4,
                          "olmoe.expert_used_count": 2}, tensors, arch="olmoe")
        meta = read_gguf_meta(path)
        assert meta.nonexpert_bytes == 512, meta.nonexpert_bytes
        # The expert tensors must NOT be counted in the core.
        assert meta.total_expert_bytes > 0
        assert meta.to_dict()["nonexpert_bytes"] == 512
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe tests/test_moe_trace.py`
Expected: FAIL — `AttributeError: 'ModelMeta' object has no attribute 'nonexpert_bytes'` (or `TypeError` on the missing constructor arg).

- [ ] **Step 3: Add the field to `ModelMeta`**

In the `ModelMeta` dataclass add after `total_expert_bytes`:

```python
    nonexpert_bytes: int | None
```

In `__post_init__` add:

```python
        _check_non_negative("nonexpert_bytes", self.nonexpert_bytes, optional=True)
```

In `to_dict` add `"nonexpert_bytes": self.nonexpert_bytes,` to the returned dict.

- [ ] **Step 4: Measure it in `read_gguf_meta`**

Inside `read_gguf_meta`, after the expert-tensor loop (it already has `tensors = {tensor.name: tensor for tensor in reader.tensors}`), before the `return`:

```python
    # Exact resident weight core: every tensor that is not a per-expert FFN blob.
    # Excludes GGUF metadata/padding by construction, so it is exact, not the file-size upper bound.
    nonexpert_bytes = sum(
        t.n_bytes for name, t in tensors.items() if not name.endswith("_exps.weight")
    )
```

Add `nonexpert_bytes=nonexpert_bytes,` to the `ModelMeta(...)` construction.

- [ ] **Step 5: Pass `None` on the HF path**

In `read_hf_meta`, add `nonexpert_bytes=None,` to its `ModelMeta(...)` construction (HF config cannot measure tensor bytes; consistent with `expert_bytes=None` there).

- [ ] **Step 6: Run tests to verify green**

Run: `./.venv/Scripts/python.exe tests/test_moe_trace.py`
Expected: PASS, ending `All MoE Trace Tests Passed!` (existing meta tests still pass because the new field is keyword-constructed).

- [ ] **Step 7: Commit**

```bash
git add src/llm_lab/moe/meta.py tests/test_moe_trace.py
git commit -m "feat(moe): measure exact non-expert resident core from GGUF tensors"
```

---

### Task 2: Reuse distance (LRU stack distance) in `analysis.py`

Reuse distance = the number of **distinct** experts seen between two consecutive requests of the same (layer, expert) blob — the canonical LRU miss-curve predictor, consistent with `replay_cache`'s LRU. First sight of a blob has no finite distance (`None`, "cold").

**Files:**
- Create: `src/llm_lab/moe/analysis.py`
- Test: `tests/test_moe_trace.py` (append)

**Interfaces:**
- Consumes: `TraceStep` and the request order defined by `trace._requests` (step by step, layer by layer, expert by expert). Re-derive that order locally rather than importing the private helper.
- Produces:
  - `stack_distances(steps: Sequence[TraceStep]) -> list[int | None]` — one entry per request in request order; `None` on first sight.
  - `reuse_distance_histogram(steps) -> dict[str, object]` — `{"histogram": {str(d): count}, "cold": int, "total_requests": int}` (keys are stringified ints so the dict is JSON-safe).

- [ ] **Step 1: Write the failing test**

```python
from llm_lab.moe.analysis import stack_distances, reuse_distance_histogram

def _steps(layer_experts_per_pos):
    from llm_lab.moe.trace import TraceStep
    return [TraceStep(kind="step", pos=i, tok_id=i, layer_experts=le)
            for i, le in enumerate(layer_experts_per_pos)]

def test_stack_distance_counts_distinct_between_reuses():
    # One layer, requests in order: A A B A
    #   A(cold) A(dist 0: nothing distinct since last A)
    #   B(cold) A(dist 1: only B seen since last A)
    steps = _steps([[[0]], [[0]], [[1]], [[0]]])
    assert stack_distances(steps) == [None, 0, None, 1]
    hist = reuse_distance_histogram(steps)
    assert hist == {"histogram": {"0": 1, "1": 1}, "cold": 2, "total_requests": 4}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe tests/test_moe_trace.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_lab.moe.analysis'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/llm_lab/moe/analysis.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe tests/test_moe_trace.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm_lab/moe/analysis.py tests/test_moe_trace.py
git commit -m "feat(moe): compute LRU stack-distance reuse histogram"
```

---

### Task 3: Per-layer sparse co-activation pair counts

Co-activation = how often two experts are selected **in the same layer at the same position** (SP-MoE Fig 2 style, §5). Store per-layer sparse `{(e_lo, e_hi): count}` with `e_lo < e_hi` — never a dense `n_expert²×n_layer` tensor.

**Files:**
- Modify: `src/llm_lab/moe/analysis.py`
- Test: `tests/test_moe_trace.py` (append)

**Interfaces:**
- Consumes: `TraceStep`.
- Produces: `coactivation_pairs(steps) -> dict[int, dict[tuple[int, int], int]]` — `{layer: {(e_lo, e_hi): count}}`; a helper `coactivation_to_json(pairs) -> dict` renders it JSON-safe as `{str(layer): {"e_lo,e_hi": count}}`.

- [ ] **Step 1: Write the failing test**

```python
from llm_lab.moe.analysis import coactivation_pairs, coactivation_to_json

def test_coactivation_counts_within_layer_unordered_pairs():
    # Layer 0 selects {0,1,2} at pos0 and {0,1} at pos1; layer 1 selects {3,4} once.
    steps = _steps([[[0, 1, 2], [3, 4]], [[0, 1], [5, 6]]])
    pairs = coactivation_pairs(steps)
    # Layer 0: (0,1) co-occurs twice; (0,2),(1,2) once each.
    assert pairs[0][(0, 1)] == 2
    assert pairs[0][(0, 2)] == 1 and pairs[0][(1, 2)] == 1
    # Layer 1: (3,4) once, (5,6) once; no cross-layer or cross-position mixing.
    assert pairs[1][(3, 4)] == 1 and pairs[1][(5, 6)] == 1
    assert (3, 5) not in pairs[1]
    assert coactivation_to_json(pairs)["0"]["0,1"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe tests/test_moe_trace.py`
Expected: FAIL — `ImportError: cannot import name 'coactivation_pairs'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/llm_lab/moe/analysis.py`:

```python
from itertools import combinations


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe tests/test_moe_trace.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm_lab/moe/analysis.py tests/test_moe_trace.py
git commit -m "feat(moe): count per-layer sparse expert co-activation pairs"
```

---
### Task 4: Hit-rate curve over the working-set-fraction grid

Sweep `replay_cache` over capacities = `round(f × n_expert_total)` and record hit-rate as a function of f. This is the transferable invariant (Q1 decision (a)): the curve is measured on OLMoE, then read at 30B's per-tier f. `n_expert_total = n_layer × n_expert` (total distinct blobs, i.e. the working set at f=1).

**Files:**
- Modify: `src/llm_lab/moe/analysis.py`
- Test: `tests/test_moe_trace.py` (append)

**Interfaces:**
- Consumes: `trace.replay_cache(steps, capacity_experts, logits_sha256) -> {"requests","hits","misses",...}`.
- Produces: `hit_rate_curve(steps, fractions, n_expert_total, logits_sha256) -> list[dict]` — one dict per f, sorted by fraction: `{"fraction": float, "capacity_experts": int, "hits": int, "misses": int, "requests": int, "hit_rate": float, "miss_rate": float}`. `hit_rate = hits/requests` (0.0 when `requests==0`).

- [ ] **Step 1: Write the failing test**

```python
from llm_lab.moe.analysis import hit_rate_curve

def test_hit_rate_curve_invariants_f0_f1_and_monotonic():
    # 3 positions, 1 layer, top-1, experts 0,1,2,0 -> working set of 3 distinct blobs.
    steps = _steps([[[0]], [[1]], [[2]], [[0]]])
    curve = hit_rate_curve(steps, [0.0, 1/3, 2/3, 1.0], n_expert_total=3, logits_sha256="x")
    by_f = {round(row["fraction"], 4): row for row in curve}
    # f=0 -> capacity 0 -> every request misses.
    assert by_f[0.0]["hit_rate"] == 0.0 and by_f[0.0]["misses"] == 4
    # f=1 -> capacity 3 >= working set -> only the 3 cold misses, the final 0 is a hit.
    assert by_f[1.0]["misses"] == 3 and by_f[1.0]["hits"] == 1
    # Monotonic non-decreasing hit_rate in f.
    rates = [row["hit_rate"] for row in curve]
    assert rates == sorted(rates)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe tests/test_moe_trace.py`
Expected: FAIL — `ImportError: cannot import name 'hit_rate_curve'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/llm_lab/moe/analysis.py`:

```python
from llm_lab.moe.trace import replay_cache


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe tests/test_moe_trace.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm_lab/moe/analysis.py tests/test_moe_trace.py
git commit -m "feat(moe): sweep hit-rate over working-set-fraction grid"
```

---

### Task 5: Projection, tier model, and gate verdict

Turn a hit-rate curve + geometry into a `PROJECTED` tok/s per tier and the branch verdict. This is where honesty rule #3 lives: formula and inputs travel with every number.

Definitions (all decimal GB in the arithmetic; bytes in the byte fields):
- `avg_expert_bytes = total_expert_bytes / n_expert_total` (Q4_K_M is mixed-quant, so the average is the honest per-blob size).
- `requests_per_token = n_layer × n_expert_used`.
- For a tier: `expert_cache_bytes = ram_bytes − nonexpert_bytes − reserve_bytes`; if `≤ 0`, the core does **not** fit → tier dropped. `capacity_experts = floor(expert_cache_bytes / avg_expert_bytes)`; `f = capacity_experts / n_expert_total`.
- Read `miss_rate` at that f from the (OLMoE-measured) curve by **linear interpolation** between bracketing grid points.
- `miss_requests_per_token = requests_per_token × miss_rate`; `cold_bytes_per_token = miss_requests_per_token × avg_expert_bytes`; `tok_s = B_cold_bytes_s / cold_bytes_per_token`.

**Files:**
- Modify: `src/llm_lab/moe/analysis.py`
- Test: `tests/test_moe_trace.py` (append)

**Interfaces:**
- Consumes: `hit_rate_curve` output.
- Produces:
  - `interp_miss_rate(curve, f) -> float` — linear interpolation of `miss_rate` at fraction `f` (clamped to the curve's endpoints).
  - `project_tier(ram_bytes, nonexpert_bytes, reserve_bytes, avg_expert_bytes, n_expert_total, n_layer, n_expert_used, curve, b_cold_bytes_s) -> dict` — `{"ram_bytes","core_fits":bool,"capacity_experts","fraction","miss_rate","miss_bytes_per_token","tok_s"}`; when `core_fits` is False, numeric fields are `None`.
  - `gate_branch(tok_s_16gb, spreads_straddle: bool) -> str` — returns `"build_m3m4"` (≥10), `"escalate_m5m6"` (≤5), or `"re_rent"` (in (5,10) **or** `spreads_straddle`).

- [ ] **Step 1: Write the failing test**

```python
from llm_lab.moe.analysis import interp_miss_rate, project_tier, gate_branch

def test_tok_s_arithmetic_and_gate_bands_are_exact():
    # Curve: at f=0 miss_rate 1.0, at f=1 miss_rate 0.5, linear between.
    curve = [{"fraction": 0.0, "miss_rate": 1.0}, {"fraction": 1.0, "miss_rate": 0.5}]
    assert interp_miss_rate(curve, 0.5) == 0.75
    assert interp_miss_rate(curve, 2.0) == 0.5   # clamped to the last point

    # Geometry chosen so the arithmetic is checkable by hand:
    # avg_expert = 1e6 B, requests/token = 4, ram leaves capacity for the whole set (f=1, miss 0.5).
    tier = project_tier(
        ram_bytes=10_000_000, nonexpert_bytes=1_000_000, reserve_bytes=0,
        avg_expert_bytes=1_000_000, n_expert_total=9, n_layer=2, n_expert_used=2,
        curve=[{"fraction": 0.0, "miss_rate": 1.0}, {"fraction": 1.0, "miss_rate": 0.5}],
        b_cold_bytes_s=2_620_000_000,
    )
    # cache bytes = 9e6 -> capacity 9 -> f=1.0 -> miss_rate 0.5
    # miss/token = 4 * 0.5 = 2 -> cold bytes/token = 2 * 1e6 = 2e6
    # tok/s = 2.62e9 / 2e6 = 1310.0
    assert tier["core_fits"] is True and tier["capacity_experts"] == 9
    assert tier["miss_bytes_per_token"] == 2_000_000
    assert tier["tok_s"] == 1310.0

    # Core larger than RAM -> dropped.
    dropped = project_tier(
        ram_bytes=500_000, nonexpert_bytes=1_000_000, reserve_bytes=0,
        avg_expert_bytes=1_000_000, n_expert_total=9, n_layer=2, n_expert_used=2,
        curve=curve, b_cold_bytes_s=2_620_000_000,
    )
    assert dropped["core_fits"] is False and dropped["tok_s"] is None

    assert gate_branch(12.0, False) == "build_m3m4"
    assert gate_branch(4.0, False) == "escalate_m5m6"
    assert gate_branch(7.0, False) == "re_rent"
    assert gate_branch(12.0, True) == "re_rent"   # spread straddles a line -> re-rent regardless
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe tests/test_moe_trace.py`
Expected: FAIL — `ImportError: cannot import name 'interp_miss_rate'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/llm_lab/moe/analysis.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe tests/test_moe_trace.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm_lab/moe/analysis.py tests/test_moe_trace.py
git commit -m "feat(moe): project per-tier tok/s and emit gate branch verdict"
```

---
### Task 6: `analyze_traces` orchestrator + `moe analyze` CLI subcommand

One orchestrator builds the whole M2 object from the curve-source traces (OLMoE) plus the target geometry (30B). Union growth reuses `window_unions`. The curve is computed **per trace** (each trace is an independent decode, so the LRU cache resets per trace), then averaged; the per-trace 16 GB tok/s spread feeds the re-rent trigger.

**Files:**
- Modify: `src/llm_lab/moe/analysis.py` (add `union_growth`, `analyze_traces`)
- Modify: `src/llm_lab/__init__.py` (add the `analyze` subparser + dispatch branch; extend `_moe_output_paths`)
- Test: `tests/test_moe_trace.py` (append: orchestrator unit test + help-inventory update)

**Interfaces:**
- Consumes: `trace.read_trace`, `trace.window_unions`, `baseline.sha256_file`; Tasks 2–5 functions.
- Produces:
  - `union_growth(steps, windows) -> list[dict]` — `[{"window": w, "mean_union": float, "max_union": int}]`.
  - `analyze_traces(traces: list[tuple[str, Sequence[TraceStep], str]], geometry: dict, fractions, windows, tiers_gb, reserve_bytes, b_cold_bytes_s) -> dict` where each trace tuple is `(name, steps, logits_sha256)` and `geometry` has `n_layer,n_expert,n_expert_used,total_expert_bytes,nonexpert_bytes`. Returns the full gate object (see keys asserted below).
- CLI: `moe analyze --trace PATH --logits PATH` (both repeatable, zipped 1:1) `--geometry meta.json --output OUT [--force]`, with defaults for `--fractions`, `--windows`, `--reserve-bytes 2500000000`, `--b-cold-bytes 2620000000`, `--tiers-gb 4,8,16,32`.

- [ ] **Step 1: Write the failing test**

```python
def test_analyze_traces_emits_gate_object_and_cli_runs(tmp_path=None):
    import tempfile, json as _json
    from llm_lab.moe.analysis import analyze_traces
    steps = _steps([[[0]], [[1]], [[2]], [[0]]])
    geometry = {"n_layer": 1, "n_expert": 3, "n_expert_used": 1,
                "total_expert_bytes": 3_000_000, "nonexpert_bytes": 1_000_000}
    obj = analyze_traces(
        traces=[("t0", steps, "shaX")], geometry=geometry,
        fractions=[0.0, 0.5, 1.0], windows=[1, 2, 4],
        tiers_gb=[4, 8, 16, 32], reserve_bytes=0, b_cold_bytes_s=2_620_000_000)
    assert obj["projection_scheme"] == "normalized-working-set-fraction"
    assert obj["branch"] in {"build_m3m4", "escalate_m5m6", "re_rent"}
    assert {t["ram_bytes"] for t in obj["tiers"]} == {4e9, 8e9, 16e9, 32e9}
    assert "projected" in obj["threat_to_validity"].lower() or "§11" in obj["threat_to_validity"]
    assert len(obj["per_trace"]) == 1 and obj["per_trace"][0]["name"] == "t0"

    with tempfile.TemporaryDirectory() as d:
        from pathlib import Path as _P
        tp, lp = _P(d) / "t.jsonl", _P(d) / "l.pt"
        from llm_lab.moe.trace import write_trace
        write_trace(tp, sample_header(n_layer=1, n_expert=3, n_expert_used=1), steps)
        lp.write_bytes(b"logits")
        gp = _P(d) / "geo.json"; gp.write_text(_json.dumps(geometry))
        op = _P(d) / "out.json"
        main(["moe", "analyze", "--trace", str(tp), "--logits", str(lp),
              "--geometry", str(gp), "--output", str(op)])
        assert _json.loads(op.read_text())["branch"] in {"build_m3m4", "escalate_m5m6", "re_rent"}
```

Note: `sample_header` and `write_trace` already exist in the test module; `sample_header` accepts the `n_layer/n_expert/n_expert_used` overrides (verify its signature and pass matching values so `_validate_steps` accepts the single-layer top-1 steps).

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe tests/test_moe_trace.py`
Expected: FAIL — `ImportError: cannot import name 'analyze_traces'`.

- [ ] **Step 3: Add `union_growth` and `analyze_traces`**

Append to `src/llm_lab/moe/analysis.py`:

```python
from statistics import mean

from llm_lab.moe.trace import window_unions


def union_growth(steps, windows) -> list[dict]:
    """Distinct (layer,expert) blobs touched in a sliding window, vs window length."""
    rows = []
    for w in sorted(set(windows)):
        sizes = [len(u) for u in window_unions(steps, w)] or [0]
        rows.append({"window": w, "mean_union": mean(sizes), "max_union": max(sizes)})
    return rows


def analyze_traces(traces, geometry, fractions, windows, tiers_gb,
                   reserve_bytes, b_cold_bytes_s) -> dict:
    n_layer = geometry["n_layer"]
    n_expert_total = n_layer * geometry["n_expert"]
    n_expert_used = geometry["n_expert_used"]
    avg_expert_bytes = geometry["total_expert_bytes"] / n_expert_total
    nonexpert_bytes = geometry["nonexpert_bytes"]

    per_trace, curves, tok_s_16gb = [], [], []
    for name, steps, logits_sha in traces:
        curve = hit_rate_curve(steps, fractions, n_expert_total, logits_sha)
        curves.append(curve)
        t16 = project_tier(16e9, nonexpert_bytes, reserve_bytes, avg_expert_bytes,
                           n_expert_total, n_layer, n_expert_used, curve, b_cold_bytes_s)
        if t16["tok_s"] is not None:
            tok_s_16gb.append(t16["tok_s"])
        per_trace.append({
            "name": name, "logits_sha256": logits_sha,
            "hit_rate_curve": curve,
            "reuse_distance": reuse_distance_histogram(steps),
            "coactivation": coactivation_to_json(coactivation_pairs(steps)),
            "union_growth": union_growth(steps, windows),
            "tok_s_16gb": t16["tok_s"],
        })

    # Mean curve across traces at each fraction (curves share the fraction grid).
    mean_curve = [{"fraction": f,
                   "miss_rate": mean(interp_miss_rate(c, f) for c in curves)}
                  for f in sorted(set(fractions))]

    tiers = [project_tier(gb * 1e9, nonexpert_bytes, reserve_bytes, avg_expert_bytes,
                          n_expert_total, n_layer, n_expert_used, mean_curve, b_cold_bytes_s)
             for gb in tiers_gb]
    tier16 = next(t for t in tiers if t["ram_bytes"] == 16e9)
    point_16 = tier16["tok_s"]

    # Spread straddles a gate line if the per-trace 16 GB tok/s cross 5 or 10.
    lo, hi = (min(tok_s_16gb), max(tok_s_16gb)) if tok_s_16gb else (point_16, point_16)
    straddle = any(lo < line < hi for line in (5.0, 10.0))
    branch = gate_branch(point_16 if point_16 is not None else 0.0, straddle)

    return {
        "projection_scheme": "normalized-working-set-fraction",
        "geometry": geometry,
        "avg_expert_bytes": avg_expert_bytes,
        "b_cold_bytes_s": b_cold_bytes_s,
        "reserve_bytes": reserve_bytes,
        "mean_curve": mean_curve,
        "tiers": tiers,
        "projected_ceiling_tok_s": point_16,
        "miss_bytes_per_token": tier16["miss_bytes_per_token"],
        "band_16gb_tok_s": [lo, hi],
        "spreads_straddle_gate_line": straddle,
        "branch": branch,
        "per_trace": per_trace,
        "threat_to_validity": ("All 30B numbers are PROJECTED (int8-OLMoE -> Q4-30B, two hops); "
                               "valid for the Q4 runtime only if routing is precision-stable "
                               "-- §11-Q1 UNRESOLVED."),
    }
```

- [ ] **Step 4: Wire the CLI**

In `src/llm_lab/__init__.py`, after the `replay-cache` subparser block, add:

```python
    analyze_p = moe_sub.add_parser("analyze", help="Derive cache/reuse/co-activation curves and the tier gate")
    analyze_p.add_argument("--trace", action="append", required=True)
    analyze_p.add_argument("--logits", action="append", required=True)
    analyze_p.add_argument("--geometry", required=True)
    analyze_p.add_argument("--output", required=True)
    analyze_p.add_argument("--fractions", default="0.02,0.05,0.1,0.15,0.2,0.3,0.5,0.75,1.0")
    analyze_p.add_argument("--windows", default="1,2,4,8,16,32,64,128,256,512")
    analyze_p.add_argument("--tiers-gb", default="4,8,16,32")
    analyze_p.add_argument("--reserve-bytes", type=int, default=2_500_000_000)
    analyze_p.add_argument("--b-cold-bytes", type=int, default=2_620_000_000)
    analyze_p.add_argument("--force", action="store_true")
```

Extend `_moe_output_paths` so the guard covers the new command (it already returns `[args.output]` for non-`trace` commands — confirm `analyze` falls into that default branch; it does, no change needed).

In `_run_moe`, add before the final return path:

```python
    if args.moe_command == "analyze":
        import json as _json
        from llm_lab.moe.analysis import analyze_traces
        from llm_lab.moe.baseline import sha256_file
        from llm_lab.moe.trace import read_trace
        if len(args.trace) != len(args.logits):
            _reject(parser, "analyze needs one --logits per --trace")
        traces = []
        for tpath, lpath in zip(args.trace, args.logits):
            _, steps = read_trace(Path(tpath))
            traces.append((Path(tpath).name, steps, sha256_file(Path(lpath))))
        geometry = _json.loads(Path(args.geometry).read_text(encoding="utf-8"))
        result = analyze_traces(
            traces, geometry,
            fractions=[float(x) for x in args.fractions.split(",")],
            windows=[int(x) for x in args.windows.split(",")],
            tiers_gb=[int(x) for x in args.tiers_gb.split(",")],
            reserve_bytes=args.reserve_bytes, b_cold_bytes_s=args.b_cold_bytes)
        _emit(result, args.output)
        return
```

- [ ] **Step 5: Update the help-inventory test**

The existing `moe --help` inventory test lists the commands. Add `"analyze"` to its expected-commands tuple so a missing subcommand fails loudly:

```python
    for command in ("meta", "run", "trace", "compare", "replay-cache", "analyze"):
        assert command in text, f"moe --help omits the {command!r} command"
```

(Leave the forbidden-wording assertions unchanged — `analyze`'s help text must not leak "speedup", "PageCC", "GPU", "lossless to BF16".)

- [ ] **Step 6: Run tests to verify green**

Run: `./.venv/Scripts/python.exe tests/test_moe_trace.py`
Expected: PASS, ending `All MoE Trace Tests Passed!`.

- [ ] **Step 7: Commit**

```bash
git add src/llm_lab/moe/analysis.py src/llm_lab/__init__.py tests/test_moe_trace.py
git commit -m "feat(moe): add moe analyze command producing the M2 tier gate object"
```

---

### Task 7: Run M2 on the surviving traces and write the report + gate artifact

No new code. Produce the milestone deliverable: run `moe analyze` on the 5 surviving int8-OLMoE traces with the **30B** geometry, save the command output, and write the report ending in the branch verdict. Every number traces to a saved command (honesty rule #1).

**Files:**
- Create: `artifacts/m2/evidence/m2-geometry-30b.json` (from `moe meta` on the local 30B GGUF)
- Create: `artifacts/m2/evidence/m2-analyze-output.txt` (stdout+stderr of the analyze run)
- Create: `artifacts/m2/evidence/m2-gate.json` (the emitted gate object)
- Create: `artifacts/m2/evidence/m2-commands.ps1` (exact commands run)
- Create: `artifacts/m2/evidence/m2-files.sha256` (hashes of every artifact above + the 5 input traces)
- Create: `artifacts/m2/evidence/M2-REPORT.md`

- [ ] **Step 1: Capture the 30B geometry (measured, incl. exact core from Task 1)**

Run and save:

```bash
./.venv/Scripts/python.exe -m llm_lab moe meta \
  --backend gguf --model-path models/m1/qwen3-gguf/Qwen3-30B-A3B-Q4_K_M.gguf \
  --output artifacts/m2/evidence/m2-geometry-30b.json
```

Confirm the JSON shows `n_layer=48, n_expert=128, n_expert_used=8, total_expert_bytes=17553162240, nonexpert_bytes≈997554176` (exact non-expert core, ~0.93 GiB — **not** the 1.00 GB file-minus-experts upper bound).

- [ ] **Step 2: Run the analysis on all 5 OLMoE traces**

The OLMoE logits `.pt` files are present under `artifacts/m1/raw/`. Pair each trace with its logits file (1:1) and tee output to the evidence file:

```bash
./.venv/Scripts/python.exe -m llm_lab moe analyze \
  --trace artifacts/m1/raw/m1b-trace-prose-a.jsonl --logits artifacts/m1/raw/m1b-logits-prose-a.pt \
  --trace artifacts/m1/raw/m1b-trace-prose-b.jsonl --logits artifacts/m1/raw/m1b-logits-prose-b.pt \
  --trace artifacts/m1/raw/m1b-trace-code-a.jsonl  --logits artifacts/m1/raw/m1b-logits-code-a.pt \
  --trace artifacts/m1/raw/m1b-trace-code-b.jsonl  --logits artifacts/m1/raw/m1b-logits-code-b.pt \
  --trace artifacts/m1/raw/m1b-trace-factual.jsonl --logits artifacts/m1/raw/m1b-logits-factual.pt \
  --geometry artifacts/m2/evidence/m2-geometry-30b.json \
  --output artifacts/m2/evidence/m2-gate.json 2>&1 | tee artifacts/m2/evidence/m2-analyze-output.txt
```

(Record the exact invocation in `m2-commands.ps1` as the PowerShell equivalent — this is the showcase-box shell.)

- [ ] **Step 3: Hash every artifact and input**

```bash
./.venv/Scripts/python.exe -c "import hashlib,glob; [print(hashlib.sha256(open(f,'rb').read()).hexdigest(), f) for f in sorted(glob.glob('artifacts/m2/evidence/*')+glob.glob('artifacts/m1/raw/m1b-trace-*.jsonl'))]" > artifacts/m2/evidence/m2-files.sha256
```

- [ ] **Step 4: Write `M2-REPORT.md`**

Structure it after `artifacts/m1/evidence/M1-REPORT.md`. Required sections, every number cited to `m2-gate.json` / `m2-analyze-output.txt`:
1. **What survives** — the raw-trace inventory: 5 int8-OLMoE traces (500 positions), 30B BF16 trace GONE (VPS destroyed), so the gate is projected.
2. **Measured (OLMoE, int8)** — reuse-distance histogram shape, union-growth sublinearity across windows {1…512}, co-activation summary, per-trace hit-rate(f).
3. **Projected (30B, Q4)** — the working-set-fraction method with formula and inputs (`avg_expert_bytes`, `requests_per_token=384`, `B_cold=2.62 GB/s`), the per-tier {4,8,16,32} GB table with `core_fits`, `f`, `miss_bytes_per_token`, `tok_s`; each row stamped `PROJECTED`.
4. **Gate verdict** — `projected_ceiling_tok_s`, the `band_16gb_tok_s` spread, `spreads_straddle_gate_line`, and `branch` ∈ {build_m3m4, escalate_m5m6, re_rent}, with the §7 threshold each was compared against.
5. **Threat to validity** — the two projection hops (int8→Q4, OLMoE→30B) and §11-Q1 UNRESOLVED, verbatim from `threat_to_validity`.
6. **Core-fit** — exact non-expert core (0.93 GiB, Task 1) vs each tier; note any tier dropped.

- [ ] **Step 5: Verify the report matches the JSON**

Run: `./.venv/Scripts/python.exe tests/test_moe_trace.py` (suite still green) and spot-check that every numeric claim in `M2-REPORT.md` appears in `m2-gate.json`. If the branch is `re_rent`, the report states the re-rent is the next action and the ported bash scripts (`artifacts/m1/evidence/m1c-rented-commands.sh`) are the vehicle — but M2 itself is done at the verdict.

- [ ] **Step 6: Commit**

```bash
git add artifacts/m2/evidence/
git commit -m "docs(moe): report M2 projected tier gate from surviving OLMoE traces"
```

---

## Self-Review

**1. Spec coverage (§7 M2 outputs):** cache-hit curves → Task 4; union growth vs window → Task 6 (`union_growth`); reuse distance → Task 2; co-activation → Task 3; predicted tok/s per tier → Task 5; numeric decision gate (≥8/build, >500MB/escalate, core-doesn't-fit/drop) → Task 5 (`gate_branch`, tuned to the confirmed 10/5 tok/s band) + Task 7 verdict. Exact core for the drop clause → Task 1. Reproducibility/honesty (§10) → Task 7 evidence bundle.

**2. Placeholder scan:** every code step carries real code; every run step carries an exact command and expected output. No TBD/TODO.

**3. Type consistency:** `hit_rate_curve` rows carry `fraction`/`miss_rate` consumed by `interp_miss_rate` and `analyze_traces` (checked). `project_tier` returns `ram_bytes`/`core_fits`/`tok_s`/`miss_bytes_per_token` consumed by `analyze_traces` (checked). `ModelMeta.nonexpert_bytes` (Task 1) → `geometry["nonexpert_bytes"]` (Task 6). `analyze_traces` trace tuple `(name, steps, logits_sha256)` matches the CLI's construction.

**Known deviation from the spec's literal gate wording:** §7 states "≥ ~8 tok/s" and "> ~500 MB/token". This plan implements the **confirmed refinement** from grilling: build ≥10, escalate ≤5, and re-rent the 30B BF16 trace in the (5,10) band or when inter-domain spread straddles a line — because every 30B number here is a two-hop projection, so a single point estimate at 8 cannot honestly decide the branch. The 8 tok/s ⇔ 327 MB/token line remains the reference midpoint in the report.



