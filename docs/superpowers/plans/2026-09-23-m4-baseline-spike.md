# M4 Baseline Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure stock llama.cpp's warm-steady-state and cold decode throughput on this box and decide — against a ceiling computed from this spike's own measurements, gated primarily on warm steady state — which memory regime the box is in and whether the M4 C++ expert-cache patch is worth building.

**Architecture:** A measurement spike, not a feature. Build stock llama.cpp from source (the same base we would later patch), then measure **warm-steady-state** and **cold** decode tok/s *and residual disk-bytes-per-token* on the resident Qwen3-30B-A3B-Q4_K_M GGUF under `--n-cpu-moe`, and compute the I/O-bound ceiling (independent scatter bandwidth ÷ measured miss-bytes/token). Gate **primarily on warm steady state**, with cold as the worst-case bound: if warm still misses substantially the model doesn't fit → a real streaming baseline exists → patch justified; if warm barely misses the OS effectively caches the whole model → the thesis is moot *on this box* and that IS the finding (don't greenlight a patch whose benefit only appears in a cold-start regime users rarely hit). Go → round-4 patch design. No-go → escalate an honest committed negative and close the milestone.

**Tech Stack:** Windows 11, PowerShell + Git-Bash; CMake + Ninja + MSVC Build Tools (or clang/LLVM); upstream llama.cpp (CPU/AVX2, no CUDA); `./.venv/Scripts/python.exe` for aggregation (no pytest — `__main__`/assert self-checks).

**Spec:** `docs/superpowers/specs/2026-09-11-moe-exact-runtime-design.md` (§7 M4 row, §3 physical limits, §10 honesty rules). Spine decision: `docs/adr/0001-m4-forward-pass-patch-llamacpp-gated-on-spike.md`. Tracker: issue #2.

## Global Constraints

- Python only via `./.venv/Scripts/python.exe` — never bare `python`/`hf`. No pytest; self-checks are `assert`-based `__main__` blocks.
- Build llama.cpp from upstream commit `1a064ab` (the build LM Studio's 2.27.0 bundle reports), CPU/AVX2 backend, **no CUDA**. Clone **outside** the repo at `C:\Rafi\Projects\llama.cpp` (round 4 forks the same clone); nothing built lands in `llm-lab`.
- Disk: build budget ≤7 GiB. Tripwire: if free space on `C:` drops **below 5 GiB** at any point, reclaim `models/m3/qwen3-repack/` (regenerable via `moe repack`) — the spike does not need the repack, only the GGUF `models/m1/qwen3-gguf/Qwen3-30B-A3B-Q4_K_M.gguf`.
- Honesty (spec §10): every number is written next to the exact command that produced it, into `artifacts/m4/evidence/`. Baselines measured **this machine/session**. `--n-cpu-moe` is ≈ a no-op on a GPU-less box — every report states this; the real baseline is OS-paged mmap.
- Report **warm-steady-state and cold separately** (cold = standby list evicted before the run; warm-steady = after a warmup pass fills the page cache). For the stock baseline, queue depth is the OS's, not ours — report it as such, do not claim QD≥16.
- Verdict (gated **primarily on warm steady state**, cold as worst-case bound): **go** = warm-steady is a genuine streaming regime (residual miss-bytes/token is a large fraction of the ~1.0 GB/token uncached expert traffic — the OS is *not* caching the whole model) **and** warm-steady tok/s sits materially below the warm ceiling (beyond noise). **no-go** = warm barely misses (model fits cache → thesis moot on this box) OR warm already saturates the drive → `passed:false`, `branch:escalate`, committed evidence, close issue #2.
- **Regime boundaries are anchored, not magic.** The two soft cutoffs are the only judgment in the gate, so: (a) commit the **raw** warm miss-bytes/token and warm tok/s + ceiling **with min/max spreads** (in `m4-gate.json` and per-run in `baseline-runs.jsonl`) so a near-boundary result is a visible judgment call; (b) the streaming cutoff `miss_frac=0.5` ≈ 500 MB/token is stated as *reasoning* (mid the physical band between the ~2.6 tok/s floor and ~10 tok/s interactive line), not a bright line; (c) "materially below" is **measurement-spread separation** (fastest warm run below the most-pessimistic ceiling, M3-style), never a fresh eyeballed margin.
- **Warm residual miss traffic is a physical measurement** — the observed `\PhysicalDisk` read counter (bytes the drive actually served), never inferred from cache-size arithmetic. The whole regime call hinges on this number.

## Review Focus

- **Disk exhaustion mid-build** — the 5 GiB tripwire must fire *before* a build write fails; Task 1 checks free space after toolchain install and again before configuring, and reclaims the repack if under threshold.
- **Standby-list eviction silently failing** (no admin / tool absent) → "cold" runs are actually warm → a false no-go. Task 3 verifies eviction moved free memory before trusting any cold number.
- **`--n-cpu-moe` misread as "we exercised the offload path"** on a CPU-only box → Task 3's report states the flag is ≈ inert here and names what was actually measured (mmap paging).
- **Unrelated I/O inflating miss-bytes/token** → false headroom. Task 3 quiesces other processes and measures the disk counter only across the model's run window.
- **Circular ceiling** — deriving scatter bandwidth from the same decode run as miss-bytes makes ceiling ≡ measured tok/s and headroom always zero. The bandwidth input is an **independent** scattered-read measurement (M3's committed microbenchmark, same host), never the decode run's own throughput. Task 2's self-check pins this separation.

---

### Task 1: Build stock llama.cpp (CPU/AVX2, from `1a064ab`)

**Files:**
- Create (outside repo): `C:\Rafi\Projects\llama.cpp\` (git clone + `build\` tree)
- Create: `artifacts/m4/evidence/build-log.txt` (configure + compile output, tail)
- Create: `artifacts/m4/evidence/build-env.json` (toolchain versions, commit, free-disk before/after)

**Interfaces:**
- Produces: `C:\Rafi\Projects\llama.cpp\build\bin\llama-bench.exe` and `llama-cli.exe`, built at commit `1a064ab`, with `--n-cpu-moe` in `llama-bench --help`. Task 3 consumes these binaries and the GGUF path.

- [ ] **Step 1: Record free disk + install the minimal toolchain**

Run (PowerShell): `Get-PSDrive C | Select-Object Free` → record bytes. Install ONLY what is missing: MSVC C++ Build Tools (or `winget install LLVM.LLVM` for the lighter clang path), CMake, Ninja, Windows SDK. Do **not** install full Visual Studio. Capture `cmake --version`, `ninja --version`, and `cl`/`clang --version` into `build-env.json`.

- [ ] **Step 2: Re-measure free disk and apply the tripwire**

Run: `Get-PSDrive C | Select-Object Free`. If free `< 5 GiB`: `Remove-Item -Recurse -Force models/m3/qwen3-repack` (regenerable via `moe repack`; not needed by this spike) and record the reclaim in `build-env.json`. Record before/after bytes.

- [ ] **Step 3: Clone llama.cpp at the pinned commit**

Run: `git -C C:\Rafi\Projects clone https://github.com/ggml-org/llama.cpp` then `git -C C:\Rafi\Projects\llama.cpp checkout 1a064ab`. If that exact commit is not fetchable, check out the nearest tag whose `llama-bench --help` still lists `--n-cpu-moe` and record the substituted ref in `build-env.json`.

- [ ] **Step 4: Configure CPU-only, build the two binaries**

Run: `cmake -S C:\Rafi\Projects\llama.cpp -B C:\Rafi\Projects\llama.cpp\build -G Ninja -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DLLAMA_CURL=OFF` then `cmake --build C:\Rafi\Projects\llama.cpp\build --target llama-bench llama-cli -j`. Tee output to `build-log.txt` (tail only). Expected: both binaries produced, exit 0.

- [ ] **Step 5: Verify build identity, flag, and a 1-token smoke decode**

Run: `C:\Rafi\Projects\llama.cpp\build\bin\llama-bench.exe --help | Select-String -Pattern "n-cpu-moe"` (expected: match) and `... llama-cli.exe --version` (expected: build near `1a064ab`). Smoke: `llama-cli -m models/m1/qwen3-gguf/Qwen3-30B-A3B-Q4_K_M.gguf -p "hi" -n 1 -no-cnv` → expected: exits 0, emits one token. Record all three into `build-env.json`.

- [ ] **Step 6: Commit**

---

### Task 2: Spike measurement harness + self-check

**Files:**
- Create: `scripts/m4-spike/aggregate.py` (ceiling + verdict + gate-writer; `--selfcheck`)
- Create: `scripts/m4-spike/run-baseline.ps1` (evict standby list, run llama-bench, capture disk-bytes)

**Interfaces:**
- Produces: `compute_ceiling(scatter_bps, miss_bytes_per_token) -> float` (returns `inf` if miss ≤ 0); `verdict(warm_tok_s_max, warm_ceiling_min_tok_s, warm_miss_bytes_per_token_median, per_token_expert_bytes, miss_frac=0.5) -> dict` (keys `verdict`, `branch`, `streaming_regime`, `headroom`, `reason`); `build_gate(runs, scatter_bps, scatter_provenance, per_token_expert_bytes, miss_frac=0.5) -> dict` — computes medians **and min/max spreads**, a headline `warm_ceiling` (from median miss) plus a conservative `warm_ceiling_min` (from **max** miss), and gates via `verdict`. `runs` carries `cold_tok_s`, `cold_miss_bytes_per_token`, `warm_tok_s`, `warm_miss_bytes_per_token`. **Headroom = spread separation** (fastest warm run below the most-pessimistic ceiling), never a fixed margin. Task 3 writes the result to `artifacts/m4/evidence/m4-gate.json`.
- Consumes: `statistics.median` (stdlib) for run aggregation — no custom median.

- [ ] **Step 1: Write the failing self-check**

```python
# scripts/m4-spike/aggregate.py  (self-check portion)
PER_TOKEN_EXPERT_BYTES = 8 * 48 * 2_654_208  # top-8 × 48 layers × 2.65 MB/expert ≈ 1.02e9 (spec §3)
# miss_frac=0.5 → ≈500 MB/token threshold: squarely in the physical band between the ~2.6 tok/s
# drive floor and the ~10 tok/s interactive line, so below it the OS is clearly caching most of the
# model. It is a regime boundary, not a bright line — raw numbers + spreads are committed so a
# near-boundary result reads as a visible judgment call, never a hidden flip.

def _selfcheck():
    assert abs(compute_ceiling(1.021e9, 1.0e8) - 10.21) < 1e-6
    assert compute_ceiling(1.021e9, 0) == float("inf")
    # streaming (8e8 > 500MB) AND separated (fastest warm 0.85 < pessimistic ceiling 1.10) -> go
    assert verdict(0.85, 1.10, 8.0e8, 1.0e9)["verdict"] == "go"
    # streaming but fastest warm overlaps the ceiling -> no headroom
    r = verdict(1.15, 1.10, 8.0e8, 1.0e9)
    assert r["verdict"] == "no_go" and r["reason"] == "stock_saturates_no_headroom"
    # warm barely misses (< 500MB/token) -> model fits cache -> moot
    r = verdict(0.85, 1.10, 1.0e8, 1.0e9)
    assert r["verdict"] == "no_go" and r["reason"] == "model_fits_cache_moot"
    g = build_gate({"cold_tok_s": [0.5], "cold_miss_bytes_per_token": [9.8e8],
                    "warm_tok_s": [0.80, 0.82, 0.78], "warm_miss_bytes_per_token": [7.9e8, 8.0e8, 8.1e8]},
                   1.021e9, "M3 microbench (same host)", 1.0e9)
    assert g["spike_verdict"] == "go" and g["branch"] == "proceed_patch"
    assert g["warm_tok_s_max"] == 0.82 and g["warm_miss_bytes_per_token_max"] == 8.1e8  # spreads committed
    g2 = build_gate({"cold_tok_s": [0.5], "cold_miss_bytes_per_token": [9.8e8],
                     "warm_tok_s": [6.0], "warm_miss_bytes_per_token": [1.0e8]},
                    1.021e9, "M3 microbench (same host)", 1.0e9)
    assert g2["passed"] is False and g2["reason"] == "model_fits_cache_moot"
    print("aggregate self-check OK")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `./.venv/Scripts/python.exe scripts/m4-spike/aggregate.py --selfcheck`
Expected: FAIL — `NameError: compute_ceiling is not defined`.

- [ ] **Step 3: Implement the minimal harness**

```python
import json, statistics, sys

def compute_ceiling(scatter_bps, miss_bytes_per_token):
    return scatter_bps / miss_bytes_per_token if miss_bytes_per_token > 0 else float("inf")

def verdict(warm_tok_s_max, warm_ceiling_min_tok_s, warm_miss_bytes_per_token_median,
            per_token_expert_bytes, miss_frac=0.5):
    streaming_regime = warm_miss_bytes_per_token_median > miss_frac * per_token_expert_bytes
    headroom = warm_tok_s_max < warm_ceiling_min_tok_s   # spread separation, not a fixed margin
    go = streaming_regime and headroom
    if go:
        reason = "go"
    elif not streaming_regime:
        reason = "model_fits_cache_moot"
    else:
        reason = "stock_saturates_no_headroom"
    return {"verdict": "go" if go else "no_go",
            "branch": "proceed_patch" if go else "escalate",
            "streaming_regime": streaming_regime, "headroom": headroom, "reason": reason}

def _stats(xs):
    return {"median": statistics.median(xs), "min": min(xs), "max": max(xs)}

def build_gate(runs, scatter_bps, scatter_provenance, per_token_expert_bytes, miss_frac=0.5):
    w_tok, w_mbpt = _stats(runs["warm_tok_s"]), _stats(runs["warm_miss_bytes_per_token"])
    c_tok, c_mbpt = _stats(runs["cold_tok_s"]), _stats(runs["cold_miss_bytes_per_token"])
    warm_ceiling = compute_ceiling(scatter_bps, w_mbpt["median"])       # headline
    warm_ceiling_min = compute_ceiling(scatter_bps, w_mbpt["max"])      # conservative (most misses)
    v = verdict(w_tok["max"], warm_ceiling_min, w_mbpt["median"], per_token_expert_bytes, miss_frac)
    gate = {"phase": "baseline_spike", "gated_on": "warm_steady_state",
            "ncpumoe_note": "flag ~inert on CPU-only box; baseline is OS-paged mmap",
            "miss_bytes_source": "physical PhysicalDisk read counter (bytes the drive served), not cache-size arithmetic",
            "miss_frac": miss_frac, "miss_frac_rationale": "0.5*per_token ≈ 500 MB/token, mid physical band (~2.6 floor..~10 interactive tok/s)",
            "warm_tok_s_median": w_tok["median"], "warm_tok_s_min": w_tok["min"], "warm_tok_s_max": w_tok["max"],
            "warm_miss_bytes_per_token_median": w_mbpt["median"], "warm_miss_bytes_per_token_min": w_mbpt["min"], "warm_miss_bytes_per_token_max": w_mbpt["max"],
            "warm_ceiling_tok_s": warm_ceiling, "warm_ceiling_min_tok_s": warm_ceiling_min,
            "cold_tok_s_median": c_tok["median"], "cold_tok_s_min": c_tok["min"], "cold_tok_s_max": c_tok["max"],
            "cold_miss_bytes_per_token_median": c_mbpt["median"], "cold_ceiling_tok_s": compute_ceiling(scatter_bps, c_mbpt["median"]),
            "per_token_expert_bytes": per_token_expert_bytes, "scatter_bps": scatter_bps, "scatter_provenance": scatter_provenance,
            "spike_verdict": v["verdict"], "branch": v["branch"],
            "streaming_regime": v["streaming_regime"], "headroom": v["headroom"], "reason": v["reason"]}
    if v["verdict"] == "no_go":
        gate["passed"] = False
    return gate

if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
```

- [ ] **Step 4: Run it to verify it passes**

Run: `./.venv/Scripts/python.exe scripts/m4-spike/aggregate.py --selfcheck`
Expected: PASS — prints `aggregate self-check OK`.

- [ ] **Step 5: Write the cold-run wrapper (no test — exercised for real in Task 3)**

`scripts/m4-spike/run-baseline.ps1`: params `-Bin`, `-Model`, `-NGen`, `-Threads`, `-Cold` (switch), `-State` (string `cold`/`warm`/`warmup`), `-OutJson`. When `-Cold`: evict the standby list first — primary `RAMMap64.exe -Et` or `EmptyStandbyList.exe standbylist` if present on PATH; fallback: allocate/read a >RAM scratch file to force eviction — then **assert** free memory rose (record `pre`/`post` standby bytes via `Get-Counter "\Memory\Standby Cache Standby List Bytes"`; if it did not drop, write `"cold_verified": false`). Sample the **physical** `\PhysicalDisk(_Total)\Disk Read Bytes/sec` counter (real bytes the drive served — an ETW/perf measurement, **NOT** inferred from cache-size arithmetic) across the run window on **every** run (cold and warm) so warm **residual** miss traffic is the observed physical read; sum × interval = `disk_bytes`. Invoke `llama-bench -m $Model -ngl 0 --n-cpu-moe 999 -t $Threads -n $NGen -p 0 -r 1 -o json` (tg-only). Emit one JSON line: `{state, tok_s, disk_bytes, tokens, cold_verified}` (`cold_verified` omitted/null for warm).

- [ ] **Step 6: Commit**

```bash
git add scripts/m4-spike/aggregate.py scripts/m4-spike/run-baseline.ps1
git commit -m "feat(m4-spike): measurement harness (ceiling+verdict) with self-check"
```

---

### Task 3: Execute the spike on this box

**Files:**
- Create: `artifacts/m4/evidence/baseline-runs.jsonl` (raw per-run JSON lines)
- Create: `artifacts/m4/evidence/scatter-bandwidth.txt` (value + provenance)
- Create: `artifacts/m4/evidence/m4-gate.json` (the committed verdict)
- Create: `artifacts/m4/evidence/M4-SPIKE-REPORT.md` (commands + numbers + verdict)

**Interfaces:**
- Consumes: Task 1's `llama-bench.exe` + the GGUF; Task 2's `run-baseline.ps1` and `aggregate.build_gate`.

- [ ] **Step 1: Fix the run parameters**

Threads = physical cores: `(Get-CimInstance Win32_Processor).NumberOfCores` → record. `NGen = 128`. Quiesce other heavy processes so the disk counter reflects only the model. Record cores + NGen at the top of `M4-SPIKE-REPORT.md`.

- [ ] **Step 2: Five COLD runs (fresh eviction each), then warm up, then five WARM-STEADY runs**

```powershell
$bin="C:\Rafi\Projects\llama.cpp\build\bin\llama-bench.exe"
$m="models/m1/qwen3-gguf/Qwen3-30B-A3B-Q4_K_M.gguf"; $out="artifacts/m4/evidence/baseline-runs.jsonl"
1..5 | % { scripts/m4-spike/run-baseline.ps1 -Bin $bin -Model $m -NGen 128 -Threads <cores> -Cold -State cold -OutJson $out }
1..3 | % { scripts/m4-spike/run-baseline.ps1 -Bin $bin -Model $m -NGen 128 -Threads <cores> -State warmup -OutJson $out }  # fill page cache; discarded in aggregation
1..5 | % { scripts/m4-spike/run-baseline.ps1 -Bin $bin -Model $m -NGen 128 -Threads <cores> -State warm -OutJson $out }
```
Expected: 5 `cold` + 3 `warmup` + 5 `warm` JSON lines. **Abort and fix eviction if any `cold` line has `cold_verified:false`** — a warm "cold" run fabricates a false regime (Review Focus). The 3 warmup lines exist only to fill the cache and are dropped by `state` during aggregation.

- [ ] **Step 3: Independent scatter bandwidth (never the decode run's own throughput)**

Primary: reuse M3's committed same-host microbenchmark — `throughput.stock_bps.median` = `1021317250.7` from `artifacts/m3/evidence/m3-gate.json`. If that probe script is locatable under `scripts/`/`benchmarks/`, rerun it once for a same-session number and prefer it. Write the value + provenance string to `scatter-bandwidth.txt`.

- [ ] **Step 4: Aggregate → gate JSON**

```bash
./.venv/Scripts/python.exe scripts/m4-spike/aggregate.py --selfcheck   # re-confirm green on this box
```
Then a short driver (inline or `aggregate.py --emit`) parses `baseline-runs.jsonl`, **drops `state=="warmup"` lines**, and builds `{cold_tok_s, cold_miss_bytes_per_token, warm_tok_s, warm_miss_bytes_per_token}` (each `*_miss_bytes_per_token` = `disk_bytes/tokens` per run of that state), calls `build_gate(runs, <scatter_bps>, "<provenance>", PER_TOKEN_EXPERT_BYTES)`, and writes the returned dict to `artifacts/m4/evidence/m4-gate.json`.

- [ ] **Step 5: Write the report**

`M4-SPIKE-REPORT.md`: cores/NGen; the exact `llama-bench` command; **warm-steady medians (the primary gate) and cold medians (worst-case bound)**, each ± min/max; residual warm & cold miss-bytes/token (stating these are **physical `\PhysicalDisk` counter** reads, not inferred); scatter bandwidth + provenance; warm headline & conservative (`warm_ceiling_min`) ceilings; the `miss_frac`≈500 MB/token band-anchor rationale; the regime determination (streaming vs model-fits-cache) and which verdict `reason` fired (`go` / `model_fits_cache_moot` / `stock_saturates_no_headroom`); the `--n-cpu-moe` ≈-inert-on-CPU note; and `cold_verified` status. Every number cites its command. **If any value sits within its own spread of a boundary, say so explicitly and call the verdict a judgment call, not a clean flip.**

- [ ] **Step 6: Commit evidence**

```bash
git add artifacts/m4/evidence/
git commit -m "test(m4-spike): stock cold baseline + measured ceiling + go/no-go gate"
```

- [ ] **Step 7: Update issue #2**

Run: `gh issue comment 2 --body "<verdict + link to m4-gate.json + report>"`. On **no_go**: `gh issue close 2 --comment "M4 premise did not survive contact — see artifacts/m4/evidence/m4-gate.json (passed:false, branch:escalate)."`. On **go**: comment "proceed to round-4 patch design against the built 2.27.0 source" and leave open.

---

## Self-Review

**1. Spec coverage.** Spike scope = spec §7 M4 gate #2 baseline only ("beats llama.cpp `--n-cpu-moe` on the same box"), reframed by ADR-0001 as a *pre-build* measurement, and gated **primarily on warm steady state** (cold = worst-case bound) so the box's actual memory regime — not a cold-start users rarely hit — decides the go/no-go. Gate #1 (output identity across cache sizes) and the loader/cache/async design are deliberately **out** of this spike — they are round-4+, behind the go/no-go — so their absence here is by design, not a gap. §3 physical-limit inputs (`PER_TOKEN_EXPERT_BYTES`, scatter bandwidth) are consumed. §10 honesty (committed numbers, same-session baseline, `--n-cpu-moe` note, no unmeasured constants — the ceiling is computed, not carried) is enforced per task.

**2. Placeholder scan.** `<cores>` and `<scatter_bps>`/`<provenance>` are runtime-probed values with the exact command that yields each named in-step (Task 3 Steps 1, 3) — defined procedures, not TBDs. All code steps carry runnable code.

**3. Type consistency.** `verdict`/`build_gate`/`compute_ceiling`/`_stats` signatures match between Task 2's Interfaces, its implementation, its self-check, and Task 3 Step 4's call. `verdict(warm_tok_s_max, warm_ceiling_min_tok_s, warm_miss_bytes_per_token_median, per_token_expert_bytes, miss_frac=0.5)` — the fixed `noise_margin` is gone; headroom is spread separation. `PER_TOKEN_EXPERT_BYTES` is defined once in `aggregate.py` and reused. The `runs` dict shape — `cold_tok_s`, `cold_miss_bytes_per_token`, `warm_tok_s`, `warm_miss_bytes_per_token` — is identical in the self-check, `build_gate`, and Task 3's aggregation driver. The earlier cold-primary / working_set-vs-RAM formulation was removed everywhere.

**4. Review Focus.** Disk tripwire → Task 1 Steps 2/4. Eviction-verified-cold → Task 2 Step 5 (`cold_verified`) + Task 3 Step 2 (abort on false). Cold-only-benefit trap → warm-primary gate + `reason=model_fits_cache_moot` closes the milestone honestly when the OS caches the model. Hidden-boundary-flip → raw warm miss/tok/ceiling committed **with spreads**, spread-separation headroom, band-anchored `miss_frac` (Global Constraints); near-boundary results flagged as judgment calls in the report. Inferred-miss-traffic → **physical `\PhysicalDisk` counter only**, never cache-size arithmetic (Task 2 Step 5). `--n-cpu-moe` inertness → `ncpumoe_note` in the gate + report. Unrelated-I/O isolation → Task 3 Step 1 quiesce. Circular-ceiling → Task 2 self-check separates bandwidth (independent) from the run's tok/s + Task 3 Step 3 forbids using the decode run's throughput.

---

## Execution Handoff

Execution method already supplied by the user: **subagent-driven-development**.

