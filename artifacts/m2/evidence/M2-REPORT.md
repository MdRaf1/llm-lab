# M2 Tier Gate — Projected Results

Milestone 2 answers one question: on a 16 GB box, does Qwen3-30B-A3B-Q4_K_M route
densely enough to be worth building the low-memory expert-cache design (M3/M4), or must
we escalate / re-rent first? The 30B BF16 routing trace is **gone** (the rented VPS was
destroyed; only compact summaries survive), so the gate is a **PROJECTION** from the
surviving int8-OLMoE routing traces onto the measured 30B Q4 geometry. Every number below
names the file it came from (`m2-gate.json` / `m2-analyze-output.txt`); nothing here was
hand-tuned toward a verdict.

Two labelled projection hops separate the evidence from the target and are never asserted
away: **int8 → Q4** (quantization) and **OLMoE-1B-7B → 30B-A3B** (model). See §5.

---

## 1. What survives

- **5 int8-OLMoE routing traces**, 100 generated positions each = **500 positions**, across
  prose / code / factual domains (`m1b-trace-{prose-a,prose-b,code-a,code-b,factual}.jsonl`),
  each paired 1:1 with its `m1b-logits-*.pt`. Geometry of the trace source: 16 layers ×
  64 experts, top-8.
- **30B BF16 raw trace: GONE.** The Vultr rental that produced it was destroyed; only the
  M1c compact summaries remain. There is therefore **no measured 30B routing record** to
  gate on — the gate is projected, and every 30B number carries the `PROJECTED` label.
- Measured 30B Q4 geometry survives on the local box and is re-measured here (§ below).

## 2. Measured (OLMoE, int8)

All from `m2-gate.json` `per_trace[*]` and `mean_curve` — arithmetic over the recorded
traces + content-addressed logits digests, no model re-execution.

**Union growth is sublinear in the window** (prose-a, `per_trace[0].union_growth`; the
other four are the same shape). Distinct experts touched over a sliding window of W tokens:

| window W | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| mean distinct experts | 128 | 211 | 337 | 493 | 635 | 759 | 846 |

Doubling W never doubles the union (2→211 not 256; 4→337 not 512; 64→846, not thousands):
the working set saturates, which is the whole premise of an expert cache. (Windows ≥ 128
report 0 — a 100-position trace has no complete window that long; benign.)

**Cache-hit / miss curve vs working-set fraction** (`mean_curve`, mean over 5 traces):

| fraction f | 0.02 | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|---|---|---|---|
| miss_rate | 1.00 | 1.00 | 1.00 | 0.585 | 0.530 | 0.390 | 0.204 | 0.074 | 0.067 |

Miss rate falls steeply once ~15% of experts are resident and reaches ~7% at full
residency — locality is real, not flat.

**Reuse distance** is tightly clustered (prose-a `reuse_distance`: `cold=909`, histogram
peaked at distance 127) — recently-used experts recur soon, the second premise of caching.

**Per-trace projected 16 GB tok/s** (`per_trace[*].tok_s_16gb`, PROJECTED — see §3/§5):

| trace | prose-a | prose-b | code-a | code-b | factual |
|---|---|---|---|---|---|
| tok/s @16 GB | 22.90 | 23.34 | 35.11 | 24.69 | 24.74 |

## 3. Projected (30B, Q4) — PROJECTED

Method (`projection_scheme = "normalized-working-set-fraction"`): each trace's hit-rate(f)
curve is normalized to its **own** working set (OLMoE, 1024 expert slots = 16×64) and
re-projected onto the **30B target total** (48×128 = 6144 slots). Tier tok/s is then:

```
tok_s = 1 / ( miss_bytes_per_token / B_cold )
miss_bytes_per_token = requests_per_token * avg_expert_bytes * miss_rate(f_tier)
```

Inputs (stored fields cited to `m2-gate.json`; PROJECTED where marked):
- `avg_expert_bytes = 2856960.0` (2.857 MB; = `total_expert_bytes 17553162240` / (48×128)) — stored.
- `b_cold_bytes_s = 2620000000` (2.62 GB/s cold SSD stream) — stored.
- `reserve_bytes = 2500000000` (2.5 GB non-expert headroom) — stored.
- exact non-expert core `nonexpert_bytes = 997554176` (0.93 GiB, Task 1) — stored (`geometry`).
- `requests_per_token = 384` (48 layers × 8 experts used) — **derived** geometry product, not
  a stored field; that this is the multiplier used is confirmed by reconstructing the 16 GB
  tier exactly: `1 / (102674053 / 2620000000) = 25.52 tok/s`, matching `tiers[2].tok_s`.

**Per-tier table** (`m2-gate.json.tiers`), every row **PROJECTED**:

| tier RAM | core_fits | fraction f | miss_bytes/token | tok/s |
|---|---|---|---|---|
| 4 GB  | true | 0.0285 | 1,097,072,640 | 2.39  |
| 8 GB  | true | 0.2563 | 494,791,797   | 5.30  |
| 16 GB | true | 0.7122 | 102,674,053   | 25.52 |
| 32 GB | true | 1.0000 | 73,452,442    | 35.67 |

At 16 GB the projected working-set fraction is 0.712 → miss_rate low → **25.52 tok/s**.

## 4. Gate verdict

From `m2-gate.json`:

- `projected_ceiling_tok_s = 25.52` (the 16 GB tier — the target box), **PROJECTED**.
- `band_16gb_tok_s = [22.90, 35.11]` — the inter-domain spread of the 5 traces at 16 GB.
- `spreads_straddle_gate_line = false`.
- **`branch = build_m3m4`.**

§7 thresholds applied (confirmed refinement: build ≥ 10 tok/s, escalate ≤ 5, re-rent in the
(5,10) band or when the inter-domain spread straddles a gate line; 8 tok/s ⇔ 327 MB/token is
the reference midpoint). The entire 16 GB band **[22.90, 35.11]** sits above the build line
(both endpoints ≥ 10), and `spreads_straddle_gate_line = false` confirms no domain crosses
it → **build_m3m4**. This is the pipeline's own verdict, reported as emitted; it is not
`re_rent` and not `escalate_m5m6`.

Because the verdict is `build_m3m4`, the next milestone builds the expert-cache design
(M3/M4) directly. No 30B BF16 re-rent is required by the gate. **M2 is DONE at this verdict.**

## 5. Threat to validity

Verbatim from `m2-gate.json.threat_to_validity`:

> All 30B numbers are PROJECTED (int8-OLMoE -> Q4-30B, two hops); valid for the Q4 runtime
> only if routing is precision-stable -- §11-Q1 UNRESOLVED.

The two hops are **int8 → Q4** (quantization changes weights, may change routing) and
**OLMoE-1B-7B → 30B-A3B** (a different, larger model). Whether Q4 routing matches the BF16
routing the projection is calibrated from is **spec §11 open-Q1, UNRESOLVED**. The 16 GB
tok/s is therefore a projection of *OLMoE int8 routing locality onto 30B Q4 geometry*, not a
measured 30B number; a favourable band does not retire §11-Q1.

## 6. Core-fit

Exact non-expert resident core (Task 1, `nonexpert_bytes`): **997,554,176 B = 0.93 GiB** —
the measured resident weight core, not the ~1.00 GB file-minus-experts upper bound. Against
each tier's expert budget (tier − `reserve_bytes` 2.5 GB), the 0.93 GiB core fits every tier:

| tier | core_fits |
|---|---|
| 4 GB | true |
| 8 GB | true |
| 16 GB | true |
| 32 GB | true |

**No tier was dropped for a core that does not fit.** Even the 4 GB tier (1.5 GB after
reserve) holds the 0.93 GiB core, leaving ~0.5 GB for the expert working set (f ≈ 0.028).

---

## Reproducibility

Exact commands in `m2-commands.ps1`; full analyze console in `m2-analyze-output.txt`;
SHA-256 of every evidence artifact + the 5 input traces in `m2-files.sha256`. Raw traces
and logits stay gitignored under `artifacts/m1/raw/`; only this compact evidence is
committed under `artifacts/m2/evidence/`.
