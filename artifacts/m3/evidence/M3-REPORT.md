# M3 Expert-Contiguous Repack — Measured Results

Milestone 3 asks two independent questions about laying each expert's weights out
contiguously on disk: (1) does the repack preserve inference **exactly**, and (2) does the
contiguous layout **read faster** than the stock scattered layout on the real device? Unlike
M2, nothing here is projected. Both questions are answered by **direct measurement** on this
box, and both gates are derived — not asserted — by `derive_m3_gate`. Every number below
names the field it came from (`m3-gate.json`, mirrored byte-for-byte into
`m3-bench-output.txt`); nothing was hand-tuned toward a verdict.

The gate is a clean PASS: `passed=true`, `throughput_verdict=separated_win`,
`branch=proceed_m4`, `output_exact.all_equal=true`.

---

## 1. Output-exact — the byte-identity gate (MEASURED)

Repack rewrote **6144 expert blobs** (48 layers × 128 experts) into one contiguous packed
file plus a `manifest.json` of offsets. The read-back path then compared every source
role-slice against its repacked copy:

- `output_exact.n_slices = 18432` — every expert carries three role-slices (gate / up / down),
  so 6144 × 3 = 18432 slices were checked.
- `output_exact.all_equal = true` — **all 18432 slices matched byte-for-byte.**
- `output_exact.rollup_sha256 = 2007138502a7631ba4876b3bdf7a3f71bdde0d19881f2efcf4517c3be081e4cb`
  — a single content-addressed witness folded over all source/read-back pairs.

Because the repacked bytes are provably identical to the stock bytes, **inference is
unchanged without ever running the model.** The layout moved; the content did not. This gate
needs no routing, no logits, and no forward pass — it is a pure identity proof over the
weight bytes.

## 2. Throughput — the band-separation gate (MEASURED)

**The fair comparison.** Per token the repacked arm issues **8 aligned single-blob reads**
(one contiguous read per resident expert) at real scattered file offsets; the stock arm
issues **24 scattered unaligned role-slice reads** (8 experts × 3 roles) at their original
positions. Same experts, same `NO_BUFFERING` unbuffered primitive, same QD proxy on both
arms; the stock arm's sector-rounding is the honest cost of its misalignment, not a handicap
added to it.

**Both arms, `n=5` runs, `seed=1234`, `experts_per_layer=8`, `working_set_bytes=1097072640`
(1.097 GB per token):**

| arm | min | median | max |
|---|---|---|---|
| repacked_bps | 1,121,039,912.8 (1.121 GB/s) | 1,134,797,497.4 (1.135 GB/s) | 1,157,442,161.0 (1.157 GB/s) |
| stock_bps    | 1,018,215,136.0 (1.018 GB/s) | 1,021,317,250.7 (1.021 GB/s) | 1,027,839,046.4 (1.028 GB/s) |

The verdict is **`separated_win`** because `repacked_bps.min` (1.121 GB/s) **≥**
`stock_bps.max` (1.028 GB/s): the two bands do not overlap at all. This is a parameter-free
criterion — full band separation, no threshold to tune, no distributional assumption.

**The two ratios, reported honestly.** `qd16_ratio = 1.1111` (≈1.11×, the QD-proxy-16 arm on
which the gate is decided) and `qd1_ratio = 1.2044` (≈1.20×, single-threaded). The
`throughput.note` states plainly that QD here is a **thread-count proxy applied identically
to both arms** (`qd_proxy_threads=16`), not a measured device queue depth. Honest
observation: the QD-proxy-16 ratio (~1.11×) is **smaller** than the QD=1 ratio (~1.20×) — the
layout advantage did **not** amplify under the thread-count proxy; if anything it compressed.
We do not spin a "concurrency amplifies the win" story, because the evidence says the
opposite. The gate passes on **band separation at QD-proxy-16 regardless** of which ratio is
larger.

**NVMe bandwidth, re-measured in-session.** `nvme_seq_bps = 1238057388.7` (~1.24 GB/s) is
the NVMe bandwidth **re-measured on this box this session**. Per honesty rule #2 it
**REPLACES** the spec §3 constant (2.62 GB/s) — the live number governs, not the stored one.
Honestly: this re-measured figure is well **below** §3's 2.62 GB/s, but that is a
**methodology difference, not a contradiction to hide.** §3's constant is a buffered
sequential-stream number; this M3 figure is a **cold, unbuffered, single-queue (QD-1)** read
through the same `NO_BUFFERING` primitive the arms use. Unbuffered single-queue reads are
expected to land well under a warm sequential stream on the same device; the two measure
different things.

## 3. §11-Q1 stays open — and M3 does not depend on it

Spec §11 open-Q1 (routing precision-stability, int8/BF16 → Q4) remains **UNRESOLVED**, and
**neither M3 gate leans on it.** The byte-identity gate needs no routing at all — it is an
identity proof over weight bytes. The throughput gate uses a **seeded synthetic scatter**
(`seed=1234`) to choose which experts are resident, so it too is independent of the real
routing distribution. A favourable M3 does not retire §11-Q1, and it does not need to.

## 4. No hand-tuning

Carrying the M2 honesty line: **nothing here was hand-tuned toward a verdict.** `passed` and
`branch` are computed by `derive_m3_gate` from the measured `output_exact` and `throughput`
fields; the report only reads them back. `separated_win` is a mechanical band-comparison, not
a chosen threshold.

## 5. Threats to validity

Two measurement caveats are disclosed here rather than swept under the PASS. Neither weakens
either gate (`passed=true`, `separated_win`, byte-identity `all_equal=true`), but both bound
what the throughput number does and does not prove.

- **Physical file placement.** The throughput comparison is file-vs-file, exactly as spec §7's
  gate is worded: the repacked arm reads a freshly-written ~17.5 GB packed file, the stock arm
  reads the older ~18.5 GB source GGUF. Beyond the intended within-file layout effect, the two
  files may differ in on-disk fragmentation, which could bias the ratio. The modest, consistent
  margin (qd16 1.11× / qd1 1.20×, with full band separation) is more consistent with a layout
  effect than with a gross placement artifact — but the confound is inherent to the M3 premise
  (repack *produces* a new file) and is disclosed rather than controlled.
- **Stock-offset validation is fixture-proven.** The `t.data_offset + e*per == t.data[e]`
  equivalence used by the stock arm is unit-proven on the small Q8_0 fixture (n_expert=4), not
  on the real Q4_K/Q6_K 128-expert file. This weakens neither gate: byte-identity never uses
  the stock offsets (it sources via `expert_slice`/`t.data[e]`), and the stock arm exists only
  for timing — a hypothetical offset error would still read the same byte volume at scattered
  positions, leaving the layout verdict intact.

---

## Reproducibility

Exact commands in `m3-commands.ps1`; the repack console in `m3-repack-output.txt`; the
bench-read console in `m3-bench-output.txt` (byte-identical to `m3-gate.json`, emitted via the
same `_emit` path — their SHA-256s match in `m3-files.sha256`, exactly as M2's gate/analyze
pair did). SHA-256 of every evidence artifact plus the runtime `manifest.json` is in
`m3-files.sha256`. The 17.5 GB packed file and its manifest stay gitignored under
`models/m3/`; only this compact evidence is committed under `artifacts/m3/evidence/`.

**Caveat (evidence fidelity).** The preflight's loud **ABORT** path (required-vs-available
disk check) is implemented and fired clean — the disk check ran and passed at runtime — but
its **success path did not echo the numbers to the transcript**, a minor evidence-fidelity
gap versus M1c, which did print them. The check itself is not in question; only its console
trace is thinner than M1c's.
