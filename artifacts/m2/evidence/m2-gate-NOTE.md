# m2-gate.json — forward note

`m2-gate.json` is the faithful, **byte-immutable** record of what M2 projected with the information it had. Its `passed` verdict is derived from its fields; do not edit it. This sibling note is the forward breadcrumb.

## Superseded projection

M2 projected cold bandwidth `b_cold_bytes_s = 2.62e9` (2.62 GB/s, **sequential**) and a 16 GB `projected_ceiling_tok_s = 25.52`.

M3 then **measured** the real scattered-expert read pattern at a QD16 proxy (`artifacts/m3/evidence/m3-gate.json`):

- `throughput.repacked_bps.median` = 1.135 GB/s
- `throughput.stock_bps.median` = 1.021 GB/s
- `throughput.qd16_ratio` = 1.111

Scattered reads run ~2.3–2.6× slower than M2's sequential figure, so M2's projected tok/s ceiling is optimistic for the real access pattern. **M4 planning uses the measured M3 bandwidth, not M2's projection**, and M4's gate #2 replaces the projection entirely with a same-box measurement. Rationale recorded in `docs/adr/0001-m4-forward-pass-patch-llamacpp-gated-on-spike.md`.
