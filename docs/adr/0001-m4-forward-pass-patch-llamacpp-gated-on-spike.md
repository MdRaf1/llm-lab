# ADR-0001: M4 forward-pass — patch llama.cpp, gated on a measured baseline spike

## Status

Accepted (2026-09-23). Spine decision for Milestone 4. Settled independent of the spike's go/no-go outcome.

## Context

M4 is the tiered exact async expert loader (spec §7): a RAM/NVMe expert cache with real async I/O (IOCP, QD≥16) that demand-loads Q4 experts from the M3 repack on a miss, with the resident core read from the source GGUF. It must clear two gates to reach M5: (1) output identical at every cache size, and (2) beat the llama.cpp `--n-cpu-moe` baseline on this box.

The open decision was **how the forward pass executes** while demand-loading Q4 experts. Three candidates:

- **(a)** custom exact CPU forward pass reusing `src/llm_lab/core/kernels.py` / `layers.py`;
- **(b)** patch llama.cpp to back expert tensors with the tiered cache;
- **(c)** transformers/HF integration.

Ground-truth finding (this session): the repo's "kept-real" INT4 code implements a self-invented per-row **symmetric** 4-bit format (`scale = max|w|/7`, values −8..7), tested only to `cos_sim > 0.98`. It is **not** Q4_K_M (256-element super-blocks, 8×32 sub-blocks, 6-bit asymmetric sub-scales/mins) and cannot dequantize a real expert tensor. Path (a) therefore has no head start toward bit-exactness and would require reimplementing the entire Q4_K_M inference stack (super-block dequant + GQA attention + RMSNorm + RoPE + gating dispatch + exact KV cache + embed/LM-head). Path (c) is off the precision path (the repack is Q4_K_M, not BF16).

Physical reality (spec §3 + M3 measurements): scattered expert reads measure ~1.135 GB/s repacked vs ~1.021 GB/s stock at a QD16 proxy (`artifacts/m3/evidence/m3-gate.json`) — a thin **1.11×** margin, not M2's projected 2.62 GB/s sequential figure. On a no-GPU box, stock llama.cpp already OS-page-faults experts via mmap, so gate #2 is genuinely uncertain.

## Decision

1. **Forward pass = (b) patch llama.cpp.** Selection stays llama.cpp's; the cache only backs expert bytes (M3 proved byte-identity). Exactness becomes structural — the cache changes *where* bytes live, never which experts compute or the arithmetic — and gate #2 stays a same-kernel comparison.
2. **The C++ patch is gated on a measurement spike.** First build stock llama.cpp (~b2.27.0, CPU/AVX2) from source, then measure stock **warm-steady-state** and **cold** decode tok/s under `--n-cpu-moe` plus residual miss-bytes/token, and compute the I/O-bound ceiling from the spike's **own** measurements (independent scatter bandwidth ÷ measured miss-bytes/token). Gate **primarily on warm steady state**, with cold as the worst-case bound: build the patch only if warm-steady is a genuine streaming regime (residual misses are a large fraction of per-token expert traffic — the OS is *not* caching the whole model) **and** warm-steady tok/s sits materially below the warm ceiling. If warm barely misses (the box effectively caches the model → thesis moot) or already saturates the drive, STOP and escalate a committed negative finding. The patch is never written blind, and never greenlit on a cold-start benefit users rarely hit.
3. **Exactness claim (M1 phrasing).** Bit-identical to `Qwen3-30B-A3B-Q4_K_M` under greedy decode, fixed seed and threads, **on this host**. There is no fully-resident reference on a 16 GB box (16.3 GB experts + core + KV exceed RAM); stock llama.cpp itself pages via mmap. The reference is valid because token output is paging-invariant (same bytes, kernels, arithmetic regardless of what is resident).
4. **VRAM tier deferred — no hardware.** M4 targets RAM/NVMe on the 16 GB no-GPU box. VRAM is documented as deferred, not failed.
5. **Routing invariant (spec §4).** Router/gate weights are part of the always-resident core, never cached or evicted; top-8 selection runs before any cache lookup. Cache size can change latency, never selection or token IDs.

## Consequences

- Commits M4 to building llama.cpp from source (none on this box — only LM Studio's `llama-server`) and, on go, a C++ expert-loader patch (deferred since M1, spec §8.4).
- Gate #2 may fail honestly: if warm-steady shows the box effectively caches the model (thesis moot) or already saturates the drive, M4 closes as "premise didn't survive contact," with committed evidence at `artifacts/m4/evidence/m4-gate.json` (`passed:false`, `branch:escalate`).
- §11-Q1 (Q4-vs-BF16 routing) stays out of scope; M4 exactness is Q4-loader vs Q4 on-box.
- The patch seam (provisional: cache-backed ggml buffer + ensure-resident hook on the already-selected top-8) is finalized in round 4 against the built 2.27.0 source, not now.

## Evidence

- M3 measured bandwidth: `artifacts/m3/evidence/m3-gate.json` → `throughput.repacked_bps.median` (1.135 GB/s), `throughput.stock_bps.median` (1.021 GB/s), `throughput.qd16_ratio` (1.111).
- M2 projection (superseded for M4 planning): `artifacts/m2/evidence/m2-gate.json` → `b_cold_bytes_s` (2.62 GB/s), `projected_ceiling_tok_s` (25.52). See `artifacts/m2/evidence/m2-gate-NOTE.md`.
