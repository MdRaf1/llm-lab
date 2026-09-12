# Session handoff — resume prompt

Paste the block below as the **first message** of a new Claude Code session in
`C:\Rafi\Projects\llm-lab`. It reconstructs the decisions, the constraints, and the
things that must not be re-litigated.

---

```
Read docs/superpowers/plans/2026-09-12-m1-moe-reference-pipeline.md and
docs/superpowers/specs/2026-09-11-moe-exact-runtime-design.md first. The plan is approved and
binding; the spec is the design it implements. Do not re-litigate either.

Then invoke superpowers:subagent-driven-development and execute the plan starting at Task 1.

ALREADY SETTLED (do not reopen)
- M0 truth-in-labeling is done; the sim_* modules are simulations and stay labelled as such.
- Dense 70B on 16 GB is physically excluded. Target is MoE. PageCC is deferred to M6/M7.
- We are not first at draft-guided expert prefetch (SP-MoE, MoE-SpeQ, Apple SpecMD).
- M1 phase order is normative: M1a tiny random OLMoE (no download) -> M1b OLMoE-1B-7B on BOTH
  numeric paths -> M1c Qwen3-30B-A3B. Task 7 must pass before anything Qwen is downloaded.
- A trace step's tok_id is the PROCESSED token whose hidden state selected those experts, not
  the next token emitted from its logits. Generated positions only.
- Exactness means token-ID equality WITHIN one numeric path. Never assert identity between
  HF int8 and GGUF Q4 - they are different checkpoints numerically.
- Cache independence is an observational replay over one immutable trace plus a
  content-addressed logits file, not a second model execution.
- Large raw traces/logits live in ignored artifacts/m1/raw/. Commit scripts, transcripts,
  compact summaries and SHA-256 manifests under artifacts/m1/evidence/.

SEQUENCING AND COST
- Tasks 1-6 are local, seconds to minutes, zero downloads. Finish them before touching models.
- Task 7 pulls ~3.9 GiB of GGUF plus the pinned OLMoE BF16 shards. Gate on the preflight's
  measured byte counts, never a rounded size.
- Task 8 requires artifacts/m1/evidence/m1b-gate.json with passed=true, and its BF16 trace
  needs a rented ~128 GB machine. It cannot run on this box.

REPORTING RULES (non-negotiable, this repo violated all five once)
1. No number anywhere unless its command and output are saved in the repo.
2. Baselines measured on the same machine/session/checkpoint. Never a literature constant.
3. Projections labeled PROJECTED, with the formula and inputs shown.
4. Output identity = token-ID equality, never .strip() text comparison.
5. "Lossless" always names checkpoint + quantization + decode mode + seed.

TOOLING
- Bare `python` is not on PATH. Use ./.venv/Scripts/python.exe. Restore deps with `uv sync`.
- There is NO pytest here. tests/test_moe_trace.py is a directly executable assert script:
  ./.venv/Scripts/python.exe tests/test_moe_trace.py
- bitsandbytes==0.50.2 and accelerate==1.15.0 are verified working on this Windows CPU box but
  are deliberately NOT installed yet; Task 4 Step 3 adds them.
- WebSearch is broken. Use the `tvly` CLI with --json --output and read results as UTF-8.
- Long-running subagents hit gateway 524 timeouts. Give each subagent exactly one task.
```

---

## Also worth knowing in the new session

- **Remote:** `https://github.com/MdRaf1/llm-lab` — **private**, branch `master`. Pushed
  with full history on purpose: `README.md`, this file, and the spec all cite `a248a03` as
  where `DEMO_SCRIPT.md` is recoverable, and the M0 commit is only meaningful as a diff
  against the version that made the false claims. Squashing later is still possible;
  un-squashing is not.
- `docs/research/raw/` holds the prior session's evidence: `novelty-research.md` (prior-art
  survey), `unweight.txt` and `backslash.txt` (full paper extractions), and the raw Tavily
  JSON results. Check there before re-running any search.
- Commit history:
  - `a248a03` — original. Contains the fabricated claims and the deleted `DEMO_SCRIPT.md`.
  - `58d1171` — design spec + first handoff.
  - `690cdf7` — M0 truth-in-labeling.
  - `b47c59b` — fixed M1 phasing after measuring that both original M1a candidates
    exceeded this machine's RAM.
  - `730bc34` — recorded the private remote and the commit history in this file.
  - HEAD (this commit) — the approved M1 implementation plan, plus this rewritten prompt.
- `models/*/manifest.json` is tracked (56 KB + 4 KB); the `.bin` weights are gitignored.
  The `llama-8b-real-partition` manifest holds real GGUF tensor offsets and is the input
  M3 will want. The `test-partition` one is output from the simulated converter — its
  numbers are synthetic.
