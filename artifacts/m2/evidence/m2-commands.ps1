# M2 Task 7 — exact commands run (PowerShell / showcase-box form).
# CLI invoked via the venv interpreter; bare python / hf are not on PATH.
# NOTE: the meta backend flag is `llama-cpp` (the task brief's `gguf` is not a
# valid --backend choice; the CLI accepts {hf, llama-cpp}). What is recorded
# here is what was actually executed.

# 0. Confirm the MoE trace suite is still green.
./.venv/Scripts/python.exe tests/test_moe_trace.py

# 1. Capture the 30B Q4_K_M geometry (measured, incl. exact non-expert core).
./.venv/Scripts/python.exe -m llm_lab moe meta `
  --backend llama-cpp `
  --model-path models/m1/qwen3-gguf/Qwen3-30B-A3B-Q4_K_M.gguf `
  --output artifacts/m2/evidence/m2-geometry-30b.json

# 2. Project the M2 tier gate from the 5 surviving int8-OLMoE traces
#    (trace+logits paired 1:1) onto the 30B Q4 geometry. Tee console to file.
./.venv/Scripts/python.exe -m llm_lab moe analyze `
  --trace artifacts/m1/raw/m1b-trace-prose-a.jsonl --logits artifacts/m1/raw/m1b-logits-prose-a.pt `
  --trace artifacts/m1/raw/m1b-trace-prose-b.jsonl --logits artifacts/m1/raw/m1b-logits-prose-b.pt `
  --trace artifacts/m1/raw/m1b-trace-code-a.jsonl  --logits artifacts/m1/raw/m1b-logits-code-a.pt `
  --trace artifacts/m1/raw/m1b-trace-code-b.jsonl  --logits artifacts/m1/raw/m1b-logits-code-b.pt `
  --trace artifacts/m1/raw/m1b-trace-factual.jsonl --logits artifacts/m1/raw/m1b-logits-factual.pt `
  --geometry artifacts/m2/evidence/m2-geometry-30b.json `
  --output artifacts/m2/evidence/m2-gate.json 2>&1 |
  Tee-Object artifacts/m2/evidence/m2-analyze-output.txt

# 3. Hash every evidence artifact + the 5 input traces (manifest excludes itself).
./.venv/Scripts/python.exe -c "import hashlib,glob; [print(hashlib.sha256(open(f,'rb').read()).hexdigest(), f) for f in sorted([p for p in glob.glob('artifacts/m2/evidence/*') if not p.endswith('m2-files.sha256')]+glob.glob('artifacts/m1/raw/m1b-trace-*.jsonl'))]" > artifacts/m2/evidence/m2-files.sha256
