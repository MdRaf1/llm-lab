# M3 expert-contiguous repack — exact commands run on this machine/session.
# Step A: repack -> packed file + manifest.json; discharges the output-exact (byte-identity) gate.
./.venv/Scripts/python.exe -m llm_lab moe repack `
  --input models/m1/qwen3-gguf/Qwen3-30B-A3B-Q4_K_M.gguf `
  --output models/m3/qwen3-repack `
  2>&1 | Tee-Object artifacts/m3/evidence/m3-repack-output.txt

# Step B: bench-read -> m3-gate.json (stdout byte-identical to the file, via _emit).
./.venv/Scripts/python.exe -m llm_lab moe bench-read `
  --input models/m1/qwen3-gguf/Qwen3-30B-A3B-Q4_K_M.gguf `
  --repack models/m3/qwen3-repack `
  --output artifacts/m3/evidence/m3-gate.json `
  --seed 1234 --experts-per-layer 8 --runs 5 --threads 16 `
  2>&1 | Tee-Object artifacts/m3/evidence/m3-bench-output.txt

# Step C: content-address every evidence file (except the manifest) + the runtime manifest.json.
./.venv/Scripts/python.exe -c "import hashlib,glob; [print(hashlib.sha256(open(f,'rb').read()).hexdigest(), f) for f in sorted([p for p in glob.glob('artifacts/m3/evidence/*') if not p.endswith('m3-files.sha256')]+['models/m3/qwen3-repack/manifest.json'])]" > artifacts/m3/evidence/m3-files.sha256
