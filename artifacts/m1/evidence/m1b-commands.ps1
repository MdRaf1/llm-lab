$ErrorActionPreference = "Stop"

# =====================================================================
# M1b evidence: OLMoE-1B-7B on BOTH numeric paths (GGUF Q4_K_M via
# llama.cpp, HF int8 for routing). Produces m1b-gate.json, which gates
# all Qwen work. Rerunnable. Every output-writing CLI refuses to
# overwrite without --force, so we delete ONLY this phase's own
# enumerated prior outputs (never a directory wholesale, never blind
# --force). Idioms mirrored from m1a-commands.ps1.
# =====================================================================

$outputs = @(
  # Evidence (committed)
  "artifacts/m1/evidence/m1b-preflight.txt",
  "artifacts/m1/evidence/m1b-gguf-meta.json",
  "artifacts/m1/evidence/m1b-gguf-run-a.json",
  "artifacts/m1/evidence/m1b-gguf-run-b.json",
  "artifacts/m1/evidence/m1b-gguf-oracle.json",
  "artifacts/m1/evidence/m1b-hf-run-a.json",
  "artifacts/m1/evidence/m1b-hf-run-b.json",
  "artifacts/m1/evidence/m1b-hf-oracle.json",
  "artifacts/m1/evidence/m1b-trace-prose-a-summary.json",
  "artifacts/m1/evidence/m1b-trace-prose-b-summary.json",
  "artifacts/m1/evidence/m1b-trace-code-a-summary.json",
  "artifacts/m1/evidence/m1b-trace-code-b-summary.json",
  "artifacts/m1/evidence/m1b-trace-factual-summary.json",
  "artifacts/m1/evidence/m1b-trace-summary.json",
  "artifacts/m1/evidence/m1b-cache-64.json",
  "artifacts/m1/evidence/m1b-cache-512.json",
  "artifacts/m1/evidence/m1b-files.sha256",
  "artifacts/m1/evidence/m1b-gate.json",
  # Raw (gitignored) traces / logits / run-records
  "artifacts/m1/raw/m1b-trace-prose-a.jsonl",
  "artifacts/m1/raw/m1b-trace-prose-b.jsonl",
  "artifacts/m1/raw/m1b-trace-code-a.jsonl",
  "artifacts/m1/raw/m1b-trace-code-b.jsonl",
  "artifacts/m1/raw/m1b-trace-factual.jsonl",
  "artifacts/m1/raw/m1b-logits-prose-a.pt",
  "artifacts/m1/raw/m1b-logits-prose-b.pt",
  "artifacts/m1/raw/m1b-logits-code-a.pt",
  "artifacts/m1/raw/m1b-logits-code-b.pt",
  "artifacts/m1/raw/m1b-logits-factual.pt",
  "artifacts/m1/raw/m1b-trace-prose-a-run.json",
  "artifacts/m1/raw/m1b-trace-prose-b-run.json",
  "artifacts/m1/raw/m1b-trace-code-a-run.json",
  "artifacts/m1/raw/m1b-trace-code-b-run.json",
  "artifacts/m1/raw/m1b-trace-factual-run.json",
  # Temp Python helpers (gitignored under raw/)
  "artifacts/m1/raw/m1b_preflight.py",
  "artifacts/m1/raw/m1b_derive.py"
)
foreach ($path in $outputs) {
  Remove-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
}

New-Item -ItemType Directory -Force -Path artifacts/m1/evidence | Out-Null
New-Item -ItemType Directory -Force -Path artifacts/m1/raw | Out-Null
New-Item -ItemType Directory -Force -Path models/m1/olmoe-gguf | Out-Null

# Pinned identities (verbatim from plan Global Constraints). NEVER Qwen.
$GgufRepo = "allenai/OLMoE-1B-7B-0924-GGUF"
$GgufRev  = "70df85ed7132bf21b5acbcb33817f58e7d3cb949"
$GgufFile = "olmoe-1b-7b-0924-q4_k_m.gguf"
$GgufPath = "models/m1/olmoe-gguf/olmoe-1b-7b-0924-q4_k_m.gguf"
$HfRepo   = "allenai/OLMoE-1B-7B-0924"
$HfRev    = "6d84c48581ece794365f2b8e9cfb043c68ade9c5"
$Prompt   = "The capital of France is"
$Preflight = "artifacts/m1/evidence/m1b-preflight.txt"

# =====================================================================
# Step 1: Preflight + pinned-revision GGUF download.
#   Host facts, dry-run byte totals (EXACT ints via the HF dry_run API,
#   not the rounded/pinned constant), and a required-vs-free disk gate.
# =====================================================================

# Write the exact-byte preflight helper. Single-quoted here-string so PS
# leaves Python '$' and text untouched. The installed `hf` CLI supports
# `--dry-run --json` but its JSON emits only human sizes ("4.2G") and "-"
# for cached files -- unusable for exact-byte math. The same underlying
# dry-run via the Python API returns DryRunFileInfo.file_size (int) and
# .will_download (bool), so the math uses those measured ints. See report.
@'
import json, shutil, platform, sys
from importlib.metadata import version, PackageNotFoundError
import psutil
from huggingface_hub import snapshot_download, hf_hub_download

GGUF_REPO = "allenai/OLMoE-1B-7B-0924-GGUF"
GGUF_REV  = "70df85ed7132bf21b5acbcb33817f58e7d3cb949"
GGUF_FILE = "olmoe-1b-7b-0924-q4_k_m.gguf"
GGUF_LOCAL_DIR = "models/m1/olmoe-gguf"
HF_REPO   = "allenai/OLMoE-1B-7B-0924"
HF_REV    = "6d84c48581ece794365f2b8e9cfb043c68ade9c5"
HEADROOM  = 4 * (2 ** 30)

def pkgver(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "absent"

# Host facts first, flushed before any network call, so a dry-run failure
# still leaves a useful transcript.
vm = psutil.virtual_memory()
host = {
    "cpu": platform.processor(),
    "platform": platform.platform(),
    "virtual_memory": {"total": vm.total, "available": vm.available},
    "free_disk_bytes": shutil.disk_usage(".").free,
    "packages": {p: pkgver(p) for p in
        ["huggingface_hub", "transformers", "bitsandbytes", "torch",
         "llama_cpp_python", "accelerate", "psutil"]},
}
print("=== host facts ===")
print(json.dumps(host, indent=2))
sys.stdout.flush()

def info(f):
    return {"filename": f.filename, "file_size": f.file_size,
            "will_download": f.will_download, "is_cached": f.is_cached}

# GGUF single file, dry-run against the real --local-dir target so
# will_download reflects local-dir presence (what actually consumes disk).
gguf_files = [hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILE,
                              revision=GGUF_REV, local_dir=GGUF_LOCAL_DIR,
                              dry_run=True)]
# HF int8 shards stay in the HF cache; dry-run reflects cache presence.
hf_files = snapshot_download(repo_id=HF_REPO, revision=HF_REV, dry_run=True)

def totals(files):
    remote  = sum(f.file_size for f in files)
    present = sum(f.file_size for f in files if not f.will_download)
    todl    = sum(f.file_size for f in files if f.will_download)
    return remote, present, todl

g_remote, g_present, g_todl = totals(gguf_files)
h_remote, h_present, h_todl = totals(hf_files)

remote_total  = g_remote + h_remote
present_total = g_present + h_present
# required = remote bytes + 4 GiB headroom - already-present bytes
#          = bytes still to transfer + headroom.
required = remote_total + HEADROOM - present_total
free = shutil.disk_usage(".").free

report = {
    "gguf": {"repo": GGUF_REPO, "revision": GGUF_REV,
             "files": [info(f) for f in gguf_files],
             "remote_bytes": g_remote, "already_present_bytes": g_present,
             "to_download_bytes": g_todl},
    "hf": {"repo": HF_REPO, "revision": HF_REV,
           "files": [info(f) for f in hf_files],
           "remote_bytes": h_remote, "already_present_bytes": h_present,
           "to_download_bytes": h_todl},
    "remote_total_bytes": remote_total,
    "already_present_total_bytes": present_total,
    "headroom_bytes": HEADROOM,
    "required_bytes": required,
    "available_free_bytes": free,
}
print("=== dry-run exact byte math ===")
print(json.dumps(report, indent=2))
print(f"required={required} available={free}")
sys.stdout.flush()
if required > free:
    print(f"PREFLIGHT_FAIL required={required} available={free}", file=sys.stderr)
    sys.exit(3)
print("PREFLIGHT_OK")
'@ | Out-File artifacts/m1/raw/m1b_preflight.py -Encoding utf8

# PowerShell host facts + the raw `hf download --dry-run --json` for both
# targets, saved verbatim into the transcript (brief Step 1 literal).
"=== Get-PSDrive ===" | Out-File $Preflight -Encoding utf8
Get-PSDrive | Out-File -Append $Preflight -Encoding utf8

# 2>&1 on native tools under Stop can promote a benign stderr line to a
# fatal NativeCommandError, so relax to Continue and gate on exit codes.
$ErrorActionPreference = "Continue"

"=== hf download --dry-run --json (GGUF) ===" | Out-File -Append $Preflight -Encoding utf8
./.venv/Scripts/hf.exe download $GgufRepo $GgufFile `
  --revision $GgufRev --local-dir models/m1/olmoe-gguf --dry-run --json 2>&1 |
  Out-File -Append $Preflight -Encoding utf8

"=== hf download --dry-run --json (HF repo) ===" | Out-File -Append $Preflight -Encoding utf8
./.venv/Scripts/hf.exe download $HfRepo `
  --revision $HfRev --dry-run --json 2>&1 |
  Out-File -Append $Preflight -Encoding utf8

"=== exact-byte preflight (HF dry_run API) ===" | Out-File -Append $Preflight -Encoding utf8
$pre = ./.venv/Scripts/python.exe artifacts/m1/raw/m1b_preflight.py 2>&1
$preExit = $LASTEXITCODE
$pre | Out-File -Append $Preflight -Encoding utf8
$pre | Write-Host
$ErrorActionPreference = "Stop"
if ($preExit -ne 0) {
  $line = ($pre | Select-String 'required=\d+ available=\d+' | Select-Object -Last 1).Line
  throw "preflight failed (insufficient disk): $line"
}

# Pinned-revision download (idempotent: hardlinks/skips from populated cache).
./.venv/Scripts/hf.exe download $GgufRepo $GgufFile `
  --revision $GgufRev `
  --local-dir models/m1/olmoe-gguf

# =====================================================================
# Step 2: GGUF metadata + deterministic Q4_K_M baseline (llama.cpp).
#   Never compare these token IDs against HF int8.
# =====================================================================
./.venv/Scripts/llm-lab.exe moe meta --backend llama-cpp `
  --model-path $GgufPath `
  --output artifacts/m1/evidence/m1b-gguf-meta.json
./.venv/Scripts/llm-lab.exe moe run --backend llama-cpp `
  --model-path $GgufPath --prompt $Prompt `
  --seed 0 --threads 6 --n-ctx 1024 --max-new-tokens 64 `
  --output artifacts/m1/evidence/m1b-gguf-run-a.json
./.venv/Scripts/llm-lab.exe moe run --backend llama-cpp `
  --model-path $GgufPath --prompt $Prompt `
  --seed 0 --threads 6 --n-ctx 1024 --max-new-tokens 64 `
  --output artifacts/m1/evidence/m1b-gguf-run-b.json
./.venv/Scripts/llm-lab.exe moe compare `
  artifacts/m1/evidence/m1b-gguf-run-a.json artifacts/m1/evidence/m1b-gguf-run-b.json `
  --output artifacts/m1/evidence/m1b-gguf-oracle.json

# =====================================================================
# Step 3: HF int8 determinism (INT8_BITSANDBYTES, set by the CLI).
# =====================================================================
./.venv/Scripts/llm-lab.exe moe run --backend hf-int8 `
  --model-id $HfRepo --revision $HfRev --prompt $Prompt `
  --seed 0 --threads 6 --max-new-tokens 64 `
  --output artifacts/m1/evidence/m1b-hf-run-a.json
./.venv/Scripts/llm-lab.exe moe run --backend hf-int8 `
  --model-id $HfRepo --revision $HfRev --prompt $Prompt `
  --seed 0 --threads 6 --max-new-tokens 64 `
  --output artifacts/m1/evidence/m1b-hf-run-b.json
./.venv/Scripts/llm-lab.exe moe compare `
  artifacts/m1/evidence/m1b-hf-run-a.json artifacts/m1/evidence/m1b-hf-run-b.json `
  --output artifacts/m1/evidence/m1b-hf-oracle.json

# =====================================================================
# Step 4: Exactly 500 processed positions = five prompts x 100 steps.
#   hf-int8, seed 0, threads 6, max-new-tokens 100, --continue-after-eos
#   (stop_at_eos:false). Each prompt -> own raw trace/logits/run + a
#   compact per-trace summary under evidence.
# =====================================================================
$traces = @(
  @{ name = "prose-a";  prompt = "The morning fog rolled over the quiet harbor as the fishing boats prepared to depart." },
  @{ name = "prose-b";  prompt = "In the years after the war, the small town slowly rebuilt itself street by street." },
  @{ name = "code-a";   prompt = "import numpy as np; def normalize(x): return x / np.linalg.norm(x)" },
  @{ name = "code-b";   prompt = "SELECT name, COUNT(*) FROM users GROUP BY name ORDER BY COUNT(*) DESC" },
  @{ name = "factual";  prompt = "The chemical symbol for gold is" }
)
foreach ($t in $traces) {
  $n = $t.name
  ./.venv/Scripts/llm-lab.exe moe trace --backend hf-int8 `
    --model-id $HfRepo --revision $HfRev --prompt $t.prompt `
    --seed 0 --threads 6 --max-new-tokens 100 --continue-after-eos `
    --trace      "artifacts/m1/raw/m1b-trace-$n.jsonl" `
    --logits     "artifacts/m1/raw/m1b-logits-$n.pt" `
    --run-record "artifacts/m1/raw/m1b-trace-$n-run.json" `
    --summary    "artifacts/m1/evidence/m1b-trace-$n-summary.json"
}

# =====================================================================
# Step 5: Replay ONE representative trace (prose-a) at capacities 64 and
#   512. Controller ruling: File Map pins exactly two cache files; two
#   caps on one immutable trace fully demonstrate the invariant. The
#   observational-only proof is derived in Python (Step 6).
# =====================================================================
./.venv/Scripts/llm-lab.exe moe replay-cache `
  artifacts/m1/raw/m1b-trace-prose-a.jsonl `
  --logits artifacts/m1/raw/m1b-logits-prose-a.pt `
  --capacity-experts 64 `
  --output artifacts/m1/evidence/m1b-cache-64.json
./.venv/Scripts/llm-lab.exe moe replay-cache `
  artifacts/m1/raw/m1b-trace-prose-a.jsonl `
  --logits artifacts/m1/raw/m1b-logits-prose-a.pt `
  --capacity-experts 512 `
  --output artifacts/m1/evidence/m1b-cache-512.json

Write-Host "Cache replay did not execute or alter model logits. Identical logits means both"
Write-Host "capacity analyses reference the same content-addressed logits artifact."

# =====================================================================
# SHA-256 manifest of the raw traces + logits. Get-FileHash is absent on
# this PS install, so compute via .NET (same table shape m1a emitted).
# raw_hashes_present (Step 6) is derived from these entries.
# =====================================================================
$sha256 = [System.Security.Cryptography.SHA256]::Create()
Get-ChildItem `
  artifacts/m1/raw/m1b-trace-prose-a.jsonl, artifacts/m1/raw/m1b-logits-prose-a.pt, `
  artifacts/m1/raw/m1b-trace-prose-b.jsonl, artifacts/m1/raw/m1b-logits-prose-b.pt, `
  artifacts/m1/raw/m1b-trace-code-a.jsonl,  artifacts/m1/raw/m1b-logits-code-a.pt, `
  artifacts/m1/raw/m1b-trace-code-b.jsonl,  artifacts/m1/raw/m1b-logits-code-b.pt, `
  artifacts/m1/raw/m1b-trace-factual.jsonl, artifacts/m1/raw/m1b-logits-factual.pt | ForEach-Object {
  $bytes = [System.IO.File]::ReadAllBytes($_.FullName)
  $hash = ([BitConverter]::ToString($sha256.ComputeHash($bytes))).Replace("-", "")
  [PSCustomObject]@{ Algorithm = "SHA256"; Hash = $hash; Path = $_.FullName }
} | Format-Table -AutoSize | Out-File `
  artifacts/m1/evidence/m1b-files.sha256 -Encoding utf8 -Width 200

# =====================================================================
# Step 6: Derive m1b-trace-summary.json and m1b-gate.json from the
#   generated files (never hardcoded). JSON parsing in Python, not
#   fragile PS munging.
# =====================================================================
@'
import json, hashlib, sys
from pathlib import Path

EV = Path("artifacts/m1/evidence")

def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))

def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

trace_names = ["prose-a", "prose-b", "code-a", "code-b", "factual"]

# --- trace-summary: sum exactly the five 100-step per-trace summaries,
#     list each source summary's SHA-256. Any step count != 100 fails.
sources = []
per = {}
total = 0
all_100 = True
for name in trace_names:
    sp = EV / f"m1b-trace-{name}-summary.json"
    s = load(sp)
    n = int(s["n_steps"])
    per[name] = n
    total += n
    if n != 100:
        all_100 = False
    sources.append({"name": name, "summary": sp.as_posix(),
                    "sha256": sha256_file(sp), "n_steps": n})

trace_summary = {
    "total_steps": total,
    "trace_count": len(trace_names),
    "steps_per_trace": per,
    "all_traces_100_steps": all_100,
    "sources": sources,
}
(EV / "m1b-trace-summary.json").write_text(
    json.dumps(trace_summary, indent=2), encoding="utf-8")

# --- cache observational-only: two caps on the same trace must agree on
#     trace_sha256, logits_sha256 and token_ids while hit/miss differ.
c64 = load(EV / "m1b-cache-64.json")
c512 = load(EV / "m1b-cache-512.json")
cache_obs = (
    c64["trace_sha256"] == c512["trace_sha256"]
    and c64["logits_sha256"] == c512["logits_sha256"]
    and c64["token_ids"] == c512["token_ids"]
    and (c64["hits"], c64["misses"]) != (c512["hits"], c512["misses"])
)

# --- within-path oracles
gguf_ok = bool(load(EV / "m1b-gguf-oracle.json")["identical"])
hf_ok = bool(load(EV / "m1b-hf-oracle.json")["identical"])

# --- positions: summed steps >= 500 AND every trace exactly 100
positions_ok = all_100 and total >= 500

# --- raw hashes present: manifest names every raw trace + logits file
manifest = (EV / "m1b-files.sha256").read_text(encoding="utf-8")
raw_needed = []
for name in trace_names:
    raw_needed.append(f"m1b-trace-{name}.jsonl")
    raw_needed.append(f"m1b-logits-{name}.pt")
raw_hashes_present = all(fn in manifest for fn in raw_needed)

gate = {
    "gguf_deterministic": gguf_ok,
    "hf_int8_deterministic": hf_ok,
    "positions_at_least_500": positions_ok,
    "cache_observational_only": cache_obs,
    "raw_hashes_present": raw_hashes_present,
}
gate["passed"] = all(gate.values())
(EV / "m1b-gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
print(json.dumps(gate, indent=2))
if not gate["passed"]:
    print("GATE_FAILED", file=sys.stderr)
    sys.exit(4)
'@ | Out-File artifacts/m1/raw/m1b_derive.py -Encoding utf8

$ErrorActionPreference = "Continue"
$derive = ./.venv/Scripts/python.exe artifacts/m1/raw/m1b_derive.py 2>&1
$deriveExit = $LASTEXITCODE
$derive | Write-Host
$ErrorActionPreference = "Stop"
if ($deriveExit -ne 0) { throw "M1b gate failed (passed != true); see m1b-gate.json / m1b-output.txt" }

# =====================================================================
# Step 7: Regression tests after the real-model gate. A passing run
#   intentionally writes one "error:" line to stderr; relax to Continue
#   for the capture and gate on the real exit code.
# =====================================================================
$ErrorActionPreference = "Continue"
./.venv/Scripts/python.exe tests/test_moe_trace.py 2>&1 | Write-Host
$testExit = $LASTEXITCODE
$ErrorActionPreference = "Stop"
if ($testExit -ne 0) { throw "regression test suite failed with exit $testExit" }

Write-Host "M1b complete: gate passed, 500 positions captured, regression tests green."
