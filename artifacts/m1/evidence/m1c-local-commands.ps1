$ErrorActionPreference = "Stop"

# =====================================================================
# M1c LOCAL evidence: Qwen3-30B-A3B Q4_K_M (GGUF via llama.cpp) on THIS
# 16 GB box. Implements brief Steps 1-3 plus a LOCAL PARTIAL gate.
#
# This script does NOT trace (llama.cpp routing needs a deferred C++
# patch; trace backends are tiny/hf-* only) and does NOT write the full
# m1c-gate.json. The BF16 routing trace + full gate live in the sibling
# script artifacts/m1/evidence/m1c-rented-commands.ps1, run later on a
# rented ~128 GB Windows host.
#
# Rerunnable. Every output-writing CLI refuses to overwrite without
# --force, so we delete ONLY this phase's own enumerated prior outputs
# (never a directory wholesale, never blind --force). The downloaded
# GGUF under models/ is deliberately NOT in the delete list -- re-fetch
# is expensive and the byte/hash assertion re-verifies it each run.
# Idioms mirrored from m1b-commands.ps1.
# =====================================================================

$outputs = @(
  # Evidence (committed by the controller after this script passes)
  "artifacts/m1/evidence/m1c-local-preflight.txt",
  "artifacts/m1/evidence/m1c-meta.json",
  "artifacts/m1/evidence/m1c-run-a.json",
  "artifacts/m1/evidence/m1c-run-b.json",
  "artifacts/m1/evidence/m1c-oracle.json",
  "artifacts/m1/evidence/m1c-local-gate.json",
  # Temp Python helpers (gitignored under raw/)
  "artifacts/m1/raw/m1c_local_preflight.py",
  "artifacts/m1/raw/m1c_local_derive.py"
)
foreach ($path in $outputs) {
  Remove-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
}

New-Item -ItemType Directory -Force -Path artifacts/m1/evidence | Out-Null
New-Item -ItemType Directory -Force -Path artifacts/m1/raw | Out-Null
New-Item -ItemType Directory -Force -Path models/m1/qwen3-gguf | Out-Null

# Pinned identities (verbatim from plan Global Constraints). Q4 file ONLY.
$GgufRepo  = "Qwen/Qwen3-30B-A3B-GGUF"
$GgufRev   = "e4d4bafdfb96a411a163846265362aceb0b9c63a"
$GgufFile  = "Qwen3-30B-A3B-Q4_K_M.gguf"
$GgufPath  = "models/m1/qwen3-gguf/Qwen3-30B-A3B-Q4_K_M.gguf"
$GgufBytes = 18556685824
$Prompt    = "The capital of France is"
$Preflight = "artifacts/m1/evidence/m1c-local-preflight.txt"

# =====================================================================
# Step 1: Refuse unless M1b passed, then preflight exact free disk.
#   Host facts, m1b-gate.passed assertion, and a required-vs-free disk
#   gate (official Q4 bytes + 10 GiB headroom). Distinct exit codes so
#   the transcript names which precondition failed.
# =====================================================================

# Single-quoted here-string so PS leaves Python '$' and text untouched.
@'
import json, os, shutil, platform, sys
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
import psutil

GGUF_BYTES = 18556685824
GGUF_PATH  = "models/m1/qwen3-gguf/Qwen3-30B-A3B-Q4_K_M.gguf"
HEADROOM   = 10 * (2 ** 30)

# Subtract bytes already on disk so a re-run with the GGUF present does not
# double-count it (mirrors the accepted M1b dry-run-delta approach; a plain
# stat suffices here since the official size is pinned). Step 2's streamed
# SHA-256 + exact byte-count assert is what guarantees the present file is
# complete -- this preflight only decides transfer room.
PRESENT     = os.path.getsize(GGUF_PATH) if os.path.exists(GGUF_PATH) else 0
TO_DOWNLOAD = max(0, GGUF_BYTES - PRESENT)
REQUIRED    = TO_DOWNLOAD + HEADROOM

def pkgver(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "absent"

# --- Step 1a: refuse unless M1b passed. -----------------------------
gate_path = Path("artifacts/m1/evidence/m1b-gate.json")
if not gate_path.exists():
    print("M1B_GATE_MISSING artifacts/m1/evidence/m1b-gate.json", file=sys.stderr)
    sys.exit(2)
m1b = json.loads(gate_path.read_text(encoding="utf-8"))
if m1b.get("passed") is not True:
    print(f"M1B_GATE_NOT_PASSED passed={m1b.get('passed')!r}", file=sys.stderr)
    sys.exit(2)

# --- Step 1b: host facts, flushed before the disk gate so a short-disk
#     failure still leaves a useful transcript. ---------------------
vm = psutil.virtual_memory()
host = {
    "cpu": platform.processor(),
    "platform": platform.platform(),
    "virtual_memory": {"total": vm.total, "available": vm.available},
    "free_disk_bytes": shutil.disk_usage(".").free,
    "packages": {p: pkgver(p) for p in
        ["huggingface_hub", "transformers", "llama_cpp_python",
         "torch", "psutil"]},
    "m1b_gate_passed": True,
}
print("=== host facts ===")
print(json.dumps(host, indent=2))
sys.stdout.flush()

# --- Step 1c: exact free-disk gate. ---------------------------------
free = shutil.disk_usage(".").free
report = {
    "gguf_official_bytes": GGUF_BYTES,
    "already_present_bytes": PRESENT,
    "to_download_bytes": TO_DOWNLOAD,
    "headroom_bytes": HEADROOM,
    "required_bytes": REQUIRED,
    "available_free_bytes": free,
}
print("=== disk preflight ===")
print(json.dumps(report, indent=2))
print(f"required={REQUIRED} available={free}")
sys.stdout.flush()
if REQUIRED > free:
    print(f"PREFLIGHT_FAIL required={REQUIRED} available={free}", file=sys.stderr)
    sys.exit(3)
print("PREFLIGHT_OK")
'@ | Out-File artifacts/m1/raw/m1c_local_preflight.py -Encoding utf8

"=== Get-PSDrive ===" | Out-File $Preflight -Encoding utf8
Get-PSDrive | Out-File -Append $Preflight -Encoding utf8

# 2>&1 on native tools under Stop can promote a benign stderr line to a
# fatal NativeCommandError, so relax to Continue and gate on exit codes.
$ErrorActionPreference = "Continue"
"=== local preflight (m1b gate + disk + host facts) ===" | Out-File -Append $Preflight -Encoding utf8
$pre = ./.venv/Scripts/python.exe artifacts/m1/raw/m1c_local_preflight.py 2>&1
$preExit = $LASTEXITCODE
$pre | Out-File -Append $Preflight -Encoding utf8
$pre | Write-Host
$ErrorActionPreference = "Stop"
if ($preExit -eq 2) {
  throw "M1c refused: M1b gate is absent or passed != true (see m1b-gate.json)."
} elseif ($preExit -eq 3) {
  $line = ($pre | Select-String 'required=\d+ available=\d+' | Select-Object -Last 1).Line
  throw "preflight failed (insufficient disk): $line"
} elseif ($preExit -ne 0) {
  throw "local preflight failed with exit $preExit"
}

# =====================================================================
# Step 2: Download ONLY the approved Q4_K_M file at the pinned revision.
#   (Idempotent: hardlinks/skips from a populated cache.) Then compute
#   the file's .NET SHA-256 AND exact byte count and assert the byte
#   count == official size EXACTLY before any load. Get-FileHash is
#   absent on this PS install, so hash via .NET, streamed (never read an
#   18.5 GiB file fully into memory on a 16 GB box).
# =====================================================================
./.venv/Scripts/hf.exe download $GgufRepo $GgufFile `
  --revision $GgufRev `
  --local-dir models/m1/qwen3-gguf

$fileInfo = Get-Item -LiteralPath $GgufPath
$actualBytes = $fileInfo.Length

$sha256 = [System.Security.Cryptography.SHA256]::Create()
$stream = [System.IO.File]::OpenRead($GgufPath)
try {
  $hashBytes = $sha256.ComputeHash($stream)
} finally {
  $stream.Dispose()
}
$GgufSha = ([BitConverter]::ToString($hashBytes)).Replace("-", "").ToLower()

"=== downloaded Q4 file identity ===" | Out-File -Append $Preflight -Encoding utf8
"path=$GgufPath"                       | Out-File -Append $Preflight -Encoding utf8
"expected_bytes=$GgufBytes"            | Out-File -Append $Preflight -Encoding utf8
"actual_bytes=$actualBytes"            | Out-File -Append $Preflight -Encoding utf8
"sha256=$GgufSha"                      | Out-File -Append $Preflight -Encoding utf8
Write-Host "Q4 file: bytes=$actualBytes sha256=$GgufSha"

if ($actualBytes -ne $GgufBytes) {
  throw "Q4 byte-count mismatch: got=$actualBytes expected=$GgufBytes (aborting before load)"
}

# =====================================================================
# Step 3: Metadata + deterministic Q4_K_M baseline (llama.cpp). Same
#   literal prompt, seed 0, threads 6, n-ctx 2048, max-new-tokens 256
#   for BOTH runs; compare by token ID. All numbers come from the tool
#   output -- never copy spec estimates.
# =====================================================================
./.venv/Scripts/llm-lab.exe moe meta --backend llama-cpp `
  --model-path $GgufPath `
  --output artifacts/m1/evidence/m1c-meta.json
./.venv/Scripts/llm-lab.exe moe run --backend llama-cpp `
  --model-path $GgufPath --prompt $Prompt `
  --seed 0 --threads 6 --n-ctx 2048 --max-new-tokens 256 `
  --output artifacts/m1/evidence/m1c-run-a.json
./.venv/Scripts/llm-lab.exe moe run --backend llama-cpp `
  --model-path $GgufPath --prompt $Prompt `
  --seed 0 --threads 6 --n-ctx 2048 --max-new-tokens 256 `
  --output artifacts/m1/evidence/m1c-run-b.json
./.venv/Scripts/llm-lab.exe moe compare `
  artifacts/m1/evidence/m1c-run-a.json artifacts/m1/evidence/m1c-run-b.json `
  --output artifacts/m1/evidence/m1c-oracle.json

# =====================================================================
# LOCAL PARTIAL gate: derive m1c-local-gate.json from the generated
#   files (never hardcoded). This covers only the LOCAL half of spec
#   §8.11; the rented script derives the full m1c-gate.json. JSON in
#   Python, not fragile PS munging.
# =====================================================================
@'
import json, sys
from pathlib import Path

EV = Path("artifacts/m1/evidence")

def load(p):
    return json.loads((EV / p).read_text(encoding="utf-8"))

run_a  = load("m1c-run-a.json")
oracle = load("m1c-oracle.json")
meta   = load("m1c-meta.json")

# local_generation_real: run-a produced real tokens, exactly 256 of them.
local_generation_real = (
    isinstance(run_a.get("token_ids"), list)
    and len(run_a["token_ids"]) > 0
    and int(run_a.get("n_generated", -1)) == 256
)

# q4_deterministic: the two llama.cpp runs are token-identical.
q4_deterministic = (
    oracle.get("identical") is True
    and oracle.get("fingerprints_match") is True
    and oracle.get("first_divergence") is None
)

# measured_metadata: layer/expert facts are integers (report the values).
# Under mixed per-layer quant, total_expert_bytes may be an int while
# expert_bytes is null -- that is expected; we do NOT require non-null.
meta_ints = {k: meta.get(k) for k in ("n_layer", "n_expert", "n_expert_used")}
measured_metadata = all(isinstance(v, int) for v in meta_ints.values())

gate = {
    "local_generation_real": bool(local_generation_real),
    "q4_deterministic": bool(q4_deterministic),
    "measured_metadata": bool(measured_metadata),
    "metadata_values": meta_ints,
    "total_expert_bytes": meta.get("total_expert_bytes"),
    "expert_bytes": meta.get("expert_bytes"),
    "n_generated": run_a.get("n_generated"),
}
gate["passed_local"] = (
    gate["local_generation_real"]
    and gate["q4_deterministic"]
    and gate["measured_metadata"]
)
(EV / "m1c-local-gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
print(json.dumps(gate, indent=2))
if not gate["passed_local"]:
    print("LOCAL_GATE_FAILED", file=sys.stderr)
    sys.exit(4)
'@ | Out-File artifacts/m1/raw/m1c_local_derive.py -Encoding utf8

$ErrorActionPreference = "Continue"
$derive = ./.venv/Scripts/python.exe artifacts/m1/raw/m1c_local_derive.py 2>&1
$deriveExit = $LASTEXITCODE
$derive | Write-Host
$ErrorActionPreference = "Stop"
if ($deriveExit -ne 0) {
  throw "M1c LOCAL gate failed (passed_local != true); see m1c-local-gate.json / m1c-local-output.txt"
}

Write-Host "M1c LOCAL complete: Q4 baseline deterministic, metadata measured, local gate passed."
Write-Host "BF16 routing trace + full m1c-gate.json are produced later by the rented script"
Write-Host "  artifacts/m1/evidence/m1c-rented-commands.ps1 on a ~128 GB host. Controller commits."
