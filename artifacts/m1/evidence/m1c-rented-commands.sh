#!/usr/bin/env bash
set -euo pipefail

# =====================================================================
# M1c RENTED evidence (Linux/Vultr Ubuntu 22.04 x64 port of
# m1c-rented-commands.ps1): Qwen3-30B-A3B BF16 routing trace on a rented
# ~128 GB host (BF16 weights ~61 GiB, ~80 GB RAM). Implements brief
# Steps 4 + 6. Faithful port -- same steps, gate rules, values, prompts,
# and evidence files as the .ps1; only shell + paths + SHA tool differ.
#
# Precondition: the controller ran m1c-local-commands.ps1 on the main
# host AND committed the local evidence (m1c-local-gate.json,
# m1c-oracle.json, m1c-meta.json). This host is a fresh clone of that
# commit, so those files are present and read here as gate inputs.
#
# Setup NOT done here: git/curl/uv are installed by separate setup steps
# the controller gives the user. This script starts at `uv sync --frozen`.
#
# Run from repo root as:
#   bash artifacts/m1/evidence/m1c-rented-commands.sh 2>&1 \
#     | tee artifacts/m1/evidence/m1c-rented-output.txt
# (The script writes to stdout/stderr; it does NOT self-tee or git commit.)
# =====================================================================

# Pinned identities (verbatim from plan Global Constraints). BF16 repo.
HF_REPO="Qwen/Qwen3-30B-A3B"
HF_REV="ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"

# Overwrite discipline: delete ONLY this phase's own enumerated prior
# outputs (never a directory wholesale). Idempotent re-run.
outputs=(
  # Evidence (compact; copied back + committed by the controller)
  "artifacts/m1/evidence/m1c-trace-prose-a-summary.json"
  "artifacts/m1/evidence/m1c-trace-prose-b-summary.json"
  "artifacts/m1/evidence/m1c-trace-code-summary.json"
  "artifacts/m1/evidence/m1c-trace-factual-summary.json"
  "artifacts/m1/evidence/m1c-trace-summary.json"
  "artifacts/m1/evidence/m1c-files.sha256"
  "artifacts/m1/evidence/m1c-gate.json"
  # Raw (gitignored) traces / logits / run-records
  "artifacts/m1/raw/m1c-trace-prose-a.jsonl"
  "artifacts/m1/raw/m1c-trace-prose-b.jsonl"
  "artifacts/m1/raw/m1c-trace-code.jsonl"
  "artifacts/m1/raw/m1c-trace-factual.jsonl"
  "artifacts/m1/raw/m1c-logits-prose-a.pt"
  "artifacts/m1/raw/m1c-logits-prose-b.pt"
  "artifacts/m1/raw/m1c-logits-code.pt"
  "artifacts/m1/raw/m1c-logits-factual.pt"
  "artifacts/m1/raw/m1c-trace-prose-a-run.json"
  "artifacts/m1/raw/m1c-trace-prose-b-run.json"
  "artifacts/m1/raw/m1c-trace-code-run.json"
  "artifacts/m1/raw/m1c-trace-factual-run.json"
  # Temp Python helpers (gitignored under raw/)
  "artifacts/m1/raw/m1c_rented_preflight.py"
  "artifacts/m1/raw/m1c_rented_derive.py"
)
for path in "${outputs[@]}"; do
  rm -f "$path"
done

mkdir -p artifacts/m1/evidence
mkdir -p artifacts/m1/raw

# =====================================================================
# Step 4a-setup: build .venv from the committed UNIVERSAL lock. Do NOT
#   fall back to a re-resolve -- a changed lock would break
#   reproducibility; surface it so the controller decides.
# =====================================================================
if ! uv sync --frozen; then
  echo "BLOCKER: 'uv sync --frozen' failed (lock out of sync or resolve error);" >&2
  echo "  stop before downloading; report to controller (do NOT re-resolve)." >&2
  exit 1
fi

# =====================================================================
# NEW: native import smoke-check (fail-fast at ~\$0 before the 61 GB
#   download). Guards the top-level `from llama_cpp import Llama` and the
#   hf-bf16 loader so a bad native lib fails now, not after the download.
# =====================================================================
if ! .venv/bin/python -c "import llama_cpp; import llm_lab.moe.baseline; import torch; print('imports-ok', torch.__version__)"; then
  echo "BLOCKER: native import failed on this host; stop before downloading; report to controller" >&2
  exit 1
fi

# =====================================================================
# Step 4a: Preflight. Raw `hf download --dry-run --json` saved verbatim,
#   then EXACT remote byte total via the HF Python dry_run API (the CLI
#   JSON emits only human sizes like "61G" / "-" -- unusable for exact
#   math; DryRunFileInfo.file_size/.will_download give measured ints).
#   Gate: to_download + 16 GiB free disk AND 80 GiB available RAM.
# =====================================================================
echo "=== hf download --dry-run --json (BF16 repo) ==="
# Do not let a benign dry-run stderr line under `set -e` abort the run.
.venv/bin/hf download "$HF_REPO" --revision "$HF_REV" --dry-run --json 2>&1 || true

cat > artifacts/m1/raw/m1c_rented_preflight.py <<'PY'
import json, shutil, platform, sys
from importlib.metadata import version, PackageNotFoundError
import psutil
from huggingface_hub import snapshot_download

HF_REPO = "Qwen/Qwen3-30B-A3B"
HF_REV  = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
DISK_HEADROOM = 16 * (2 ** 30)
RAM_REQUIRED  = 80 * (2 ** 30)

def pkgver(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "absent"

# Host facts first, flushed before any gate, so a failure still leaves a
# useful transcript.
vm = psutil.virtual_memory()
host = {
    "cpu": platform.processor(),
    "platform": platform.platform(),
    "virtual_memory": {"total": vm.total, "available": vm.available},
    "free_disk_bytes": shutil.disk_usage(".").free,
    "packages": {p: pkgver(p) for p in
        ["huggingface_hub", "transformers", "torch", "accelerate", "psutil"]},
}
print("=== host facts ===")
print(json.dumps(host, indent=2))
sys.stdout.flush()

def info(f):
    return {"filename": f.filename, "file_size": f.file_size,
            "will_download": f.will_download, "is_cached": f.is_cached}

# BF16 shards live in the HF cache; load_hf_bf16 loads by id+revision, so
# dry-run against the cache reflects what actually consumes disk on load.
files = snapshot_download(repo_id=HF_REPO, revision=HF_REV, dry_run=True)
remote  = sum(f.file_size for f in files)
present = sum(f.file_size for f in files if not f.will_download)
todl    = sum(f.file_size for f in files if f.will_download)
# required = bytes still to transfer + headroom (present bytes already there).
required_disk = todl + DISK_HEADROOM
free = shutil.disk_usage(".").free
avail_ram = psutil.virtual_memory().available

report = {
    "repo": HF_REPO, "revision": HF_REV,
    "files": [info(f) for f in files],
    "remote_bytes": remote,
    "already_present_bytes": present,
    "to_download_bytes": todl,
    "disk_headroom_bytes": DISK_HEADROOM,
    "required_disk_bytes": required_disk,
    "available_free_disk_bytes": free,
    "required_ram_bytes": RAM_REQUIRED,
    "available_ram_bytes": avail_ram,
}
print("=== dry-run exact byte math ===")
print(json.dumps(report, indent=2))
print(f"required_disk={required_disk} available_disk={free} "
      f"required_ram={RAM_REQUIRED} available_ram={avail_ram}")
sys.stdout.flush()

if required_disk > free:
    print(f"PREFLIGHT_FAIL_DISK required={required_disk} available={free}", file=sys.stderr)
    sys.exit(3)
if avail_ram < RAM_REQUIRED:
    print(f"PREFLIGHT_FAIL_RAM required={RAM_REQUIRED} available={avail_ram}", file=sys.stderr)
    sys.exit(5)
print("PREFLIGHT_OK")
PY

echo "=== exact-byte preflight (HF dry_run API) + disk/RAM gate ==="
set +e
.venv/bin/python artifacts/m1/raw/m1c_rented_preflight.py
preExit=$?
set -e
if [ "$preExit" -eq 3 ]; then
  echo "rented preflight failed (insufficient disk); see PREFLIGHT_FAIL_DISK above" >&2
  exit 3
elif [ "$preExit" -eq 5 ]; then
  echo "rented preflight failed (insufficient RAM); see PREFLIGHT_FAIL_RAM above" >&2
  exit 5
elif [ "$preExit" -ne 0 ]; then
  echo "rented preflight failed with exit $preExit" >&2
  exit "$preExit"
fi

# =====================================================================
# Step 4b: Stage the pinned BF16 revision into the HF cache
#   (load_hf_bf16 loads by id+revision from cache -- no --local-dir).
#   uv sync already ran above.
# =====================================================================
.venv/bin/hf download "$HF_REPO" --revision "$HF_REV"

# =====================================================================
# Step 4c: Four literal prompts (two prose, one code, one factual
#   recall), each traced for exactly 500 processed generated positions.
#   --max-new-tokens 500 --continue-after-eos (stop_at_eos:false). Each
#   prompt -> own raw trace/logits/run under raw/ (gitignored) + a
#   compact summary under evidence. hf-bf16 only; pinned model/revision.
#   Same literal strings as the .ps1.
# =====================================================================
trace_names=("prose-a" "prose-b" "code" "factual")
trace_prompts=(
  "The morning fog rolled over the quiet harbor as the fishing boats prepared to depart."
  "In the years after the war, the small town slowly rebuilt itself street by street."
  "import numpy as np; def normalize(x): return x / np.linalg.norm(x)"
  "The chemical symbol for gold is"
)
for i in "${!trace_names[@]}"; do
  n="${trace_names[$i]}"
  p="${trace_prompts[$i]}"
  .venv/bin/llm-lab moe trace --backend hf-bf16 \
    --model-id "$HF_REPO" --revision "$HF_REV" --prompt "$p" \
    --seed 0 --threads 6 --max-new-tokens 500 --continue-after-eos \
    --trace      "artifacts/m1/raw/m1c-trace-$n.jsonl" \
    --logits     "artifacts/m1/raw/m1c-logits-$n.pt" \
    --run-record "artifacts/m1/raw/m1c-trace-$n-run.json" \
    --summary    "artifacts/m1/evidence/m1c-trace-$n-summary.json"
done

# =====================================================================
# SHA-256 manifest of the rented raw traces + logits, via coreutils
# sha256sum ("<hex>  <path>"). The gate's evidence_hashes_complete does a
# basename substring check over this text, so this shape is compatible.
# =====================================================================
sha256sum \
  artifacts/m1/raw/m1c-trace-prose-a.jsonl artifacts/m1/raw/m1c-logits-prose-a.pt \
  artifacts/m1/raw/m1c-trace-prose-b.jsonl artifacts/m1/raw/m1c-logits-prose-b.pt \
  artifacts/m1/raw/m1c-trace-code.jsonl    artifacts/m1/raw/m1c-logits-code.pt \
  artifacts/m1/raw/m1c-trace-factual.jsonl artifacts/m1/raw/m1c-logits-factual.pt \
  > artifacts/m1/evidence/m1c-files.sha256

# =====================================================================
# Step 6: Derive m1c-trace-summary.json (sum exactly the four 500-step
#   summaries = 2000) and the FULL m1c-gate.json (spec §8.11). Reads the
#   COMMITTED local evidence (m1c-local-gate.json, m1c-oracle.json,
#   m1c-meta.json) PLUS this host's rented outputs. Booleans only, no
#   hardcoded true. Identical derivation logic to the .ps1.
# =====================================================================
cat > artifacts/m1/raw/m1c_rented_derive.py <<'PY'
import json, hashlib, sys
from pathlib import Path

EV = Path("artifacts/m1/evidence")

def load(p):
    return json.loads((EV / p).read_text(encoding="utf-8"))

def sha256_file(p):
    return hashlib.sha256((EV / p).read_bytes()).hexdigest()

# Domain per trace: two prose, one code, one factual-recall.
traces = [
    ("prose-a", "prose",   "m1c-trace-prose-a-summary.json"),
    ("prose-b", "prose",   "m1c-trace-prose-b-summary.json"),
    ("code",    "code",    "m1c-trace-code-summary.json"),
    ("factual", "factual", "m1c-trace-factual-summary.json"),
]

# --- trace-summary: sum exactly the four 500-step per-trace summaries,
#     list each source's SHA-256. Any step count != 500 fails the gate.
sources = []
total = 0
all_500 = True
domains = set()
precisions = []
for name, domain, fn in traces:
    s = load(fn)
    n = int(s["n_steps"])
    total += n
    if n != 500:
        all_500 = False
    domains.add(domain)
    precisions.append(s.get("precision"))
    sources.append({"name": name, "domain": domain, "summary": fn,
                    "sha256": sha256_file(fn), "n_steps": n,
                    "precision": s.get("precision")})

trace_summary = {
    "total_steps": total,
    "trace_count": len(traces),
    "all_traces_500_steps": all_500,
    "domains": sorted(domains),
    "sources": sources,
}
(EV / "m1c-trace-summary.json").write_text(
    json.dumps(trace_summary, indent=2), encoding="utf-8")

# --- local half (from COMMITTED local evidence). -------------------
local_gate = load("m1c-local-gate.json")
oracle     = load("m1c-oracle.json")
meta       = load("m1c-meta.json")

local_generation_real = local_gate.get("local_generation_real") is True
q4_deterministic = (
    oracle.get("identical") is True
    and oracle.get("fingerprints_match") is True
    and oracle.get("first_divergence") is None
)
measured_metadata = all(
    isinstance(meta.get(k), int) for k in ("n_layer", "n_expert", "n_expert_used")
)

# --- rented half. --------------------------------------------------
positions_2000 = all_500 and total == 2000
three_domains  = len(domains) >= 3
bf16_precision = len(precisions) == len(traces) and all(p == "BF16" for p in precisions)

# evidence_hashes_complete: manifest names every rented raw trace + logits.
manifest = (EV / "m1c-files.sha256").read_text(encoding="utf-8")
raw_needed = []
for name, _domain, _fn in traces:
    raw_needed.append(f"m1c-trace-{name}.jsonl")
    raw_needed.append(f"m1c-logits-{name}.pt")
evidence_hashes_complete = all(fn in manifest for fn in raw_needed)

gate = {
    "local_generation_real": bool(local_generation_real),
    "q4_deterministic": bool(q4_deterministic),
    "positions_2000": bool(positions_2000),
    "three_domains": bool(three_domains),
    "bf16_precision": bool(bf16_precision),
    "measured_metadata": bool(measured_metadata),
    "evidence_hashes_complete": bool(evidence_hashes_complete),
    "total_trace_steps": total,
    "domains": sorted(domains),
}
gate["passed"] = all(
    gate[k] for k in (
        "local_generation_real", "q4_deterministic", "positions_2000",
        "three_domains", "bf16_precision", "measured_metadata",
        "evidence_hashes_complete",
    )
)
(EV / "m1c-gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
print(json.dumps(gate, indent=2))
if not gate["passed"]:
    print("GATE_FAILED", file=sys.stderr)
    sys.exit(4)
PY

set +e
.venv/bin/python artifacts/m1/raw/m1c_rented_derive.py
deriveExit=$?
set -e
if [ "$deriveExit" -ne 0 ]; then
  echo "M1c FULL gate failed (passed != true); see m1c-gate.json / m1c-rented-output.txt" >&2
  exit "$deriveExit"
fi

echo "M1c RENTED complete: 2000 BF16 positions across three domains, full gate passed."
echo "Copy back the compact evidence (summaries, m1c-files.sha256, m1c-gate.json) +"
echo "  m1c-rented-output.txt to the main host. Controller commits (this host does not)."
