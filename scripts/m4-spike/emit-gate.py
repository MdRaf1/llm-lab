#!/usr/bin/env python
"""Emit driver for the M4 baseline spike (Task 3, resolution 2).

Reads baseline-runs.jsonl, DROPS state=="warmup" lines, groups the surviving
cold/warm runs, and computes per-run miss-bytes/token = disk_bytes/tokens
(physical PhysicalDisk counter reads, never cache-size arithmetic). Feeds those
into aggregate.build_gate with the independent M3 same-host scatter bandwidth
and writes the returned gate dict to artifacts/m4/evidence/m4-gate.json.

Run: ./.venv/Scripts/python.exe scripts/m4-spike/emit-gate.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate import build_gate, PER_TOKEN_EXPERT_BYTES  # noqa: E402

# Independent scatter bandwidth — M3 committed same-host microbench (resolution 1).
# NOT derived from this spike's decode throughput (that would be circular).
SCATTER_BPS = 1021317250.7
SCATTER_PROVENANCE = (
    "M3 microbench, same host "
    "(artifacts/m3/evidence/m3-gate.json:throughput.stock_bps.median)"
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
JSONL = os.path.join(REPO, "artifacts", "m4", "evidence", "baseline-runs.jsonl")
GATE = os.path.join(REPO, "artifacts", "m4", "evidence", "m4-gate.json")


def main():
    runs = {"cold_tok_s": [], "cold_miss_bytes_per_token": [],
            "warm_tok_s": [], "warm_miss_bytes_per_token": []}
    cold_verified_all = []
    with open(JSONL, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            state = r["state"]
            if state == "warmup":       # cache-fill runs, discarded in aggregation
                continue
            mbpt = r["disk_bytes"] / r["tokens"]
            if state == "cold":
                runs["cold_tok_s"].append(r["tok_s"])
                runs["cold_miss_bytes_per_token"].append(mbpt)
                cold_verified_all.append(bool(r.get("cold_verified", False)))
            elif state == "warm":
                runs["warm_tok_s"].append(r["tok_s"])
                runs["warm_miss_bytes_per_token"].append(mbpt)
            else:
                raise ValueError(f"unexpected state: {state!r}")

    gate = build_gate(runs, SCATTER_BPS, SCATTER_PROVENANCE, PER_TOKEN_EXPERT_BYTES)
    gate["cold_verified_all"] = all(cold_verified_all) and len(cold_verified_all) > 0
    gate["cold_verified_runs"] = cold_verified_all
    gate["n_cold"] = len(runs["cold_tok_s"])
    gate["n_warm"] = len(runs["warm_tok_s"])

    with open(GATE, "w", encoding="utf-8") as fh:
        json.dump(gate, fh, indent=2)
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
