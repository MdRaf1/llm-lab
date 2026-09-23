# src/llm_lab/frontier/expert_repack.py
"""M3 expert-contiguous repack: slice, plan, write, verify. Layout-only, output-exact."""
from __future__ import annotations

EXPERT_PARTS = ("gate", "up", "down")
ALIGN = 4096
HEADROOM_BYTES = 10 * (2 ** 30)
PACKED_NAME = "qwen3-experts.packed"
MANIFEST_NAME = "manifest.json"


def expert_tensor_name(layer: int, part: str) -> str:
    return f"blk.{layer}.ffn_{part}_exps.weight"


def per_expert_bytes(tensor, n_expert: int) -> int:
    name = tensor.name
    if len(tensor.shape) != 3:
        raise ValueError(f"{name}: expected 3-D stacked expert tensor, got shape {list(tensor.shape)}")
    if int(tensor.shape[-1]) != n_expert:
        raise ValueError(f"{name}: outer dim {int(tensor.shape[-1])} != expert_count {n_expert}")
    if tensor.n_bytes % n_expert:
        raise ValueError(f"{name}: {tensor.n_bytes} bytes not divisible by expert_count {n_expert}")
    return tensor.n_bytes // n_expert


def expert_slice(tensor, e: int, n_expert: int) -> bytes:
    per_expert_bytes(tensor, n_expert)  # guards
    return tensor.data[e].tobytes()


def align_up(n: int, a: int = ALIGN) -> int:
    return -(-n // a) * a


def plan_layout(reader, n_layer: int, n_expert: int) -> tuple[list[dict], int]:
    tmap = {t.name: t for t in reader.tensors}
    blobs: list[dict] = []
    offset = 0
    for layer in range(n_layer):
        pers = {}
        for part in EXPERT_PARTS:
            t = tmap.get(expert_tensor_name(layer, part))
            if t is None:
                raise ValueError(f"missing expert tensor {expert_tensor_name(layer, part)}")
            pers[part] = (per_expert_bytes(t, n_expert), t.tensor_type.name)
        for e in range(n_expert):
            sub = 0
            roles = {}
            for part in EXPERT_PARTS:
                length, dtype = pers[part]
                roles[part] = {"sub_offset": sub, "length": length, "dtype": dtype}
                sub += length
            offset = align_up(offset)
            blobs.append({"layer": layer, "expert": e, "offset": offset, "length": sub, "roles": roles})
            offset += sub
    expected = align_up(blobs[-1]["offset"] + blobs[-1]["length"]) if blobs else 0
    return blobs, expected


def check_disk(expected_output_bytes: int, free_bytes: int,
               headroom_bytes: int = HEADROOM_BYTES) -> dict:
    required = expected_output_bytes + headroom_bytes
    report = {
        "expected_output_bytes": expected_output_bytes,
        "headroom_bytes": headroom_bytes,
        "required_bytes": required,
        "available_free_bytes": free_bytes,
    }
    if required > free_bytes:
        raise RuntimeError(f"PREFLIGHT_FAIL required={required} available={free_bytes}")
    return report


def write_packed(reader, blobs, n_expert, out_dir, expected_bytes, *, free_bytes) -> list[dict]:
    import hashlib
    import os
    from pathlib import Path
    check_disk(expected_bytes, free_bytes)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / (PACKED_NAME + ".tmp")
    tmap = {t.name: t for t in reader.tensors}
    try:
        with open(tmp, "wb") as f:
            for b in blobs:
                pad = b["offset"] - f.tell()
                if pad > 0:
                    f.write(b"\x00" * pad)
                for part in EXPERT_PARTS:
                    t = tmap[expert_tensor_name(b["layer"], part)]
                    data = expert_slice(t, b["expert"], n_expert)
                    b["roles"][part]["src_sha256"] = hashlib.sha256(data).hexdigest()
                    f.write(data)
        os.replace(tmp, out / PACKED_NAME)
    except OSError:
        if tmp.exists():
            tmp.unlink()
        raise
    return blobs


def verify_repack(packed_path, blobs) -> dict:
    import hashlib
    from pathlib import Path
    raw = Path(packed_path).read_bytes()
    pairs = []
    all_equal = True
    for b in blobs:
        for part, r in b["roles"].items():
            start = b["offset"] + r["sub_offset"]
            read_sha = hashlib.sha256(raw[start:start + r["length"]]).hexdigest()
            r["read_sha256"] = read_sha
            if read_sha != r["src_sha256"]:
                all_equal = False
            pairs.append(f"{r['src_sha256']}:{read_sha}")
    rollup = hashlib.sha256("\n".join(sorted(pairs)).encode("utf-8")).hexdigest()
    return {"n_slices": len(pairs), "all_equal": all_equal, "rollup_sha256": rollup}


def repack_model(input_path: str, out_dir: str) -> dict:
    """Orchestrate slice->plan->write->verify for one GGUF, emitting the packed file + manifest.

    source_sha256 is left None: hashing the 18 GB source is optional/slow; the per-slice
    src/read hashes in output_exact are the byte-identity witness.
    """
    import json
    import shutil
    from pathlib import Path
    import gguf
    from llm_lab.moe.meta import read_gguf_meta
    meta = read_gguf_meta(Path(input_path))
    reader = gguf.GGUFReader(str(input_path))
    blobs, expected = plan_layout(reader, meta.n_layer, meta.n_expert)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    blobs = write_packed(reader, blobs, meta.n_expert, out, expected,
                         free_bytes=shutil.disk_usage(out).free)
    witness = verify_repack(out / PACKED_NAME, blobs)
    manifest = {
        "source_file": Path(input_path).name, "source_sha256": None,
        "architecture": meta.model, "num_layers": meta.n_layer,
        "num_experts": meta.n_expert, "top_k": meta.n_expert_used,
        "packed_file": PACKED_NAME, "alignment": ALIGN,
        "output_exact": witness, "blobs": blobs,
    }
    (out / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
