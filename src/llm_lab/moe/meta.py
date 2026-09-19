"""
Checkpoint architecture facts, read strictly from the checkpoint itself.
Guarantees:
1. Every reported number was read out of the file. Nothing is defaulted, guessed, or filled in
   from a spec assumption, so this module can correct the numbers a design doc merely assumed.
2. A required key that is absent raises ValueError naming that key; a missing path raises
   FileNotFoundError naming the exact path.
3. GGUF expert bytes are measured from each expert tensor's real byte count, never derived from a
   nominal bits-per-weight: quantized block sizes are not derivable from a bit count.
4. Anything the file does not say stays None, which serializes to JSON null.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import gguf

from llm_lab.moe.baseline import _check_non_negative

# Only dtypes whose width is unambiguous. Anything else leaves expert_bytes unmeasured.
_HF_DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}

# Qwen3-MoE carries both keys and only moe_intermediate_size sizes one expert, so it wins.
# OLMoE carries intermediate_size alone.
_HF_EXPERT_WIDTH_KEYS = ("moe_intermediate_size", "intermediate_size")

# An MoE layer's experts live in three stacked tensors; all three must be present.
_EXPERT_TENSOR_PARTS = ("gate", "up", "down")


def _check_positive(name: str, value: object) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


@dataclass(frozen=True)
class ModelMeta:
    """What one checkpoint says about its own MoE shape. Becomes the trace header downstream."""

    model: str
    path: str
    quant: str | None
    n_layer: int
    n_expert: int
    n_expert_used: int
    expert_bytes: int | None
    total_expert_bytes: int | None

    def __post_init__(self) -> None:
        _check_positive("n_layer", self.n_layer)
        _check_positive("n_expert", self.n_expert)
        _check_positive("n_expert_used", self.n_expert_used)
        _check_non_negative("expert_bytes", self.expert_bytes, optional=True)
        _check_non_negative("total_expert_bytes", self.total_expert_bytes, optional=True)

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "path": self.path,
            "quant": self.quant,
            "n_layer": self.n_layer,
            "n_expert": self.n_expert,
            "n_expert_used": self.n_expert_used,
            "expert_bytes": self.expert_bytes,
            "total_expert_bytes": self.total_expert_bytes,
        }


def _require_file(path: Path, what: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{what} not found: {path}")
    return path


def _config_value(config: dict, keys: tuple[str, ...], path: Path) -> object:
    """Value of the first key the config actually has, or a ValueError naming what we needed."""
    for key in keys:
        if key in config:
            return config[key]
    raise ValueError(f"{path}: config.json is missing required key {' or '.join(keys)}")


def _config_int(config: dict, keys: tuple[str, ...], path: Path) -> int:
    value = _config_value(config, keys, path)
    if not isinstance(value, int):
        raise ValueError(f"{path}: {keys[-1]} must be an integer, got {value!r}")
    return value


def read_hf_meta(config_path: Path) -> ModelMeta:
    """Read a local HF config.json with stdlib json; the file path is the evidence.

    `quant` is the config's dtype string verbatim, since that is all an HF checkpoint says about
    its numeric format. `expert_bytes` is computed only when the dtype width is known.
    """
    config_path = _require_file(config_path, "HF config")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    model = _config_value(config, ("model_type",), config_path)
    hidden_size = _config_int(config, ("hidden_size",), config_path)
    expert_width = _config_int(config, _HF_EXPERT_WIDTH_KEYS, config_path)
    n_layer = _config_int(config, ("num_hidden_layers",), config_path)
    n_expert = _config_int(config, ("num_experts",), config_path)
    n_expert_used = _config_int(config, ("num_experts_per_tok",), config_path)

    dtype = config.get("torch_dtype")
    dtype_bytes = _HF_DTYPE_BYTES.get(dtype)
    expert_bytes = None if dtype_bytes is None else 3 * hidden_size * expert_width * dtype_bytes

    return ModelMeta(
        model=str(model),
        path=str(config_path),
        quant=dtype,
        n_layer=n_layer,
        n_expert=n_expert,
        n_expert_used=n_expert_used,
        expert_bytes=expert_bytes,
        total_expert_bytes=None if expert_bytes is None else n_layer * n_expert * expert_bytes,
    )


def _gguf_value(reader: gguf.GGUFReader, key: str, path: Path) -> object:
    field = reader.fields.get(key)
    if field is None:
        raise ValueError(f"{path}: GGUF is missing required key {key}")
    return field.contents()


def _gguf_int(reader: gguf.GGUFReader, key: str, path: Path) -> int:
    value = _gguf_value(reader, key, path)
    if not isinstance(value, int):
        raise ValueError(f"{path}: GGUF key {key} must be an integer, got {value!r}")
    return value


def _gguf_quant(reader: gguf.GGUFReader) -> str | None:
    """The declared file type, e.g. MOSTLY_Q4_K_M reported as Q4_K_M. Unknown or absent is None."""
    field = reader.fields.get(gguf.Keys.General.FILE_TYPE)
    if field is None:
        return None
    try:
        return gguf.LlamaFileType(field.contents()).name.removeprefix("MOSTLY_")
    except ValueError:
        return None


def read_gguf_meta(model_path: Path) -> ModelMeta:
    """Read a GGUF's architecture facts and measure expert bytes from the tensors themselves."""
    model_path = _require_file(model_path, "GGUF model")
    reader = gguf.GGUFReader(model_path)

    arch = _gguf_value(reader, gguf.Keys.General.ARCHITECTURE, model_path)
    n_layer = _gguf_int(reader, f"{arch}.block_count", model_path)
    n_expert = _gguf_int(reader, f"{arch}.expert_count", model_path)
    n_expert_used = _gguf_int(reader, f"{arch}.expert_used_count", model_path)

    # Validate before dividing by these, so a bogus header is a named error not a ZeroDivisionError.
    _check_positive("n_layer", n_layer)
    _check_positive("n_expert", n_expert)
    _check_positive("n_expert_used", n_expert_used)

    tensors = {tensor.name: tensor for tensor in reader.tensors}
    per_layer_bytes = []
    for layer in range(n_layer):
        layer_bytes = 0
        for part in _EXPERT_TENSOR_PARTS:
            name = f"blk.{layer}.ffn_{part}_exps.weight"
            tensor = tensors.get(name)
            if tensor is None:
                raise ValueError(f"{model_path}: expert tensor {name} is missing")
            if tensor.n_bytes % n_expert:
                raise ValueError(
                    f"{model_path}: expert tensor {name} holds {tensor.n_bytes} bytes, "
                    f"not divisible by expert_count {n_expert}"
                )
            layer_bytes += tensor.n_bytes // n_expert
        per_layer_bytes.append(layer_bytes)

    if len(set(per_layer_bytes)) != 1:
        raise ValueError(
            f"{model_path}: per-expert bytes differ across layers: {per_layer_bytes}"
        )
    expert_bytes = int(per_layer_bytes[0])

    return ModelMeta(
        model=str(arch),
        path=str(model_path),
        quant=_gguf_quant(reader),
        n_layer=n_layer,
        n_expert=n_expert,
        n_expert_used=n_expert_used,
        expert_bytes=expert_bytes,
        total_expert_bytes=n_layer * n_expert * expert_bytes,
    )


_BACKEND_READERS = {"hf": read_hf_meta, "llama-cpp": read_gguf_meta}


def read_meta(path: Path, backend: str) -> ModelMeta:
    """Dispatch to the reader for `backend`; an unknown backend is a loud error."""
    reader = _BACKEND_READERS.get(backend)
    if reader is None:
        raise ValueError(
            f"backend must be one of {sorted(_BACKEND_READERS)}, got {backend!r}"
        )
    return reader(Path(path))
