"""
Run records and stable configuration fingerprints.
Guarantees:
1. Record fields cannot be rebound after construction. Fields are not deep-frozen: the `host`
   dict a caller passes in stays mutable through its own reference.
2. A run is described only by measured values; anything unmeasured stays None.
3. Two runs are comparable only when their config fingerprints are identical.
4. Records are written atomically, so an interrupted run leaves no partial record.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import platform
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import psutil
import torch

# Module scope so a test can monkeypatch `baseline.Llama` with a fake and `run_llama_cpp` picks it
# up by name. Only imported here (not by the simulation CLI), so the native lib loads solely when
# the MoE pipeline runs.
from llama_cpp import Llama


def sha256_bytes(data: bytes) -> str:
    """SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """SHA-256 hex digest of a file's contents, streamed so large weights fit in RAM."""
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def sha256_token_ids(token_ids: Sequence[int]) -> str:
    """SHA-256 hex digest of token IDs serialized as canonical compact JSON."""
    payload = json.dumps([int(t) for t in token_ids], separators=(",", ":"))
    return sha256_bytes(payload.encode("utf-8"))


def host_info() -> dict[str, object]:
    """Measured host facts only; nothing derived, nothing estimated."""
    return {
        "processor": platform.processor(),
        "total_ram_bytes": psutil.virtual_memory().total,
        "platform": platform.platform(),
    }


def _check_sha(name: str, value: str | None, *, optional: bool = False) -> None:
    if value is None:
        if optional:
            return
        raise ValueError(f"{name} is required")
    if len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256 hex digest, got {value!r}")


def _check_non_negative(name: str, value: float | None, *, optional: bool = False) -> None:
    if value is None:
        if optional:
            return
        raise ValueError(f"{name} is required")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")


@dataclass(frozen=True)
class RunConfig:
    """Everything that must match for two runs to be comparable."""

    model: str
    path: str
    quant: str | None
    seed: int
    decode: str
    threads: int
    backend: str
    prompt_sha: str
    tokenizer_sha: str | None

    def __post_init__(self) -> None:
        if self.decode != "greedy":
            raise ValueError(f"decode must be 'greedy', got {self.decode!r}")
        _check_non_negative("seed", self.seed)
        _check_non_negative("threads", self.threads)
        _check_sha("prompt_sha", self.prompt_sha)
        _check_sha("tokenizer_sha", self.tokenizer_sha, optional=True)

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "path": self.path,
            "quant": self.quant,
            "seed": self.seed,
            "decode": self.decode,
            "threads": self.threads,
            "backend": self.backend,
            "prompt_sha": self.prompt_sha,
            "tokenizer_sha": self.tokenizer_sha,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> RunConfig:
        return cls(
            model=data["model"],
            path=data["path"],
            quant=data["quant"],
            seed=data["seed"],
            decode=data["decode"],
            threads=data["threads"],
            backend=data["backend"],
            prompt_sha=data["prompt_sha"],
            tokenizer_sha=data["tokenizer_sha"],
        )


def fingerprint_config(config: RunConfig) -> str:
    """Stable SHA-256 over the canonical compact JSON of a config."""
    payload = json.dumps(
        config.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class RunRecord:
    """One completed run: its fingerprint, its generated token IDs, and measured metrics."""

    config_fingerprint: str
    token_ids: tuple[int, ...]
    n_prompt: int
    n_generated: int
    wall_s: float | None
    tok_per_s: float | None
    peak_rss_bytes: int | None
    host: dict[str, object]

    def __post_init__(self) -> None:
        # Normalize before validating: callers hand us whatever their backend produced (HF lists,
        # numpy/torch scalars), and a stored record must always round-trip as a tuple of ints.
        object.__setattr__(self, "token_ids", tuple(int(t) for t in self.token_ids))
        _check_sha("config_fingerprint", self.config_fingerprint)
        _check_non_negative("n_prompt", self.n_prompt)
        _check_non_negative("n_generated", self.n_generated)
        _check_non_negative("wall_s", self.wall_s, optional=True)
        _check_non_negative("tok_per_s", self.tok_per_s, optional=True)
        _check_non_negative("peak_rss_bytes", self.peak_rss_bytes, optional=True)

    def to_dict(self) -> dict[str, object]:
        return {
            "config_fingerprint": self.config_fingerprint,
            "token_ids": list(self.token_ids),
            "n_prompt": self.n_prompt,
            "n_generated": self.n_generated,
            "wall_s": self.wall_s,
            "tok_per_s": self.tok_per_s,
            "peak_rss_bytes": self.peak_rss_bytes,
            "host": self.host,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> RunRecord:
        return cls(
            config_fingerprint=data["config_fingerprint"],
            token_ids=tuple(data["token_ids"]),
            n_prompt=data["n_prompt"],
            n_generated=data["n_generated"],
            wall_s=data["wall_s"],
            tok_per_s=data["tok_per_s"],
            peak_rss_bytes=data["peak_rss_bytes"],
            host=data["host"],
        )

    @classmethod
    def read(cls, path: Path) -> RunRecord:
        """Load a record; a missing or malformed file is a loud error."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def write(self, path: Path) -> None:
        """Write atomically via a sibling temp file, so no partial record is ever plausible."""
        path = Path(path)
        payload = json.dumps(self.to_dict(), sort_keys=True, indent=2, ensure_ascii=False)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(payload + "\n", encoding="utf-8")
        tmp.replace(path)


def seed_everything(seed: int, threads: int) -> None:
    """Force every knob that decides determinism on this torch build.

    Seeds the RNG so weight init and any sampling are reproducible, pins the thread count so op
    scheduling cannot reorder floating-point reductions, and turns on deterministic algorithm
    selection so a nondeterministic kernel raises instead of silently averaging away.
    """
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def peak_rss_bytes() -> int | None:
    """Peak resident set from psutil, or None on a platform with no peak counter."""
    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None)  # Windows-only field.
    return int(peak) if peak is not None else None


@contextlib.contextmanager
def _oom_as_runtime_error(what: str):
    """Convert an allocator/CUDA OOM into a clear failure; never a silent fallback."""
    try:
        yield
    except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover - CPU box never hits this
        raise RuntimeError(f"OOM while loading/running {what}; no fallback attempted") from exc
    except MemoryError as exc:
        raise RuntimeError(f"OOM while loading/running {what}; no fallback attempted") from exc


def build_tiny_olmoe(seed: int):
    """A locally-constructed random-weight tiny OLMoE. No download, no tokenizer, no network."""
    from transformers import OlmoeConfig, OlmoeForCausalLM

    seed_everything(seed, 1)
    config = OlmoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=127,
        pad_token_id=0,
    )
    return OlmoeForCausalLM(config).eval()


def _as_input_ids(input_ids) -> torch.Tensor:
    """Coerce a nested int sequence or tensor into a 2-D long tensor."""
    if isinstance(input_ids, torch.Tensor):
        return input_ids.to(torch.long)
    return torch.tensor(input_ids, dtype=torch.long)


def run_hf(model, input_ids, config: RunConfig, max_new_tokens: int) -> RunRecord:
    """Deterministic greedy generation producing a RunRecord; measured values only."""
    seed_everything(config.seed, config.threads)
    model.eval()
    ids = _as_input_ids(input_ids)
    n_prompt = int(ids.shape[-1])

    start = time.perf_counter()
    with torch.no_grad(), _oom_as_runtime_error(config.model):
        generated = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    wall_s = time.perf_counter() - start

    new_tokens = generated[0, n_prompt:].tolist()
    n_generated = len(new_tokens)
    return RunRecord(
        config_fingerprint=fingerprint_config(config),
        token_ids=tuple(new_tokens),
        n_prompt=n_prompt,
        n_generated=n_generated,
        wall_s=wall_s,
        tok_per_s=(n_generated / wall_s) if wall_s > 0 and n_generated else None,
        peak_rss_bytes=peak_rss_bytes(),
        host=host_info(),
    )


def load_hf_int8(model_id: str, revision: str):
    """Load a pinned int8 checkpoint on CPU; fail loudly rather than mislabel precision."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    with _oom_as_runtime_error(model_id):
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map="cpu",
        )
    from bitsandbytes.nn import Linear8bitLt

    if not any(isinstance(module, Linear8bitLt) for module in model.modules()):
        raise RuntimeError(
            f"{model_id}@{revision} loaded without any Linear8bitLt module; not int8"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    return model, tokenizer


def load_hf_bf16(model_id: str, revision: str):
    """Load a pinned bf16 checkpoint on CPU; the rented-host path, no int8/local fallback."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with _oom_as_runtime_error(model_id):
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=torch.bfloat16,
            device_map="cpu",
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    return model, tokenizer


def sha256_tokenizer(tokenizer) -> str:
    """Hash the tokenizer's on-disk files in sorted path order.

    A tokenizer's identity is its files, not its in-memory object. `init_kwargs` records the paths
    the tokenizer was loaded from; we hash the ones that actually exist. If none can be identified
    (e.g. a tokenizer built in memory with no artifact), we refuse rather than invent a digest.
    """
    paths = sorted(
        str(value)
        for value in getattr(tokenizer, "init_kwargs", {}).values()
        if isinstance(value, (str, Path)) and Path(value).is_file()
    )
    if not paths:
        raise ValueError("no local tokenizer artifact found to hash from init_kwargs")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def run_llama_cpp(
    model_path: Path,
    prompt: str,
    config: RunConfig,
    max_new_tokens: int,
    n_ctx: int,
) -> RunRecord:
    """Deterministic greedy decode on a GGUF via llama.cpp; stores the exact IDs `generate` yields.

    The prompt is tokenized once and the fingerprint's `prompt_sha` is recomputed from those exact
    IDs (the caller cannot predict llama.cpp's tokenization), so the record's config is the truth
    of what ran. Only the autoregressive decode is timed; prompt tokens do not count toward
    `tok_per_s`. Stops at EOS or `max_new_tokens`, whichever comes first.
    """
    llm = Llama(
        model_path=str(model_path),
        n_gpu_layers=0,
        seed=config.seed,
        n_threads=config.threads,
        n_ctx=n_ctx,
        verbose=False,
    )
    prompt_ids = list(llm.tokenize(prompt.encode("utf-8")))
    n_prompt = len(prompt_ids)
    eos_id = llm.token_eos()

    generated: list[int] = []
    start = time.perf_counter()
    with _oom_as_runtime_error(config.model):
        for token_id in llm.generate(
            prompt_ids,
            temp=0.0,
            top_k=1,
            top_p=1.0,
            min_p=0.0,
            typical_p=1.0,
            repeat_penalty=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        ):
            if token_id == eos_id or len(generated) >= max_new_tokens:
                break
            # Store the exact integer ID the generator yielded, never an ID recovered by
            # re-tokenizing decoded text: that round-trip is the bug the oracle exists to catch.
            generated.append(int(token_id))
    wall_s = time.perf_counter() - start

    n_generated = len(generated)
    config = replace(config, prompt_sha=sha256_token_ids(prompt_ids))
    return RunRecord(
        config_fingerprint=fingerprint_config(config),
        token_ids=tuple(generated),
        n_prompt=n_prompt,
        n_generated=n_generated,
        wall_s=wall_s,
        tok_per_s=(n_generated / wall_s) if wall_s > 0 and n_generated else None,
        peak_rss_bytes=peak_rss_bytes(),
        host=host_info(),
    )
