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

import hashlib
import json
import platform
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import psutil


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
