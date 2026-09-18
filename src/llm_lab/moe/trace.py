"""
Strict JSONL routing traces, the union arithmetic over them, and an observational cache replay.
Guarantees:
1. A trace file is one compact, sorted JSON object per UTF-8 line: line 0 is the header
   (kind="header"), every other line is a step (kind="step"). Nullable fields stay present as
   JSON null; no field beyond the approved schema is ever written.
2. A step's `tok_id` is the PROCESSED token whose hidden state selected `layer_experts` for that
   position, not the next token emitted.
3. Layer identity is part of every routing fact: an expert ID is meaningless without its layer,
   so unions are encoded as (layer, expert) pairs and the cache is keyed on (layer, expert).
4. Steps are validated against the header they belong to before a file is written and again after
   it is read. A malformed trace is a loud ValueError naming the offending field, never a silent
   partial file (writes are atomic via a sibling temp + replace).
5. `replay_cache` is analysis-only. It counts hits and misses of expert blobs under a capacity but
   never executes model logits; the immutable `logits_sha256` it echoes is the proof that cache
   capacity was not an input to model execution.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from llm_lab.moe.baseline import _check_non_negative, _check_sha
from llm_lab.moe.meta import _check_positive


@dataclass(frozen=True)
class TraceHeader:
    """The §8.5 trace header: ModelMeta's shape plus the run config that produced the trace."""

    kind: str
    model: str
    path: str
    quant: str | None
    n_layer: int
    n_expert: int
    n_expert_used: int
    expert_bytes: int | None
    total_expert_bytes: int | None
    tokenizer_sha: str | None
    prompt_sha: str
    seed: int
    decode: str
    backend: str
    threads: int
    host: dict[str, object]

    def __post_init__(self) -> None:
        if self.kind != "header":
            raise ValueError(f"header kind must be 'header', got {self.kind!r}")
        _check_positive("n_layer", self.n_layer)
        _check_positive("n_expert", self.n_expert)
        _check_positive("n_expert_used", self.n_expert_used)
        _check_non_negative("expert_bytes", self.expert_bytes, optional=True)
        _check_non_negative("total_expert_bytes", self.total_expert_bytes, optional=True)
        _check_non_negative("seed", self.seed)
        _check_non_negative("threads", self.threads)
        _check_sha("prompt_sha", self.prompt_sha)
        _check_sha("tokenizer_sha", self.tokenizer_sha, optional=True)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "model": self.model,
            "path": self.path,
            "quant": self.quant,
            "n_layer": self.n_layer,
            "n_expert": self.n_expert,
            "n_expert_used": self.n_expert_used,
            "expert_bytes": self.expert_bytes,
            "total_expert_bytes": self.total_expert_bytes,
            "tokenizer_sha": self.tokenizer_sha,
            "prompt_sha": self.prompt_sha,
            "seed": self.seed,
            "decode": self.decode,
            "backend": self.backend,
            "threads": self.threads,
            "host": self.host,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> TraceHeader:
        return cls(
            kind=data["kind"],
            model=data["model"],
            path=data["path"],
            quant=data["quant"],
            n_layer=data["n_layer"],
            n_expert=data["n_expert"],
            n_expert_used=data["n_expert_used"],
            expert_bytes=data["expert_bytes"],
            total_expert_bytes=data["total_expert_bytes"],
            tokenizer_sha=data["tokenizer_sha"],
            prompt_sha=data["prompt_sha"],
            seed=data["seed"],
            decode=data["decode"],
            backend=data["backend"],
            threads=data["threads"],
            host=data["host"],
        )


@dataclass(frozen=True)
class TraceStep:
    """One decoded position: the processed token and the experts each layer selected for it."""

    kind: str
    pos: int
    tok_id: int
    layer_experts: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if self.kind != "step":
            raise ValueError(f"step kind must be 'step', got {self.kind!r}")
        # Normalize before anything else: JSON round-trips give lists, callers give lists/scalars,
        # and a stored step must always compare and re-serialize as nested tuples of ints.
        object.__setattr__(self, "pos", int(self.pos))
        object.__setattr__(self, "tok_id", int(self.tok_id))
        object.__setattr__(
            self,
            "layer_experts",
            tuple(tuple(int(e) for e in layer) for layer in self.layer_experts),
        )
        _check_non_negative("pos", self.pos)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "pos": self.pos,
            "tok_id": self.tok_id,
            "layer_experts": [list(layer) for layer in self.layer_experts],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> TraceStep:
        return cls(
            kind=data["kind"],
            pos=data["pos"],
            tok_id=data["tok_id"],
            layer_experts=data["layer_experts"],
        )


def _validate_steps(header: TraceHeader, steps: Sequence[TraceStep]) -> None:
    """Every reject condition lives here, so write and read enforce the identical contract."""
    for expected_pos, step in enumerate(steps):
        if step.pos != expected_pos:
            raise ValueError(
                f"pos must be contiguous from 0; expected {expected_pos}, got {step.pos}"
            )
        if len(step.layer_experts) != header.n_layer:
            raise ValueError(
                f"step pos {step.pos} has {len(step.layer_experts)} layers, "
                f"header declares n_layer {header.n_layer}"
            )
        for layer, experts in enumerate(step.layer_experts):
            if len(experts) != header.n_expert_used:
                raise ValueError(
                    f"step pos {step.pos} layer {layer} has top-k {len(experts)}, "
                    f"header declares n_expert_used {header.n_expert_used}"
                )
            if len(set(experts)) != len(experts):
                raise ValueError(
                    f"step pos {step.pos} layer {layer} has duplicate expert IDs: {experts}"
                )
            for expert in experts:
                if not 0 <= expert < header.n_expert:
                    raise ValueError(
                        f"step pos {step.pos} layer {layer} expert {expert} is outside "
                        f"[0, n_expert={header.n_expert})"
                    )


def write_trace(path: Path, header: TraceHeader, steps: Iterable[TraceStep]) -> None:
    """Validate, then atomically write the header line followed by one line per step."""
    steps = list(steps)
    _validate_steps(header, steps)
    path = Path(path)
    lines = [json.dumps(header.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)]
    lines += [
        json.dumps(step.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for step in steps
    ]
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_trace(path: Path) -> tuple[TraceHeader, list[TraceStep]]:
    """Read a trace and re-validate it against its own header; a malformed file is a loud error."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"{path}: empty trace, no header line")
    header = TraceHeader.from_dict(json.loads(lines[0]))
    steps = [TraceStep.from_dict(json.loads(line)) for line in lines[1:]]
    _validate_steps(header, steps)
    return header, steps


def _requests(steps: Sequence[TraceStep]) -> list[tuple[int, int]]:
    """Every (layer, expert) routing fact in router order: step by step, layer by layer."""
    return [
        (layer, expert)
        for step in steps
        for layer, experts in enumerate(step.layer_experts)
        for expert in experts
    ]


def window_unions(steps: Sequence[TraceStep], window: int) -> list[tuple[int, ...]]:
    """Sorted (layer, expert) pairs touched in each sliding window of `window` consecutive steps."""
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    unions = []
    for start in range(len(steps) - window + 1):
        pairs = {
            (layer, expert)
            for step in steps[start : start + window]
            for layer, experts in enumerate(step.layer_experts)
            for expert in experts
        }
        unions.append(tuple(sorted(pairs)))
    return unions


def _trace_sha256(steps: Sequence[TraceStep]) -> str:
    """Hash of the canonical step payload alone, so it never depends on cache capacity."""
    payload = json.dumps(
        [step.to_dict() for step in steps], sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def replay_cache(
    steps: Sequence[TraceStep], capacity_experts: int, logits_sha256: str
) -> dict[str, object]:
    """Count expert-blob cache hits/misses under an LRU of `capacity_experts` blobs.

    Analysis only: this never executes or claims to execute model logits. `logits_sha256` is
    echoed back unchanged as proof that cache capacity was not an input to model execution, so
    two capacities over the same trace must agree on trace_sha256, logits_sha256, and token_ids.
    """
    if capacity_experts < 0:
        raise ValueError(f"capacity_experts must be non-negative, got {capacity_experts}")

    cache: OrderedDict[tuple[int, int], None] = OrderedDict()
    hits = misses = 0
    for key in _requests(steps):
        if key in cache:
            hits += 1
            cache.move_to_end(key)
            continue
        misses += 1
        cache[key] = None
        while len(cache) > capacity_experts:
            cache.popitem(last=False)

    return {
        "requests": hits + misses,
        "hits": hits,
        "misses": misses,
        "capacity_experts": capacity_experts,
        "trace_sha256": _trace_sha256(steps),
        "logits_sha256": logits_sha256,
        "token_ids": [step.tok_id for step in steps],
    }
