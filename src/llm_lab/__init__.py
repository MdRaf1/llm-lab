"""
LLM-LAB: low-memory LLM inference research.

Most subcommands here are SIMULATIONS inherited from the project's first iteration and
are labelled as such. The current plan and the only measured work live in
docs/superpowers/specs/2026-09-11-moe-exact-runtime-design.md.
"""

import sys
import json
import argparse
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()


def _parse_token_ids(text: str) -> list[int]:
    """Parse a comma-separated CSV of integer token IDs into a flat list."""
    return [int(part) for part in text.split(",") if part.strip() != ""]


def _emit(payload: dict, output: str) -> None:
    """Write canonical JSON to `output` and print the identical text, so a tee preserves it."""
    text = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)
    Path(output).write_text(text + "\n", encoding="utf-8")
    print(text)


def _add_moe_subparsers(subparsers) -> None:
    """The five deterministic MoE reference subcommands. Help text stays free of claims."""
    moe = subparsers.add_parser("moe", help="Deterministic MoE reference pipeline")
    moe_sub = moe.add_subparsers(dest="moe_command", required=True)

    meta_p = moe_sub.add_parser("meta", help="Read checkpoint architecture facts to JSON")
    meta_p.add_argument("--backend", choices=["hf", "llama-cpp"], required=True)
    meta_p.add_argument("--model-path", required=True)
    meta_p.add_argument("--output", required=True)
    meta_p.add_argument("--force", action="store_true")

    run_p = moe_sub.add_parser("run", help="Greedy decode to a run record")
    run_p.add_argument("--backend", choices=["tiny", "hf-int8", "llama-cpp"], required=True)
    run_p.add_argument("--model-path", default=None)
    run_p.add_argument("--model-id", default=None)
    run_p.add_argument("--revision", default=None)
    run_p.add_argument("--prompt", default=None)
    run_p.add_argument("--prompt-token-ids", default=None)
    run_p.add_argument("--seed", type=int, default=0)
    run_p.add_argument("--threads", type=int, default=1)
    run_p.add_argument("--max-new-tokens", type=int, default=16)
    run_p.add_argument("--n-ctx", type=int, default=None)
    run_p.add_argument("--output", required=True)
    run_p.add_argument("--force", action="store_true")

    trace_p = moe_sub.add_parser("trace", help="Greedy decode recording per-token expert routing")
    trace_p.add_argument("--backend", choices=["tiny", "hf-int8", "hf-bf16"], required=True)
    trace_p.add_argument("--model-id", default=None)
    trace_p.add_argument("--revision", default=None)
    trace_p.add_argument("--prompt", default=None)
    trace_p.add_argument("--prompt-token-ids", default=None)
    trace_p.add_argument("--seed", type=int, default=0)
    trace_p.add_argument("--threads", type=int, default=1)
    trace_p.add_argument("--max-new-tokens", type=int, default=16)
    trace_p.add_argument("--continue-after-eos", action="store_true")
    trace_p.add_argument("--trace", required=True)
    trace_p.add_argument("--logits", required=True)
    trace_p.add_argument("--run-record", required=True)
    trace_p.add_argument("--summary", required=True)
    trace_p.add_argument("--force", action="store_true")

    compare_p = moe_sub.add_parser("compare", help="Compare two run records by token ID")
    compare_p.add_argument("run_a")
    compare_p.add_argument("run_b")
    compare_p.add_argument("--output", required=True)
    compare_p.add_argument("--force", action="store_true")

    replay_p = moe_sub.add_parser("replay-cache", help="Replay expert-cache hits over a trace")
    replay_p.add_argument("trace")
    replay_p.add_argument("--logits", required=True)
    replay_p.add_argument("--capacity-experts", type=int, required=True)
    replay_p.add_argument("--output", required=True)
    replay_p.add_argument("--force", action="store_true")

    analyze_p = moe_sub.add_parser("analyze", help="Derive cache/reuse/co-activation curves and the tier gate")
    analyze_p.add_argument("--trace", action="append", required=True)
    analyze_p.add_argument("--logits", action="append", required=True)
    analyze_p.add_argument("--geometry", required=True)
    analyze_p.add_argument("--output", required=True)
    analyze_p.add_argument("--fractions", default="0.02,0.05,0.1,0.15,0.2,0.3,0.5,0.75,1.0")
    analyze_p.add_argument("--windows", default="1,2,4,8,16,32,64,128,256,512")
    analyze_p.add_argument("--tiers-gb", default="4,8,16,32")
    analyze_p.add_argument("--reserve-bytes", type=int, default=2_500_000_000)
    analyze_p.add_argument("--b-cold-bytes", type=int, default=2_620_000_000)
    analyze_p.add_argument("--force", action="store_true")


def _reject(parser, message: str) -> None:
    """Argparse-style rejection: usage + message on stderr, SystemExit(2)."""
    parser.error(message)


def _moe_output_paths(args) -> list[str]:
    """Every path a validated `moe` subcommand will write. trace writes four; the rest one."""
    if args.moe_command == "trace":
        return [args.trace, args.logits, args.run_record, args.summary]
    return [args.output]


def _guard_outputs(parser, args) -> None:
    """Refuse to clobber existing measurements. --force is the explicit, transcript-visible override."""
    if getattr(args, "force", False):
        return
    for path in _moe_output_paths(args):
        if Path(path).exists():
            _reject(parser, f"output path exists: {path} (use --force to overwrite)")


def _validate_moe_args(parser, args) -> None:
    """The exact per-backend argument matrix. Rejections are SystemExit(2), not silent coercions."""
    if args.moe_command == "run":
        if args.backend == "tiny":
            if not args.prompt_token_ids:
                _reject(parser, "tiny requires --prompt-token-ids")
            for name, val in (("--prompt", args.prompt), ("--model-path", args.model_path),
                              ("--model-id", args.model_id), ("--revision", args.revision),
                              ("--n-ctx", args.n_ctx)):
                if val is not None:
                    _reject(parser, f"tiny rejects {name}")
        elif args.backend == "hf-int8":
            if not args.prompt:
                _reject(parser, "hf-int8 requires --prompt")
            if args.prompt_token_ids is not None:
                _reject(parser, "hf-int8 rejects --prompt-token-ids")
            if not args.model_id:
                _reject(parser, "hf-int8 requires --model-id")
            if not args.revision or len(args.revision) != 40:
                _reject(parser, "hf-int8 requires a 40-character --revision")
            if args.model_path is not None:
                _reject(parser, "hf-int8 rejects --model-path")
            if args.n_ctx is not None:
                _reject(parser, "--n-ctx is llama-cpp-only")
        elif args.backend == "llama-cpp":
            if not args.prompt:
                _reject(parser, "llama-cpp requires --prompt")
            if args.prompt_token_ids is not None:
                _reject(parser, "llama-cpp rejects --prompt-token-ids")
            if not args.model_path:
                _reject(parser, "llama-cpp requires --model-path")
            if args.model_id is not None or args.revision is not None:
                _reject(parser, "llama-cpp rejects --model-id/--revision")
    elif args.moe_command == "trace":
        if args.backend == "tiny":
            if not args.prompt_token_ids:
                _reject(parser, "tiny requires --prompt-token-ids")
            for name, val in (("--prompt", args.prompt), ("--model-id", args.model_id),
                              ("--revision", args.revision)):
                if val is not None:
                    _reject(parser, f"tiny rejects {name}")
        else:  # hf-int8, hf-bf16
            if not args.prompt:
                _reject(parser, f"{args.backend} requires --prompt")
            if args.prompt_token_ids is not None:
                _reject(parser, f"{args.backend} rejects --prompt-token-ids")
            if not args.model_id:
                _reject(parser, f"{args.backend} requires --model-id")
            if not args.revision or len(args.revision) != 40:
                _reject(parser, f"{args.backend} requires a 40-character --revision")


def _run_moe(parser, args) -> None:
    """Dispatch a validated `moe` subcommand. Heavy imports stay local to the branch that needs them."""
    _validate_moe_args(parser, args)
    _guard_outputs(parser, args)

    if args.moe_command == "meta":
        from llm_lab.moe.meta import read_meta
        _emit(read_meta(Path(args.model_path), args.backend).to_dict(), args.output)
        return

    if args.moe_command == "compare":
        from llm_lab.moe.baseline import RunRecord
        from llm_lab.moe.oracle import compare_runs
        result = compare_runs(RunRecord.read(Path(args.run_a)), RunRecord.read(Path(args.run_b)))
        _emit(result, args.output)
        return

    if args.moe_command == "replay-cache":
        # Ruling A + B: read_trace re-validates the steps (replay_cache can't), sha256_file turns
        # the --logits PATH into the digest replay_cache echoes as its capacity-independent proof.
        from llm_lab.moe.baseline import sha256_file
        from llm_lab.moe.trace import read_trace, replay_cache
        _, steps = read_trace(Path(args.trace))
        result = replay_cache(steps, args.capacity_experts, sha256_file(Path(args.logits)))
        _emit(result, args.output)
        return

    if args.moe_command == "analyze":
        import json as _json
        from llm_lab.moe.analysis import analyze_traces
        from llm_lab.moe.baseline import sha256_file
        from llm_lab.moe.trace import read_trace
        if len(args.trace) != len(args.logits):
            _reject(parser, "analyze needs one --logits per --trace")
        traces = []
        for tpath, lpath in zip(args.trace, args.logits):
            _, steps = read_trace(Path(tpath))
            traces.append((Path(tpath).name, steps, sha256_file(Path(lpath))))
        geometry = _json.loads(Path(args.geometry).read_text(encoding="utf-8"))
        result = analyze_traces(
            traces, geometry,
            fractions=[float(x) for x in args.fractions.split(",")],
            windows=[int(x) for x in args.windows.split(",")],
            tiers_gb=[int(x) for x in args.tiers_gb.split(",")],
            reserve_bytes=args.reserve_bytes, b_cold_bytes_s=args.b_cold_bytes)
        _emit(result, args.output)
        return

    if args.moe_command == "run":
        _run_moe_run(args)
        return

    if args.moe_command == "trace":
        _run_moe_trace(args)
        return


def _run_moe_run(args) -> None:
    from llm_lab.moe.baseline import (
        RunConfig, build_tiny_olmoe, run_hf, sha256_token_ids,
    )

    if args.backend == "tiny":
        token_ids = _parse_token_ids(args.prompt_token_ids)
        config = RunConfig(
            model="tiny-olmoe", path="local-random", quant="float32",
            seed=args.seed, decode="greedy", threads=args.threads, backend="tiny",
            prompt_sha=sha256_token_ids(token_ids), tokenizer_sha=None,
        )
        record = run_hf(build_tiny_olmoe(args.seed), [token_ids], config, args.max_new_tokens)
    elif args.backend == "hf-int8":
        from llm_lab.moe.baseline import load_hf_int8, sha256_tokenizer
        model, tokenizer = load_hf_int8(args.model_id, args.revision)
        ids = tokenizer(args.prompt, return_tensors="pt").input_ids
        config = RunConfig(
            model=args.model_id, path=f"{args.model_id}@{args.revision}", quant="INT8_BITSANDBYTES",
            seed=args.seed, decode="greedy", threads=args.threads, backend="hf-int8",
            prompt_sha=sha256_token_ids(ids[0].tolist()), tokenizer_sha=sha256_tokenizer(tokenizer),
        )
        record = run_hf(model, ids, config, args.max_new_tokens)
    else:  # llama-cpp
        from llm_lab.moe.baseline import run_llama_cpp, sha256_file
        from llm_lab.moe.meta import read_gguf_meta
        meta = read_gguf_meta(Path(args.model_path))  # missing file -> FileNotFoundError naming path
        # run_llama_cpp recomputes prompt_sha from the real tokenized IDs; the [0] here is a
        # placeholder it overwrites. tokenizer identity for a GGUF is the model file itself.
        config = RunConfig(
            model=meta.model, path=meta.path, quant=meta.quant,
            seed=args.seed, decode="greedy", threads=args.threads, backend="llama-cpp",
            prompt_sha=sha256_token_ids([0]), tokenizer_sha=sha256_file(Path(args.model_path)),
        )
        record = run_llama_cpp(
            Path(args.model_path), args.prompt, config, args.max_new_tokens,
            args.n_ctx if args.n_ctx is not None else 512,
        )
    _emit(record.to_dict(), args.output)


def _run_moe_trace(args) -> None:
    from llm_lab.moe.baseline import RunConfig, build_tiny_olmoe, sha256_token_ids
    from llm_lab.moe.trace import TraceHeader, trace_hf
    from llm_lab.moe.baseline import host_info

    stop_at_eos = not args.continue_after_eos

    if args.backend == "tiny":
        model = build_tiny_olmoe(args.seed)
        cfg = model.config
        token_ids = _parse_token_ids(args.prompt_token_ids)
        header = TraceHeader(
            kind="header", model="tiny-olmoe", path="local-random", quant="float32",
            n_layer=cfg.num_hidden_layers, n_expert=cfg.num_experts,
            n_expert_used=cfg.num_experts_per_tok, expert_bytes=None, total_expert_bytes=None,
            tokenizer_sha=None, prompt_sha=sha256_token_ids(token_ids), seed=args.seed,
            decode="greedy", backend="tiny", threads=args.threads, host=host_info(),
        )
        input_ids = [token_ids]
        precision = "FP32"
        quant = "float32"
    else:
        from llm_lab.moe.baseline import load_hf_int8, load_hf_bf16, sha256_tokenizer
        if args.backend == "hf-int8":
            model, tokenizer = load_hf_int8(args.model_id, args.revision)
            precision = "INT8_BITSANDBYTES"
        else:
            model, tokenizer = load_hf_bf16(args.model_id, args.revision)
            precision = "BF16"
        cfg = model.config
        input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
        quant = str(getattr(cfg, "torch_dtype", None))
        header = TraceHeader(
            kind="header", model=args.model_id, path=f"{args.model_id}@{args.revision}", quant=quant,
            n_layer=cfg.num_hidden_layers, n_expert=cfg.num_experts,
            n_expert_used=cfg.num_experts_per_tok, expert_bytes=None, total_expert_bytes=None,
            tokenizer_sha=sha256_tokenizer(tokenizer),
            prompt_sha=sha256_token_ids(input_ids[0].tolist()), seed=args.seed,
            decode="greedy", backend=args.backend, threads=args.threads, host=host_info(),
        )

    capture = trace_hf(
        model, input_ids, header, args.max_new_tokens,
        Path(args.trace), Path(args.logits), stop_at_eos=stop_at_eos,
    )
    capture.run_record.write(Path(args.run_record))

    record = capture.run_record
    summary = {
        "model": header.model, "path": header.path, "quant": quant, "backend": header.backend,
        "precision": precision, "n_layer": header.n_layer, "n_expert": header.n_expert,
        "n_expert_used": header.n_expert_used, "n_prompt": record.n_prompt,
        "n_generated": record.n_generated, "n_steps": record.n_generated,
        "stop_at_eos": capture.stop_at_eos, "seed": header.seed, "decode": header.decode,
        "threads": header.threads, "prompt_sha": header.prompt_sha,
        "tokenizer_sha": header.tokenizer_sha, "trace_sha256": capture.trace_sha256,
        "logits_sha256": capture.logits_sha256, "trace_path": str(args.trace),
        "logits_path": str(args.logits), "host": header.host,
    }
    _emit(summary, args.summary)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="LLM-LAB: low-memory LLM inference research (see docs/superpowers/specs/)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")
    
    # 1. Profile command
    subparsers.add_parser("profile", help="Run empirical hardware bandwidth & speedup ceiling profiling")
    
    # 2. Test command
    subparsers.add_parser("test", help="Run validation tests for KV cache, speculative engine, and sparse router")
    
    # 3. Simulate command (SIMULATION — no model is loaded)
    sim_parser = subparsers.add_parser("simulate", help="[SIMULATION] Random-input engine sketch; generates no real tokens")
    sim_parser.add_argument("--model", type=str, default="Llama-3.3-70B", help="Label only; no model is loaded")
    sim_parser.add_argument("--steps", type=int, default=10, help="Number of simulated steps")
    # 4. Frontier command (SIMULATION)
    front_parser = subparsers.add_parser("frontier", help="[SIMULATION] Frontier 70B sketch; reports no measured inference")
    front_parser.add_argument("--tokens", type=int, default=20, help="Number of simulated tokens")
    
    # 5. Convert command (SIMULATION — synthetic calibration)
    conv_parser = subparsers.add_parser("convert", help="[SIMULATION] Synthetic hot/cold partition; for real GGUF use frontier/gguf_partitioner.py")
    conv_parser.add_argument("--output-dir", type=str, default="models/llama-70b-partitioned", help="Output directory")
    conv_parser.add_argument("--model-name", type=str, default="Llama-3.3-70B-Frontier", help="Model name")

    # 6. Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive terminal chat (only --engine local-llama runs a real model)")
    chat_parser.add_argument("--engine", choices=["local-llama", "sim-frontier-70b", "sim-partitioned"], default="local-llama", help="Engine backend. Only local-llama runs a real model.")
    chat_parser.add_argument("--model", type=str, default=None, help="Display label for the sim engines; ignored by local-llama")

    _add_moe_subparsers(subparsers)

    args = parser.parse_args(argv)

    if args.command == "moe":
        try:
            _run_moe(parser, args)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(1)
        return

    if args.command == "profile" or args.command is None:
        from llm_lab.profiler import run_profiler
        run_profiler()
    elif args.command == "test":
        console.print(Panel.fit("[bold green]Running LLM-LAB Validation Test Suite...[/bold green]"))
        import subprocess
        python_exe = sys.executable
        test_files = [
            "tests/test_kv_cache.py",
            "tests/test_speculative.py",
            "tests/test_sparse_router.py",
            "tests/test_kernels.py",
            "tests/test_layers.py",
            "tests/test_self_drafter.py",
            "tests/test_engine.py",
            "tests/test_converter.py",
            "tests/test_gguf_stream.py",
            "tests/test_chat.py",
        ]
        for test_file in test_files:
            console.print(f"\n[bold cyan]>> Executing {test_file}...[/bold cyan]")
            res = subprocess.run([python_exe, test_file])
            if res.returncode != 0:
                console.print(f"[bold red]Failed: {test_file}[/bold red]")
                sys.exit(res.returncode)
        console.print("\n[bold green]All Validation Tests Completed Successfully![/bold green]")
    elif args.command == "simulate":
        from llm_lab.core.sim_engine import BreakthroughEngine
        console.print(Panel.fit(
            "[bold red]SIMULATION - NO MODEL IS LOADED[/bold red]\n"
            f"[dim]Shape label: {args.model}. Hidden states are np.random.randn and tokens are\n"
            "np.random.randint. The figures below describe numpy, not a language model.\n"
            "Do not quote them.[/dim]"
        ))
        engine = BreakthroughEngine(
            model_name=f"{args.model}-Breakthrough",
            total_params=70_000_000_000,
            num_layers=80,
            d_model=8192,
            d_intermediate=28672,
            num_kv_heads=8,
            head_dim=128,
            quant_bits=4,
            draft_length=4,
            hot_neuron_ratio=0.20,
        )
        results = engine.benchmark_step(list(range(128)), steps=args.steps)
        console.print(f"[SIMULATED] {results['total_tokens_generated']} random tokens in {results['elapsed_seconds']:.2f}s ({results['effective_tokens_per_second']:.1f} tok/s of numpy)")
        console.print(f"[CONFIG CONSTANT, not measured] skip ratio {results['avg_sparsity_skipped_pct']:.1f}% | KV footprint {results['kv_cache_footprint_mb']:.2f} MB")
    elif args.command == "frontier":
        from llm_lab.frontier.sim_stream_engine import Frontier70BEngine
        console.print(Panel.fit(
            "[bold red]SIMULATION - NO MODEL IS LOADED[/bold red]\n"
            "[dim]Layer buffers stay zero-filled, no I/O is performed, and the 'speedup' is\n"
            "computed against a hardcoded 0.08 tok/s. Do not quote any of it.[/dim]"
        ))
        engine = Frontier70BEngine()
        engine.benchmark_70b_generation(prompt_tokens=list(range(64)), max_tokens=args.tokens)
    elif args.command == "convert":
        from llm_lab.frontier.sim_converter import ModelPartitioner
        console.print(Panel.fit(
            "[bold red]SIMULATION - calibration is np.random.zipf, not a model run[/bold red]\n"
            "[dim]For real GGUF partitioning use frontier/gguf_partitioner.py.[/dim]"
        ))
        partitioner = ModelPartitioner()
        partitioner.partition_and_export(output_dir=args.output_dir, model_name=args.model_name)
    elif args.command == "chat":
        from llm_lab.cli.chat import InteractiveChatSession
        session = InteractiveChatSession(mode=args.engine, model_name=args.model)
        session.run_repl()

if __name__ == "__main__":
    main()
