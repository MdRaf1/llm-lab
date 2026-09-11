"""
LLM-LAB: low-memory LLM inference research.

Most subcommands here are SIMULATIONS inherited from the project's first iteration and
are labelled as such. The current plan and the only measured work live in
docs/superpowers/specs/2026-09-11-moe-exact-runtime-design.md.
"""

import sys
import argparse
from rich.console import Console
from rich.panel import Panel

console = Console()

def main():
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
    
    args = parser.parse_args()
    
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
