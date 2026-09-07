"""
LLM-LAB CLI: Breakthrough Low-Memory Frontier LLM Inference Research
"""

import sys
import argparse
from rich.console import Console
from rich.panel import Panel

console = Console()

def main():
    parser = argparse.ArgumentParser(
        description="LLM-LAB: Breakthrough Low-Memory Frontier LLM Inference Research"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")
    
    # 1. Profile command
    subparsers.add_parser("profile", help="Run empirical hardware bandwidth & speedup ceiling profiling")
    
    # 2. Test command
    subparsers.add_parser("test", help="Run validation tests for KV cache, speculative engine, and sparse router")
    
    # 3. Simulate command
    sim_parser = subparsers.add_parser("simulate", help="Run end-to-end breakthrough engine simulation")
    sim_parser.add_argument("--model", type=str, default="Llama-3.3-70B", help="Target model architecture")
    # 4. Frontier command
    front_parser = subparsers.add_parser("frontier", help="Run Frontier 70B streaming engine benchmark")
    front_parser.add_argument("--tokens", type=int, default=20, help="Number of tokens to generate")
    
    # 5. Convert command
    conv_parser = subparsers.add_parser("convert", help="Partition model weights into Hot RAM and Cold NVMe streams")
    conv_parser.add_argument("--output-dir", type=str, default="models/llama-70b-partitioned", help="Output directory")
    conv_parser.add_argument("--model-name", type=str, default="Llama-3.3-70B-Frontier", help="Model name")
    
    # 6. Chat command
    chat_parser = subparsers.add_parser("chat", help="Launch interactive live terminal chat with real-time telemetry")
    chat_parser.add_argument("--model", type=str, default="Llama-3.3-70B-Frontier", help="Model name")
    
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
            "tests/test_self_drafter.py",
            "tests/test_engine.py",
            "tests/test_converter.py",
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
        from llm_lab.core.engine import BreakthroughEngine
        console.print(Panel.fit(f"[bold magenta]Simulating Breakthrough Engine on {args.model}...[/bold magenta]"))
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
        console.print(f"Generated [bold green]{results['total_tokens_generated']}[/bold green] tokens in [bold cyan]{results['elapsed_seconds']:.2f}s[/bold cyan] ([bold yellow]{results['effective_tokens_per_second']:.1f} tok/s[/bold yellow])")
        console.print(f"Sparsity Skipped: [bold green]{results['avg_sparsity_skipped_pct']:.1f}%[/bold green], KV Footprint: [bold green]{results['kv_cache_footprint_mb']:.2f} MB[/bold green]")
    elif args.command == "frontier":
        from llm_lab.frontier.stream_engine import Frontier70BEngine
        engine = Frontier70BEngine()
        engine.benchmark_70b_generation(prompt_tokens=list(range(64)), max_tokens=args.tokens)
    elif args.command == "convert":
        from llm_lab.frontier.converter import ModelPartitioner
        partitioner = ModelPartitioner()
        partitioner.partition_and_export(output_dir=args.output_dir, model_name=args.model_name)
    elif args.command == "chat":
        from llm_lab.cli.chat import InteractiveChatSession
        session = InteractiveChatSession(model_name=args.model)
        session.run_repl()

if __name__ == "__main__":
    main()
