"""
Interactive terminal chat for LLM-LAB.

Only `--engine local-llama` generates real text (llama.cpp + a real GGUF file). The
`sim-*` engines are simulations with no model attached and refuse to generate; see
docs/superpowers/specs/2026-09-11-moe-exact-runtime-design.md section 2.
"""

import sys
import time
import psutil
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import argparse
from typing import Optional
from llm_lab.frontier.sim_stream_engine import Frontier70BEngine
from llm_lab.frontier.sim_gguf_stream_engine import PartitionedGGUFStreamEngine

console = Console()

class InteractiveChatSession:
    """
    Interactive chat session. Only one mode generates real text:

    1. local-llama        — REAL. Runs a GGUF model via llama.cpp.
    2. sim-frontier-70b   — SIMULATION. No model; refuses to generate. See
                            frontier/sim_stream_engine.py.
    3. sim-partitioned    — SIMULATION. Mmaps real partition files but never computes
                            with them. See frontier/sim_gguf_stream_engine.py.
    """
    def __init__(self, mode: str = "local-llama", partition_dir: str = "models/llama-8b-real-partition", model_name: Optional[str] = None):
        self.mode = mode
        self.max_tokens = 50
        self.conversation_history = []
        self.total_tokens_session = 0
        self.llm_real = None

        if self.mode == "sim-partitioned":
            self.model_name = "SIMULATION (partitioned GGUF, no compute)"
            self.engine = PartitionedGGUFStreamEngine(partition_dir=partition_dir)
        elif self.mode == "local-llama":
            self.model_name = "LLaMA-3-8B (llama.cpp, prompt-lookup draft)"
            self.engine = None
            from llama_cpp import Llama
            from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
            default_gguf = r"C:\Users\Md. Rafi\.ollama\models\blobs\sha256-9df3a2da16e6869d6976fe8860ced76046f6e0e689a2fedf3b34a4e73a665af2"
            self.llm_real = Llama(
                model_path=default_gguf,
                draft_model=LlamaPromptLookupDecoding(max_ngram_size=3, num_pred_tokens=4),
                n_ctx=2048,
                n_threads=6,
                verbose=False
            )
        elif self.mode == "sim-frontier-70b":
            self.model_name = "SIMULATION (no model attached)"
            self.engine = Frontier70BEngine(model_name=model_name or "sim-70b")
        else:
            raise ValueError(
                f"Unknown engine mode {self.mode!r}. "
                "Choose local-llama (real), sim-frontier-70b, or sim-partitioned."
            )

    def print_welcome_banner(self):
        mem = psutil.virtual_memory()
        free_ram_gb = mem.available / (1024**3)
        total_ram_gb = mem.total / (1024**3)

        sim_note = "" if self.mode == "local-llama" else \
            "\n[bold red]SIMULATION MODE — no model is attached; generation will refuse.[/bold red]"
        banner = f"""[bold cyan]LLM-LAB INTERACTIVE TERMINAL[/bold cyan]
[dim]Engine: {self.mode} | Model: {self.model_name}[/dim]
[dim]System RAM: {total_ram_gb:.1f} GB Total | [bold green]{free_ram_gb:.1f} GB Available[/bold green][/dim]
[dim]Type [bold white]/help[/bold white] for commands or [bold white]/exit[/bold white] to quit.[/dim]{sim_note}"""
        console.print(Panel(banner, border_style="cyan"))

    def show_system_stats(self):
        mem = psutil.virtual_memory()
        proc = psutil.Process()
        proc_mem_gb = proc.memory_info().rss / (1024**3)

        table = Table(title="Measured Process/System Memory", header_style="bold magenta")
        table.add_column("Resource", style="cyan")
        table.add_column("Measured Value", style="green")

        table.add_row("Engine Process RSS", f"{proc_mem_gb:.2f} GB")
        table.add_row("System Free RAM", f"{mem.available / (1024**3):.2f} GB")
        table.add_row("System RAM Utilization", f"{mem.percent}%")
        if self.engine and hasattr(self.engine, "kv_cache"):
            table.add_row("KV Cache Footprint", f"{self.engine.kv_cache.get_memory_bytes() / (1024**2):.2f} MB")
        console.print(table)

    def generate_response_stream(self, user_prompt: str):
        """
        Streams generated tokens to the terminal while recording engine metrics.
        """
        self.conversation_history.append({"role": "user", "content": user_prompt})
        start_time = time.perf_counter()
        tokens_generated = 0
        
        if self.mode == "local-llama" and self.llm_real is not None:
            console.print("[bold green]Assistant (Real LLaMA-3 8B):[/bold green] ", end="")
            full_resp = ""
            for chunk in self.llm_real(user_prompt, max_tokens=self.max_tokens, stream=True, temperature=0.7):
                delta = chunk["choices"][0]["text"]
                sys.stdout.write(delta)
                sys.stdout.flush()
                full_resp += delta
                tokens_generated += 1
            sys.stdout.write("\n")
            elapsed = time.perf_counter() - start_time
            effective_tok_s = tokens_generated / max(elapsed, 1e-5)
            proc_ram = psutil.Process().memory_info().rss / (1024**3)
            
            telemetry_text = (
                f"[dim]Speed: [bold green]{effective_tok_s:.1f} tok/s[/bold green] | "
                f"Latency: [bold cyan]{elapsed:.2f}s[/bold cyan] ({tokens_generated} toks) | "
                f"RAM: [bold magenta]{proc_ram:.2f} GB[/bold magenta] (Bounded) | "
                f"Speculative Acceleration: [bold blue]Prompt Lookup Active[/bold blue][/dim]"
            )
            console.print(Panel(telemetry_text, border_style="dim"))
            self.conversation_history.append({"role": "assistant", "content": full_resp})
            self.total_tokens_session += tokens_generated
            return
            
        # Simulated engines have no real inference path. Previously this branch printed
        # hardcoded canned answers with a time.sleep(0.065) cadence and fabricated
        # "sparsity"/"draft accepted" telemetry, which read as real model output.
        raise RuntimeError(
            f"--engine {self.mode} cannot generate text: it is a simulation with no "
            "model attached (see the SIMULATION warning in its module docstring). "
            "Use --engine local-llama, which runs a real GGUF model via llama.cpp."
        )

    def run_repl(self):
        self.print_welcome_banner()
        
        while True:
            try:
                user_input = console.input("\n[bold cyan]You > [/bold cyan]").strip()
                if not user_input:
                    continue
                    
                if user_input.startswith("/"):
                    cmd = user_input.lower()
                    if cmd in ["/exit", "/quit"]:
                        console.print("[bold yellow]Exiting LLM-LAB Session. Goodbye![/bold yellow]")
                        break
                    elif cmd == "/stats":
                        self.show_system_stats()
                    elif cmd == "/clear":
                        self.conversation_history.clear()
                        console.print("[bold green]Conversation context cleared.[/bold green]")
                    elif cmd == "/help":
                        console.print("""
[bold cyan]Available Commands:[/bold cyan]
  [bold white]/stats[/bold white]  - Display live hardware memory, CPU, and KV-cache telemetry
  [bold white]/clear[/bold white]  - Reset conversation context and wipe KV cache
  [bold white]/tokens <N>[/bold white] - Adjust max generation tokens
  [bold white]/exit[/bold white]   - Exit interactive session
""")
                    elif cmd.startswith("/tokens"):
                        parts = cmd.split()
                        if len(parts) > 1 and parts[1].isdigit():
                            self.max_tokens = int(parts[1])
                            console.print(f"[bold green]Max tokens set to {self.max_tokens}[/bold green]")
                    else:
                        console.print(f"[bold red]Unknown command: {user_input}. Type /help for assistance.[/bold red]")
                    continue
                    
                self.generate_response_stream(user_input)
                
            except KeyboardInterrupt:
                console.print("\n[bold yellow]Interrupted by user. Type /exit to quit.[/bold yellow]")
            except Exception as e:
                console.print(f"[bold red]Error in chat loop: {e}[/bold red]")

def start_chat():
    parser = argparse.ArgumentParser(description="LLM-LAB Interactive Terminal")
    parser.add_argument("--engine", choices=["local-llama", "sim-frontier-70b", "sim-partitioned"], default="local-llama", help="Engine backend. Only local-llama runs a real model.")
    parser.add_argument("--partition-dir", default="models/llama-8b-real-partition", help="Directory for partitioned GGUF weights")
    args, unknown = parser.parse_known_args()
    
    session = InteractiveChatSession(mode=args.engine, partition_dir=args.partition_dir)
    session.run_repl()

if __name__ == "__main__":
    start_chat()
