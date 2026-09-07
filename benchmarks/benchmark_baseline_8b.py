"""
Benchmark Baseline 8B Inference Performance on Ollama / llama.cpp.
Measures prompt prefill tokens/sec, autoregressive decode tokens/sec, and memory footprint.
"""

import time
import requests
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def benchmark_ollama(model_name: str = "sec-llama8b:latest", prompt: str = "Explain quantum computing in 3 clear sentences."):
    console.print(Panel.fit(f"[bold cyan]BASELINE 8B INFERENCE BENCHMARK: {model_name}[/bold cyan]"))
    
    url = "http://127.0.0.1:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 50,
            "temperature": 0.0, # Deterministic greedy
        }
    }
    
    console.print(f"[bold yellow]>> Sending prompt to {model_name}...[/bold yellow]")
    t0 = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=120)
    t1 = time.perf_counter()
    
    data = resp.json()
    response_text = data.get("response", "")
    
    eval_count = data.get("eval_count", 0)
    eval_duration_ns = data.get("eval_duration", 1)
    prompt_eval_count = data.get("prompt_eval_count", 0)
    prompt_eval_duration_ns = data.get("prompt_eval_duration", 1)
    
    decode_tok_s = eval_count / (eval_duration_ns / 1e9)
    prefill_tok_s = prompt_eval_count / (prompt_eval_duration_ns / 1e9)
    total_latency_s = t1 - t0
    
    console.print(Panel(response_text, title="[bold green]Model Response[/bold green]"))
    
    table = Table(title="Baseline 8B Hardware Performance (Ryzen 5 5600G)", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Measurement", style="bold green")
    table.add_column("Hardware Implication", style="dim")
    
    table.add_row("Prompt Prefill Speed", f"{prefill_tok_s:.2f} tok/s", "Compute-bound matrix multiplication")
    table.add_row("Autoregressive Decode Speed", f"{decode_tok_s:.2f} tok/s", "Strictly memory-bandwidth bound (DDR4)")
    table.add_row("Tokens Generated", str(eval_count), "Greedy decoding")
    table.add_row("Total Wall-Clock Latency", f"{total_latency_s:.2f} s", f"{eval_count + prompt_eval_count} total tokens")
    console.print(table)
    
    return {
        "decode_tok_s": decode_tok_s,
        "prefill_tok_s": prefill_tok_s,
        "eval_count": eval_count,
        "response": response_text
    }

if __name__ == "__main__":
    benchmark_ollama()
