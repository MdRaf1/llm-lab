"""
Plumbing check for the SIMULATED engine sketch (llm_lab.core.sim_engine).

This exercises wiring only. No model is loaded: hidden states are `np.random.randn`
and tokens are `np.random.randint`, so the tok/s and sparsity values it prints are
properties of numpy, not of any language model. Do not quote them.
"""

from llm_lab.core.sim_engine import BreakthroughEngine

def test_sim_engine_pipeline_runs():
    print("Initializing SIMULATED engine sketch (Llama-3.3-70B shape, no weights)...")
    
    engine = BreakthroughEngine(
        model_name="Llama-3.3-70B-Breakthrough",
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
    
    prompt = list(range(128)) # Initial prompt tokens
    print(f"Running generation step simulation (steps=10)...")
    results = engine.benchmark_step(prompt, steps=10)
    
    print("\n" + "="*50)
    print(f"Model: {results['model_name']}")
    print(f"Total Tokens Generated: {results['total_tokens_generated']}")
    print(f"Elapsed Time: {results['elapsed_seconds']:.3f} s")
    print(f"[SIMULATED - numpy speed only] {results['effective_tokens_per_second']:.1f} tok/s")
    print(f"Mean Accepted Tokens / Pass: {results['avg_accepted_tokens_per_pass']:.2f}")
    print(f"[CONFIG CONSTANT - not measured] skip ratio {results['avg_sparsity_skipped_pct']:.1f}%")
    print(f"KV Cache Footprint: {results['kv_cache_footprint_mb']:.2f} MB")
    print(f"RAM Bounded Guarantee (<2GB): {results['ram_bounded_guarantee']}")
    print("="*50 + "\n")
    
    assert results['total_tokens_generated'] > 0
    assert results['ram_bounded_guarantee'] is True
    print("Simulated engine plumbing OK (no model was loaded)")

if __name__ == "__main__":
    test_sim_engine_pipeline_runs()
