"""
Plumbing check for the SIMULATED partitioned-GGUF engine.

It verifies the manifest is readable and the mmap wiring holds. It does NOT verify
inference: the engine reads a 1024-byte slice per layer and discards it, then advances
the hidden state with `hidden + 0.005*tanh(hidden)` and emits `np.random.randint`
tokens. Its tok/s and RSS therefore measure nothing about a language model.
"""

import os
from llm_lab.frontier.sim_gguf_stream_engine import PartitionedGGUFStreamEngine

def test_sim_partitioned_gguf_plumbing():
    partition_dir = "models/llama-8b-real-partition"
    if not os.path.exists(os.path.join(partition_dir, "manifest.json")):
        print("Partitioned model not present, skipping.")
        return

    engine = PartitionedGGUFStreamEngine(partition_dir=partition_dir)
    assert engine.num_layers == 32
    assert engine.d_model == 4096
    assert engine.d_intermediate == 14336

    # Runs the simulated loop; assertions cover wiring only, not inference quality.
    results = engine.benchmark(max_tokens=10)
    assert results["tokens"] >= 10
    assert results["tok_s"] > 0

    engine.close()
    print("Simulated partitioned-GGUF plumbing OK (no real compute performed)")

if __name__ == "__main__":
    test_sim_partitioned_gguf_plumbing()
