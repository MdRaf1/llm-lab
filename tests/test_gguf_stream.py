"""
Unit test for PartitionedGGUFStreamEngine.
"""

import os
from llm_lab.frontier.gguf_stream_engine import PartitionedGGUFStreamEngine

def test_partitioned_gguf_stream_engine():
    partition_dir = "models/llama-8b-real-partition"
    if not os.path.exists(os.path.join(partition_dir, "manifest.json")):
        print("Partitioned model not present, skipping.")
        return
        
    engine = PartitionedGGUFStreamEngine(partition_dir=partition_dir)
    assert engine.num_layers == 32
    assert engine.d_model == 4096
    assert engine.d_intermediate == 14336
    
    # Run a short benchmark pass
    results = engine.benchmark(max_tokens=10)
    assert results["tokens"] >= 10
    assert results["tok_s"] > 0
    assert results["rss_mb"] < 4000 # Strictly below 4GB RAM
    
    engine.close()
    print("test_partitioned_gguf_stream_engine passed successfully!")

if __name__ == "__main__":
    test_partitioned_gguf_stream_engine()
