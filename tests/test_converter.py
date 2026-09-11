"""
Plumbing check for the SIMULATED partitioner (llm_lab.frontier.sim_converter).

The "activation skew profiling" it exercises is `np.random.zipf`, not a model run, so
the hot/cold split it produces is arbitrary. This test only confirms that the binary
streams and manifest are written and internally consistent.

For real partitioning of an actual GGUF file, see frontier/gguf_partitioner.py.
"""

import os
import json
import shutil
import tempfile
from llm_lab.frontier.sim_converter import ModelPartitioner

def test_model_partitioner():
    temp_dir = tempfile.mkdtemp()
    try:
        partitioner = ModelPartitioner(
            d_model=1024,
            d_intermediate=4096,
            num_layers=4,
            hot_ratio=0.20,
            quant_bits=4
        )
        
        manifest = partitioner.partition_and_export(output_dir=temp_dir, model_name="Test-70B-Model")
        
        assert os.path.exists(os.path.join(temp_dir, "model_hot.bin"))
        assert os.path.exists(os.path.join(temp_dir, "model_cold.bin"))
        assert os.path.exists(os.path.join(temp_dir, "manifest.json"))
        
        assert manifest["num_hot_neurons"] == int(4096 * 0.20)
        assert manifest["num_cold_neurons"] == 4096 - manifest["num_hot_neurons"]
        print(f"Hot Neurons: {manifest['num_hot_neurons']}, Cold Neurons: {manifest['num_cold_neurons']}")
        print("Model Partitioner Validation Passed!")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    test_model_partitioner()
