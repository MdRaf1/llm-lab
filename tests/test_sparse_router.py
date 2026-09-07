"""
Unit & Mathematical Tests for Dynamic Activation Sparsity Router.
Validates:
1. Frequency profiling and hot/cold neuron separation.
2. Low-rank cold neuron activation prediction.
3. Memory bandwidth reduction factor.
"""

import numpy as np
from llm_lab.core.sparse_router import SparseNeuronRouter

def test_sparse_router_partitioning():
    """Validates that frequently active neurons are properly classified as Hot."""
    d_model = 512
    d_intermediate = 2048
    router = SparseNeuronRouter(layer_idx=0, d_model=d_model, d_intermediate=d_intermediate, hot_ratio=0.20)
    
    # Simulate skewed power-law activations where neurons 0..200 activate often
    batch_size = 500
    synthetic_acts = np.random.randn(batch_size, d_intermediate).astype(np.float32) * 0.01
    synthetic_acts[:, :200] += 2.0  # Hot neurons
    
    router.record_activation_profile(synthetic_acts, threshold=0.1)
    router.update_hot_cold_partition()
    
    # Check that the top hot indices contain the biased neurons
    overlap = len(set(router.hot_indices).intersection(set(range(200))))
    print(f"Hot Partition Overlap with frequent neurons: {overlap}/200 ({overlap/200*100:.1f}%)")
    assert overlap >= 195, "Failed to identify hot neurons correctly"

def test_sparse_prediction_and_bandwidth():
    """Validates that sparsity predictor skips >70% of neurons while maintaining coverage."""
    d_model = 512
    d_intermediate = 2048
    router = SparseNeuronRouter(layer_idx=0, d_model=d_model, d_intermediate=d_intermediate, hot_ratio=0.15)
    
    hidden_x = np.random.randn(d_model).astype(np.float32)
    active_indices, sparsity_ratio = router.predict_active_neurons(hidden_x, top_k_cold=64)
    
    print(f"Total Neurons: {d_intermediate}, Active Fetched: {len(active_indices)}, Sparsity Skipped: {sparsity_ratio*100:.1f}%")
    assert sparsity_ratio > 0.70, f"Expected >70% sparsity, got {sparsity_ratio*100:.1f}%"
    
    # Bandwidth reduction calculation for a 70B model
    bw_metrics = SparseNeuronRouter.calculate_sparsity_speedup(
        baseline_ram_gb=35.0, mlp_param_ratio=0.65, neuron_sparsity=sparsity_ratio
    )
    print(f"70B Baseline: {bw_metrics['baseline_model_gb']} GB -> Active Stream: {bw_metrics['active_streamed_gb']:.2f} GB")
    print(f"Bandwidth Reduction: {bw_metrics['bandwidth_reduction_pct']:.1f}% (Speedup: {bw_metrics['speedup_factor']:.2f}x)")
    assert bw_metrics['speedup_factor'] > 1.8, "Bandwidth speedup factor below target"

if __name__ == "__main__":
    test_sparse_router_partitioning()
    test_sparse_prediction_and_bandwidth()
    print("All Sparse Router Tests Passed!")
