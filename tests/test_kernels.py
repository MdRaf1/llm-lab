"""
Unit & Performance Tests for Native Fused Quantized GEMV Kernels.
Validates:
1. Mathematical equivalence of fused INT4 GEMV against float32 dot product.
2. JIT AVX2 parallel execution speed and memory reduction.
"""

import numpy as np
from llm_lab.core.kernels import pack_weights_int4, fused_gemv_int4, benchmark_kernel

def test_int4_gemv_correctness():
    """Validates that fused INT4 GEMV approximates FP32 GEMV with >0.98 cosine similarity."""
    M, K = 512, 1024
    weights = np.random.randn(M, K).astype(np.float32) * 0.02
    x = np.random.randn(K).astype(np.float32)
    
    # Ground truth FP32
    y_ground_truth = weights @ x
    
    # Fused INT4
    w_packed, scales = pack_weights_int4(weights)
    y_int4 = fused_gemv_int4(w_packed, scales, x)
    
    # Cosine similarity
    cos_sim = np.dot(y_ground_truth, y_int4) / (
        np.linalg.norm(y_ground_truth) * np.linalg.norm(y_int4) + 1e-8
    )
    
    print(f"Fused INT4 vs FP32 Cosine Similarity: {cos_sim:.4f}")
    assert cos_sim > 0.98, f"Cosine similarity {cos_sim} below threshold 0.98"

def test_int4_kernel_performance():
    """Benchmarks fused INT4 GEMV on a realistic transformer layer projection (4096 x 4096)."""
    metrics = benchmark_kernel(M=4096, K=4096, iterations=10)
    print(f"Matrix: {metrics['M']}x{metrics['K']}")
    print(f"FP32 Size: {metrics['fp32_size_mb']:.2f} MB -> INT4 Size: {metrics['int4_size_mb']:.2f} MB ({metrics['compression_ratio']:.1f}x compression)")
    print(f"Fused INT4 Latency: {metrics['latency_ms']:.2f} ms")
    print(f"Effective Bandwidth: {metrics['effective_bandwidth_gb_s']:.2f} GB/s")
    assert metrics['compression_ratio'] > 7.5, "Expected >7.5x compression ratio"

if __name__ == "__main__":
    test_int4_gemv_correctness()
    test_int4_kernel_performance()
    print("All Kernel Tests Passed!")
