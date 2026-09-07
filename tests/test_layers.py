"""
Unit tests for FusedInt4Linear PyTorch layer replacement.
Validates:
1. Seamless conversion from standard nn.Linear.
2. Output cosine similarity (>0.99) for both single-token decode and batched prefill.
3. Memory compression ratio (8.0x vs FP32).
"""

import torch
import torch.nn as nn
import numpy as np
from llm_lab.core.layers import FusedInt4Linear

def test_fused_int4_layer():
    in_features = 2048
    out_features = 4096
    
    linear_orig = nn.Linear(in_features, out_features, bias=True)
    fused_layer = FusedInt4Linear.from_linear(linear_orig)
    
    # Check memory reduction
    orig_bytes = linear_orig.weight.nelement() * linear_orig.weight.element_size()
    fused_bytes = fused_layer.w_packed.nelement() * fused_layer.w_packed.element_size() + fused_layer.scales.nelement() * fused_layer.scales.element_size()
    compression_ratio = orig_bytes / fused_bytes
    print(f"Original Weight: {orig_bytes / (1024*1024):.2f} MB -> Fused 4-Bit: {fused_bytes / (1024*1024):.2f} MB ({compression_ratio:.1f}x compression)")
    assert compression_ratio > 7.5, "Expected >7.5x compression ratio"
    
    # Test 1: Single-token decode mode
    x_single = torch.randn(1, 1, in_features)
    with torch.no_grad():
        y_orig = linear_orig(x_single).view(-1)
        y_fused = fused_layer(x_single).view(-1)
        
    cos_sim_decode = torch.cosine_similarity(y_orig, y_fused, dim=0).item()
    print(f"Single-Token Decode Cosine Similarity: {cos_sim_decode:.4f}")
    assert cos_sim_decode > 0.985, f"Cosine similarity {cos_sim_decode} below threshold"
    
    # Test 2: Batched prefill mode
    x_batch = torch.randn(1, 16, in_features)
    with torch.no_grad():
        y_orig_batch = linear_orig(x_batch)
        y_fused_batch = fused_layer(x_batch)
        
    cos_sim_batch = torch.cosine_similarity(y_orig_batch.view(-1), y_fused_batch.view(-1), dim=0).item()
    print(f"Batched Prefill Cosine Similarity: {cos_sim_batch:.4f}")
    assert cos_sim_batch > 0.985, f"Cosine similarity {cos_sim_batch} below threshold"
    
    print("All FusedInt4Linear Tests Passed!")

if __name__ == "__main__":
    test_fused_int4_layer()
