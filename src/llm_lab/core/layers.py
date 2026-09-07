"""
Fused Quantized PyTorch Layers for Seamless LLM Replacement.
Replaces standard nn.Linear with memory-efficient 4-bit packed weights
and routes autoregressive decoding directly to our JIT AVX2 kernels.
"""

import torch
import torch.nn as nn
import numpy as np
from .kernels import pack_weights_int4, fused_gemv_int4

class FusedInt4Linear(nn.Module):
    """
    Drop-in replacement for torch.nn.Linear that stores weights in packed 4-bit format
    (8x smaller than FP32, 4x smaller than FP16) and computes decode steps via fused AVX2.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Packed weights: [out_features, in_features // 2] uint8
        self.register_buffer("w_packed", torch.empty((out_features, in_features // 2), dtype=torch.uint8))
        self.register_buffer("scales", torch.empty(out_features, dtype=torch.float32))
        
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float32))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "FusedInt4Linear":
        """Converts an existing nn.Linear into a FusedInt4Linear layer."""
        fused = cls(linear.in_features, linear.out_features, bias=(linear.bias is not None))
        
        w_np = linear.weight.detach().cpu().to(torch.float32).numpy()
        packed_np, scales_np = pack_weights_int4(w_np)
        
        fused.w_packed.copy_(torch.from_numpy(packed_np))
        fused.scales.copy_(torch.from_numpy(scales_np))
        
        if linear.bias is not None:
            fused.bias.copy_(linear.bias.detach().cpu().to(torch.float32))
            
        return fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        
        # Single-token decode mode: shape is [1, 1, in_features] or [1, in_features]
        if x.numel() == self.in_features:
            x_np = x.view(-1).detach().cpu().to(torch.float32).numpy()
            y_np = fused_gemv_int4(self.w_packed.numpy(), self.scales.numpy(), x_np)
            y = torch.from_numpy(y_np).to(dtype=x.dtype, device=x.device).view(orig_shape[0], -1, self.out_features)
            if self.bias is not None:
                y = y + self.bias
            return y
            
        # Batched / Prefill mode: unpack and run standard PyTorch GEMM
        low = (self.w_packed & 0x0F).to(torch.int8)
        low[low >= 8] -= 16
        high = ((self.w_packed >> 4) & 0x0F).to(torch.int8)
        high[high >= 8] -= 16
        
        unpacked = torch.empty((self.out_features, self.in_features), dtype=torch.float32, device=x.device)
        unpacked[:, 0::2] = low.to(torch.float32)
        unpacked[:, 1::2] = high.to(torch.float32)
        
        w_dequant = unpacked * self.scales.unsqueeze(1)
        y = torch.matmul(x, w_dequant.t().to(x.dtype))
        if self.bias is not None:
            y = y + self.bias
        return y
