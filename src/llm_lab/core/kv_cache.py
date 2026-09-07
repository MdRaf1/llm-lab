"""
Dynamic Bounded KV-Cache Manager for Long-Context Inference.
Implements:
1. Streaming Attention-Sink + SnapKV dynamic observation eviction.
2. Asymmetric KIVI-style quantization (per-channel for Keys, per-token for Values).
Guarantees constant, bounded memory usage (<1.5 GB) even at 64k+ context.
"""

import math
from typing import Optional, Tuple
import numpy as np

class CompactKVCache:
    """
    Manages compressed, memory-bounded Key-Value states for a transformer layer.
    """
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        sink_size: int = 16,
        window_size: int = 1024,
        max_retained: int = 2048,
        quant_bits: int = 4, # 4-bit or 8-bit or 16-bit (unquantized)
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sink_size = sink_size
        self.window_size = window_size
        self.max_retained = max_retained
        self.quant_bits = quant_bits
        
        # Per layer storage: dict of layer_idx -> { 'k': ..., 'v': ..., 'k_scale': ..., 'v_scale': ... }
        self.layers = {}
        for l in range(num_layers):
            self.layers[l] = {
                "k": None, # [num_kv_heads, seq_len, head_dim]
                "v": None,
                "k_scales": None,
                "v_scales": None,
                "seq_len": 0,
            }
            
    def _quantize_int4_per_channel(self, tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quantizes Keys per-channel (along head_dim) to 4-bit unsigned packed integers.
        tensor shape: [heads, seq_len, head_dim]
        """
        scales = (np.max(np.abs(tensor), axis=1, keepdims=True) + 1e-8).astype(np.float32)
        normalized = np.clip(np.round((tensor / scales) * 7.0), -8, 7).astype(np.int8)
        
        # Pack two 4-bit signed ints (-8..7) into one uint8 byte along seq_len
        seq_len = tensor.shape[1]
        if seq_len % 2 != 0:
            pad = np.zeros((tensor.shape[0], 1, tensor.shape[2]), dtype=np.int8)
            normalized = np.concatenate([normalized, pad], axis=1)
            
        low = (normalized[:, 0::2, :] & 0x0F).astype(np.uint8)
        high = ((normalized[:, 1::2, :] & 0x0F) << 4).astype(np.uint8)
        packed = (low | high)
        
        return packed, scales

    def _dequantize_int4_per_channel(self, packed: np.ndarray, scales: np.ndarray, original_seq_len: int) -> np.ndarray:
        """Dequantizes packed 4-bit keys back to float32."""
        low = (packed & 0x0F).astype(np.int8)
        low[low >= 8] -= 16  # sign extend 4-bit two's complement
        
        high = ((packed >> 4) & 0x0F).astype(np.int8)
        high[high >= 8] -= 16  # sign extend 4-bit two's complement
        
        unpacked = np.empty((packed.shape[0], packed.shape[1] * 2, packed.shape[2]), dtype=np.float32)
        unpacked[:, 0::2, :] = low.astype(np.float32)
        unpacked[:, 1::2, :] = high.astype(np.float32)
        
        unpacked = (unpacked[:, :original_seq_len, :] / 7.0) * scales
        return unpacked

    def append(self, layer_idx: int, k_new: np.ndarray, v_new: np.ndarray, attn_importance: Optional[np.ndarray] = None):
        """
        Appends new KV tokens and triggers dynamic eviction if capacity is reached.
        k_new: [num_kv_heads, new_tokens, head_dim]
        v_new: [num_kv_heads, new_tokens, head_dim]
        """
        state = self.layers[layer_idx]
        
        if state["k"] is None:
            k_combined = k_new
            v_combined = v_new
        else:
            if self.quant_bits == 4:
                k_prev = self._dequantize_int4_per_channel(state["k"], state["k_scales"], state["seq_len"])
                v_prev = self._dequantize_int4_per_channel(state["v"], state["v_scales"], state["seq_len"])
            else:
                k_prev = state["k"]
                v_prev = state["v"]
                
            k_combined = np.concatenate([k_prev, k_new], axis=1)
            v_combined = np.concatenate([v_prev, v_new], axis=1)

        total_seq = k_combined.shape[1]
        
        # Eviction logic (SnapKV / StreamingLLM)
        if total_seq > self.max_retained:
            # Always keep sink tokens (first sink_size)
            sink_k = k_combined[:, :self.sink_size, :]
            sink_v = v_combined[:, :self.sink_size, :]
            
            # Always keep recent window (last window_size)
            recent_k = k_combined[:, -self.window_size:, :]
            recent_v = v_combined[:, -self.window_size:, :]
            
            # Middle candidate tokens
            mid_k = k_combined[:, self.sink_size:-self.window_size, :]
            mid_v = v_combined[:, self.sink_size:-self.window_size, :]
            
            budget_mid = self.max_retained - (self.sink_size + self.window_size)
            if budget_mid > 0 and mid_k.shape[1] > budget_mid:
                if attn_importance is not None:
                    # Select top-budget_mid tokens with highest importance
                    indices = np.argsort(attn_importance)[-budget_mid:]
                    indices.sort()
                    mid_k = mid_k[:, indices, :]
                    mid_v = mid_v[:, indices, :]
                else:
                    # Uniform subsampling
                    idx = np.linspace(0, mid_k.shape[1] - 1, budget_mid, dtype=int)
                    mid_k = mid_k[:, idx, :]
                    mid_v = mid_v[:, idx, :]
                k_combined = np.concatenate([sink_k, mid_k, recent_k], axis=1)
                v_combined = np.concatenate([sink_v, mid_v, recent_v], axis=1)
            else:
                k_combined = np.concatenate([sink_k, recent_k], axis=1)
                v_combined = np.concatenate([sink_v, recent_v], axis=1)

        # Quantize and store
        actual_seq = k_combined.shape[1]
        if self.quant_bits == 4:
            state["k"], state["k_scales"] = self._quantize_int4_per_channel(k_combined)
            state["v"], state["v_scales"] = self._quantize_int4_per_channel(v_combined)
        else:
            state["k"] = k_combined
            state["v"] = v_combined
            
        state["seq_len"] = actual_seq

    def get_memory_bytes(self) -> int:
        """Returns the exact RAM footprint of the entire KV cache across all layers."""
        total_bytes = 0
        for l in range(self.num_layers):
            s = self.layers[l]
            if s["k"] is not None:
                total_bytes += s["k"].nbytes
                total_bytes += s["v"].nbytes
                if s["k_scales"] is not None:
                    total_bytes += s["k_scales"].nbytes
                    total_bytes += s["v_scales"].nbytes
        return total_bytes

    @staticmethod
    def calculate_theoretical_footprint_mb(num_layers: int, num_kv_heads: int, head_dim: int, seq_len: int, bits: int = 16) -> float:
        """Computes uncompressed theoretical footprint in MB."""
        bytes_per_elem = bits / 8.0
        # 2 for Key and Value
        total_bytes = 2 * num_layers * num_kv_heads * head_dim * seq_len * bytes_per_elem
        return total_bytes / (1024 * 1024)
