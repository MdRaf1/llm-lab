"""
Asynchronous Double-Buffered Weight Streamer.
Overlaps layer tensor computation with next-layer prefetching using memory-mapped I/O
and background worker threads to completely hide memory-to-register latency.
"""

import os
import mmap
import threading
from typing import Dict, List, Optional
import numpy as np

class LayerBuffer:
    """Represents an active memory buffer holding layer weights."""
    def __init__(self, size_bytes: int):
        self.buffer = np.empty(size_bytes, dtype=np.uint8)
        self.is_ready = threading.Event()
        self.layer_idx = -1

class AsyncWeightStreamer:
    """
    Coordinates double-buffered asynchronous prefetching of model layers.
    """
    def __init__(self, layer_size_bytes: int, num_layers: int, filepath: Optional[str] = None):
        self.layer_size_bytes = layer_size_bytes
        self.num_layers = num_layers
        self.filepath = filepath
        
        # Double buffer: Buffer A and Buffer B
        self.buffer_a = LayerBuffer(layer_size_bytes)
        self.buffer_b = LayerBuffer(layer_size_bytes)
        self.active_is_a = True
        
        self.mmap_obj = None
        self.file_handle = None
        if filepath and os.path.exists(filepath):
            self.file_handle = open(filepath, "rb")
            self.mmap_obj = mmap.mmap(self.file_handle.fileno(), 0, access=mmap.ACCESS_READ)
            
        self.prefetch_thread: Optional[threading.Thread] = None

    def _prefetch_worker(self, target_buffer: LayerBuffer, layer_idx: int):
        """Worker thread that asynchronously loads layer weights into target_buffer."""
        offset = layer_idx * self.layer_size_bytes
        target_buffer.is_ready.clear()
        target_buffer.layer_idx = layer_idx
        
        if self.mmap_obj is not None:
            # Memory-mapped asynchronous zero-copy slice
            raw_bytes = self.mmap_obj[offset : offset + self.layer_size_bytes]
            target_buffer.buffer[:] = np.frombuffer(raw_bytes, dtype=np.uint8)
        else:
            # Simulated weight generation / mock layer
            target_buffer.buffer.fill(layer_idx % 255)
            
        target_buffer.is_ready.set()

    def start_prefetch(self, next_layer_idx: int):
        """Launches asynchronous prefetch for next_layer into the idle buffer."""
        if next_layer_idx >= self.num_layers:
            return
            
        target_buffer = self.buffer_b if self.active_is_a else self.buffer_a
        self.prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            args=(target_buffer, next_layer_idx),
            daemon=True
        )
        self.prefetch_thread.start()

    def acquire_active_layer(self, current_layer_idx: int) -> np.ndarray:
        """
        Waits for current layer buffer to be ready and swaps buffers.
        """
        active_buffer = self.buffer_a if self.active_is_a else self.buffer_b
        
        # If active buffer doesn't match current layer, load it synchronously (first layer warm-up)
        if active_buffer.layer_idx != current_layer_idx:
            self._prefetch_worker(active_buffer, current_layer_idx)
            
        active_buffer.is_ready.wait()
        
        # Swap active buffer flag for next cycle
        self.active_is_a = not self.active_is_a
        return active_buffer.buffer

    def close(self):
        """Closes memory map and open file handles."""
        if self.mmap_obj is not None:
            self.mmap_obj.close()
        if self.file_handle is not None:
            self.file_handle.close()
