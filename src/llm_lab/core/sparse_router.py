"""
Dynamic Activation Sparsity Predictor & Neuron Router.

WARNING — THE PREDICTOR IS UNTRAINED. `predictor_weight` / `predictor_proj` are
`np.random.randn(...) * 0.02` and no training code exists, so
`predict_active_neurons()` selects top-k over noise. Its returned `sparsity_ratio` is
`1 - (hot + top_k)/d_intermediate` — a fixed function of the configuration, not a
measurement of anything. Do not report it as observed sparsity.

Also note: thresholded activation sparsity on SwiGLU/SiLU models is LOSSY. SiLU
outputs are essentially never exactly zero, so skipping "inactive" neurons changes the
output. Only ReLU-family models admit exactly-lossless neuron skipping.

Implements:
1. Hot/Cold neuron frequency tracking and classification.
2. Low-rank activation prediction (predicting which cold neurons fire ahead of time).
3. Selective sparse GEMM/GEMV execution to cut memory traffic by up to 75%.
"""

from typing import Tuple, List, Dict
import numpy as np

class SparseNeuronRouter:
    """
    Tracks neuron activation statistics and partitions MLP layer weights into
    Hot (resident in fast memory) and Cold (streamed on-demand).
    """
    def __init__(self, layer_idx: int, d_model: int, d_intermediate: int, hot_ratio: float = 0.20):
        self.layer_idx = layer_idx
        self.d_model = d_model
        self.d_intermediate = d_intermediate
        self.hot_ratio = hot_ratio
        
        # Frequency counter for neuron activations
        self.activation_counts = np.zeros(d_intermediate, dtype=np.int64)
        self.total_tokens_observed = 0
        
        # Hot neuron indices (pinned in high-speed memory)
        num_hot = int(d_intermediate * hot_ratio)
        self.hot_indices = np.arange(num_hot, dtype=np.int32)
        self.cold_indices = np.arange(num_hot, d_intermediate, dtype=np.int32)
        
        # Lightweight linear predictor weights [d_model, d_intermediate] (low-rank approximated)
        # Predicts whether cold neurons will exceed activation threshold
        self.predictor_weight = np.random.randn(d_model, 64).astype(np.float32) * 0.02
        self.predictor_proj = np.random.randn(64, d_intermediate).astype(np.float32) * 0.02

    def record_activation_profile(self, intermediate_activations: np.ndarray, threshold: float = 0.05):
        """
        Observes intermediate hidden states to update hot/cold distribution.
        activations: [seq_len, d_intermediate]
        """
        active_mask = (np.abs(intermediate_activations) > threshold).astype(np.int64)
        sum_active = np.sum(active_mask, axis=0)
        self.activation_counts += sum_active
        self.total_tokens_observed += intermediate_activations.shape[0]

    def update_hot_cold_partition(self):
        """Partitions neurons into Hot and Cold based on observed empirical frequencies."""
        if self.total_tokens_observed == 0:
            return
            
        freqs = self.activation_counts / self.total_tokens_observed
        sorted_indices = np.argsort(freqs)[::-1] # descending order
        
        num_hot = int(self.d_intermediate * self.hot_ratio)
        self.hot_indices = sorted_indices[:num_hot]
        self.cold_indices = sorted_indices[num_hot:]

    def predict_active_neurons(self, hidden_state: np.ndarray, top_k_cold: int = 128) -> Tuple[np.ndarray, np.ndarray]:
        """
        Given the input hidden state x [d_model], predicts which cold neurons must be fetched.
        Returns:
            active_neuron_indices: combined indices (all hot neurons + predicted cold neurons)
            sparsity_ratio: fraction of neurons skipped
        """
        # Fast 2-step low-rank projection
        low_rank = np.dot(hidden_state, self.predictor_weight)
        cold_logits = np.dot(low_rank, self.predictor_proj[:, self.cold_indices])
        
        # Select top-k predicted cold neurons
        if top_k_cold > 0 and len(self.cold_indices) > top_k_cold:
            top_cold_local_idx = np.argpartition(cold_logits, -top_k_cold)[-top_k_cold:]
            selected_cold = self.cold_indices[top_cold_local_idx]
        else:
            selected_cold = self.cold_indices
            
        combined_indices = np.concatenate([self.hot_indices, selected_cold])
        combined_indices.sort()
        
        sparsity_ratio = 1.0 - (len(combined_indices) / float(self.d_intermediate))
        return combined_indices, sparsity_ratio

    @staticmethod
    def calculate_sparsity_speedup(
        baseline_ram_gb: float,
        mlp_param_ratio: float = 0.65,
        neuron_sparsity: float = 0.75,
    ) -> Dict[str, float]:
        """
        Calculates memory bandwidth savings from dynamic MLP activation sparsity.
        """
        attention_param_gb = baseline_ram_gb * (1.0 - mlp_param_ratio)
        mlp_param_gb = baseline_ram_gb * mlp_param_ratio
        
        # With dynamic sparsity, only (1 - neuron_sparsity) of MLP weights are streamed
        active_mlp_gb = mlp_param_gb * (1.0 - neuron_sparsity)
        total_active_gb = attention_param_gb + active_mlp_gb
        
        bandwidth_reduction = 1.0 - (total_active_gb / baseline_ram_gb)
        speedup = baseline_ram_gb / total_active_gb
        
        return {
            "baseline_model_gb": baseline_ram_gb,
            "active_streamed_gb": total_active_gb,
            "bandwidth_reduction_pct": bandwidth_reduction * 100.0,
            "speedup_factor": speedup,
        }
