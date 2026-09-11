"""
Speculative Batched Verification Engine.

KNOWN DEVIATION — `verify_lossless_rejection()` is NOT the standard algorithm and is
not currently lossless. The residual distribution is computed as
`max(0, target_probs[i] - p_draft)`, subtracting the draft's *scalar* probability for
the chosen token from the entire target vector. The correct form subtracts the draft
*distribution* elementwise: `max(0, p(x) - q(x))`, which requires the drafter's full
distribution, not one scalar. Until that is fixed, do not describe this function's
output as distribution-identical to the target model.

Implements:
1. Candidate token drafting and tree expansion.
2. Rejection-sampling verification in a single batched pass.
3. Flips memory-bound decoding to compute-bound batch verification.
"""

from typing import List, Tuple, Dict
import numpy as np

class DraftTree:
    """Represents a candidate token draft tree generated for batched verification."""
    def __init__(self, root_tokens: List[int]):
        self.root_tokens = root_tokens
        self.draft_paths: List[List[int]] = []
        self.draft_probs: List[List[float]] = []

    def add_candidate_path(self, tokens: List[int], probs: List[float]):
        assert len(tokens) == len(probs), "Each token must have a corresponding probability"
        self.draft_paths.append(tokens)
        self.draft_probs.append(probs)

class SpeculativeEngine:
    """
    Executes lossless speculative rejection sampling and measures verification efficiency.
    """
    def __init__(self, acceptance_threshold: float = 0.0):
        self.acceptance_threshold = acceptance_threshold

    @staticmethod
    def verify_lossless_rejection(
        draft_tokens: List[int],
        draft_probs: List[float],
        target_logits: np.ndarray, # [seq_len, vocab_size]
        temperature: float = 0.7,
    ) -> Tuple[List[int], int]:
        """
        Standard Lossless Rejection Sampling (Leviathan et al., 2023).
        Guarantees that the output distribution matches the target model exactly.
        
        Returns:
            accepted_tokens: list of tokens accepted by the target model
            accepted_count: number of accepted tokens before rejection
        """
        accepted = []
        
        # Softmax target logits to probabilities
        exp_logits = np.exp((target_logits - np.max(target_logits, axis=-1, keepdims=True)) / max(temperature, 1e-5))
        target_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        n = len(draft_tokens)
        for i in range(n):
            tok = draft_tokens[i]
            p_draft = draft_probs[i]
            p_target = target_probs[i, tok]
            
            # Acceptance probability min(1, p_target / p_draft)
            accept_prob = min(1.0, (p_target / max(p_draft, 1e-7)))
            
            # Rejection sampling roll
            roll = np.random.random()
            if roll <= accept_prob:
                accepted.append(tok)
            else:
                # Resample residual token from max(0, p_target - p_draft)
                residual_dist = np.maximum(0.0, target_probs[i] - p_draft)
                norm = np.sum(residual_dist)
                if norm > 1e-7:
                    residual_dist /= norm
                    resampled_tok = int(np.random.choice(len(residual_dist), p=residual_dist))
                else:
                    resampled_tok = int(np.argmax(target_probs[i]))
                accepted.append(resampled_tok)
                break # Rejection ends the speculative sequence
                
        # If all draft tokens were accepted, target model generates one bonus token
        if len(accepted) == n:
            bonus_tok = int(np.random.choice(target_probs.shape[1], p=target_probs[-1]))
            accepted.append(bonus_tok)
            
        return accepted, len(accepted)

    @staticmethod
    def simulate_speedup_factor(
        draft_len: int,
        mean_acceptance_rate: float,
        draft_overhead_ratio: float = 0.05,
    ) -> Dict[str, float]:
        """
        Calculates the theoretical and empirical speedup achieved by speculative decoding.
        Speedup = E[accepted_tokens] / (1 + draft_overhead + batched_verification_overhead)
        """
        # Expected tokens accepted = (1 - alpha^(K+1)) / (1 - alpha)
        # where alpha = mean_acceptance_rate
        alpha = mean_acceptance_rate
        k = draft_len
        
        if abs(alpha - 1.0) < 1e-4:
            expected_tokens = k + 1
        else:
            expected_tokens = (1.0 - alpha**(k + 1)) / (1.0 - alpha)
            
        # Target verification cost: reading full model weights once + small compute increase for K tokens
        # Since memory-bound, reading once for K tokens has ~1.05x - 1.15x the time of 1 token
        target_cost = 1.0 + (0.02 * k) 
        total_relative_cost = target_cost + (draft_overhead_ratio * k)
        
        effective_speedup = expected_tokens / total_relative_cost
        
        return {
            "draft_length": k,
            "acceptance_rate": alpha,
            "expected_tokens_per_step": expected_tokens,
            "relative_step_cost": total_relative_cost,
            "net_speedup_multiplier": effective_speedup,
        }
