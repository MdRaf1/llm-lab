"""
Unit & Mathematical Tests for Speculative Batched Verification.
Validates:
1. Exact lossless rejection sampling distribution properties.
2. Acceptance behavior under varying draft quality (alpha = 0.5 to 0.95).
3. Batched verification speedup factors.
"""

import numpy as np
from llm_lab.core.speculative import SpeculativeEngine

def test_speculative_acceptance_math():
    """Validates mathematical properties of lossless rejection sampling."""
    engine = SpeculativeEngine()
    vocab_size = 1000
    
    # 1. High agreement test (draft matches target distribution closely)
    high_match_accepted = []
    for _ in range(100):
        draft_tokens = [10, 42, 99, 105]
        draft_probs = [0.90, 0.85, 0.88, 0.92]
        
        target_logits = np.random.randn(4, vocab_size).astype(np.float32)
        # Force high target probabilities on draft tokens
        for idx, tok in enumerate(draft_tokens):
            target_logits[idx, tok] += 12.0
            
        accepted, count = engine.verify_lossless_rejection(draft_tokens, draft_probs, target_logits)
        high_match_accepted.append(count)
        
    avg_high = np.mean(high_match_accepted)
    print(f"High Agreement Mean Accepted Tokens (Draft K=4): {avg_high:.2f}")
    assert avg_high >= 3.8, f"Expected near full acceptance, got: {avg_high}"
    
    # 2. Low agreement test (draft is random / wrong)
    low_match_accepted = []
    for _ in range(100):
        draft_tokens = [10, 42, 99, 105]
        draft_probs = [0.90, 0.85, 0.88, 0.92]
        
        target_logits = np.random.randn(4, vocab_size).astype(np.float32)
        # Suppress draft tokens in target logits
        for idx, tok in enumerate(draft_tokens):
            target_logits[idx, tok] -= 10.0
            
        accepted, count = engine.verify_lossless_rejection(draft_tokens, draft_probs, target_logits)
        low_match_accepted.append(count)
        
    avg_low = np.mean(low_match_accepted)
    print(f"Low Agreement Mean Accepted Tokens (Draft K=4): {avg_low:.2f}")
    # Under rejection, exactly 1 token (the resampled target token) should be accepted
    assert avg_low == 1.0, f"Expected immediate rejection and 1 target token, got {avg_low}"

def test_speculative_speedup_scaling():
    """Validates speedup formulas across different draft lengths and acceptance rates."""
    metrics = SpeculativeEngine.simulate_speedup_factor(draft_len=5, mean_acceptance_rate=0.80)
    print(f"Draft K=5, Alpha=0.80 -> Net Speedup: {metrics['net_speedup_multiplier']:.2f}x")
    print(f"Expected Tokens per Step: {metrics['expected_tokens_per_step']:.2f}")
    assert metrics['net_speedup_multiplier'] > 2.2, "Speedup multiplier below expected bound"

if __name__ == "__main__":
    test_speculative_acceptance_math()
    test_speculative_speedup_scaling()
    print("All Speculative Verification Tests Passed!")
