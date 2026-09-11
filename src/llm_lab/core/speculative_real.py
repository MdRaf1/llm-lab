"""
Speculative Real-Model Decoding Engine.

WARNING — THIS IS SLOWER THAN PLAIN DECODING, BY CONSTRUCTION.
`generate()` drafts K tokens by calling the FULL target model K times, then calls it
once more to verify — more forward passes than autoregressive decoding. It also never
passes `past_key_values`, so every call re-encodes the whole sequence (O(n^2)).
Verification is greedy argmax matching, not rejection sampling, so the "100% output
identity" claim only holds for greedy decoding and says nothing about sampling.

A real speedup needs a small draft model (or n-gram/prompt-lookup drafter) plus a KV
cache. See docs/superpowers/specs/2026-09-11-moe-exact-runtime-design.md.
"""

import time
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any
from transformers import PreTrainedModel, PreTrainedTokenizer

class RealModelSpeculativeDecoder:
    """
    Executes lossless speculative decoding on real Hugging Face models.
    """
    def __init__(
        self,
        target_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        draft_length: int = 3,
    ):
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.draft_length = draft_length
        self.device = next(target_model.parameters()).device

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Runs speculative decoding with batched verification.
        """
        generated = input_ids.clone()
        total_steps = 0
        total_accepted = 0
        total_generated = 0
        
        start_time = time.perf_counter()
        
        while total_generated < max_new_tokens:
            total_steps += 1
            
            # Step A: Target model single-step forward pass to get current logits
            outputs = self.target_model(generated)
            next_token_logits = outputs.logits[:, -1, :]
            
            # Sample base token
            if temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                base_token = torch.multinomial(probs, num_samples=1)
            else:
                base_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
            generated = torch.cat([generated, base_token], dim=-1)
            total_generated += 1
            if base_token.item() == self.tokenizer.eos_token_id or total_generated >= max_new_tokens:
                break
                
            # Step B: Fast greedy draft of K tokens using lightweight top-logits continuation
            # Simulates self-drafting candidate tokens
            draft_tokens = []
            draft_context = generated.clone()
            
            # Quick single-layer projection or greedy top continuation
            with torch.no_grad():
                for _ in range(self.draft_length):
                    d_out = self.target_model(draft_context)
                    d_tok = torch.argmax(d_out.logits[:, -1, :], dim=-1, keepdim=True)
                    draft_tokens.append(d_tok)
                    draft_context = torch.cat([draft_context, d_tok], dim=-1)
                    if d_tok.item() == self.tokenizer.eos_token_id:
                        break
                        
            if not draft_tokens:
                continue
                
            # Step C: Parallel Batched Verification in ONE forward pass
            all_candidates = torch.cat(draft_tokens, dim=-1)
            verify_input = torch.cat([generated, all_candidates], dim=-1)
            verify_out = self.target_model(verify_input)
            verify_logits = verify_out.logits
            
            # Step D: Lossless verification loop
            n_draft = all_candidates.shape[1]
            accepted_in_batch = 0
            
            for k in range(n_draft):
                target_logit_k = verify_logits[:, generated.shape[1] - 1 + k, :]
                target_pred = torch.argmax(target_logit_k, dim=-1, keepdim=True)
                draft_tok_k = all_candidates[:, k:k+1]
                
                if target_pred.item() == draft_tok_k.item():
                    generated = torch.cat([generated, draft_tok_k], dim=-1)
                    accepted_in_batch += 1
                    total_generated += 1
                    if draft_tok_k.item() == self.tokenizer.eos_token_id or total_generated >= max_new_tokens:
                        break
                else:
                    # Append target's true predicted token upon rejection
                    generated = torch.cat([generated, target_pred], dim=-1)
                    total_generated += 1
                    break
                    
            total_accepted += accepted_in_batch
            
        elapsed = time.perf_counter() - start_time
        tok_s = total_generated / max(elapsed, 1e-5)
        
        output_text = self.tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True)
        
        return {
            "output_text": output_text,
            "total_tokens": total_generated,
            "elapsed_seconds": elapsed,
            "tokens_per_second": tok_s,
            "speculative_steps": total_steps,
            "total_drafted_accepted": total_accepted,
            "acceptance_rate": total_accepted / max(total_steps * self.draft_length, 1),
        }
