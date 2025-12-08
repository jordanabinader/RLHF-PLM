"""
Length-based logit biasing for controlled sequence generation.

This module provides a custom logit processor that biases the EOS token
to encourage generation stopping near a target length.
"""

import torch
from transformers import LogitsProcessor


class LengthLogitBias(LogitsProcessor):
    """
    Biases EOS token logits to encourage stopping near target length.
    
    This processor increases the probability of generating the EOS token
    as the sequence length exceeds the target, helping the model respect
    length preferences encoded in user weights.
    """
    
    def __init__(self, eos_token_id: int, target_length: int, bias_strength: float = 5.0):
        """
        Initialize length logit bias processor.
        
        Args:
            eos_token_id: Token ID for end-of-sequence
            target_length: Target sequence length
            bias_strength: How strongly to bias EOS (higher = stronger effect)
        """
        self.eos_token_id = eos_token_id
        self.target_length = target_length
        self.bias_strength = bias_strength
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Apply length bias to logits.
        
        Args:
            input_ids: Current sequence tokens (batch_size, seq_len)
            scores: Next token logits (batch_size, vocab_size)
        
        Returns:
            Modified logits with EOS bias applied
        """
        current_length = input_ids.shape[-1]
        length_ratio = current_length / self.target_length
        
        # Only bias when we exceed target length
        if length_ratio > 1.0:
            # Linearly increase EOS bias as we exceed target
            bias = (length_ratio - 1.0) * self.bias_strength
            scores[:, self.eos_token_id] += bias
        
        return scores

