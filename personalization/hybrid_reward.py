"""
Hybrid reward function combining validity constraints with personalized rewards.

This module implements R_final(x):
- R_final(x) = R_penalty if invalid
- R_final(x) = w^(u)^T · g(x) if valid
"""
import torch
from typing import List, Tuple, Callable

from personalization.unified_property_fn import UnifiedPropertyFunction
from personalization.personas import Persona, compute_personalized_reward
from personalization.validity import validate_sequences


def create_hybrid_reward_fn(
    property_function: UnifiedPropertyFunction,
    persona: Persona,
    penalty: float = -10.0,
    min_charge: float = 0.0,
    device: str = "cuda",
) -> Callable:
    """
    Create hybrid reward function with validity constraints.
    
    The reward function enforces correctness as a hard constraint:
    - Invalid sequences receive a large negative penalty
    - Valid sequences receive personalized rewards based on persona weights
    
    Args:
        property_function: Unified property function g(x)
        persona: User persona with weights w^(u)
        penalty: Reward for invalid sequences (large negative value)
        min_charge: Minimum net charge requirement
        device: Computation device
    
    Returns:
        Reward function (sequences) -> (rewards, mask)
    """
    def reward_fn(sequences: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute hybrid rewards for sequences.
        
        Args:
            sequences: List of protein sequences
        
        Returns:
            rewards: Tensor of shape (batch_size,)
            valid_mask: Boolean tensor of shape (batch_size,)
        """
        if not sequences:
            return (
                torch.empty(0, device=device),
                torch.empty(0, dtype=torch.bool, device=device)
            )
        
        # 1. Check validity constraints
        valid_mask = validate_sequences(sequences, min_charge=min_charge)
        valid_mask = valid_mask.to(device)
        
        # 2. Compute personalized rewards for all sequences
        with torch.no_grad():
            properties = property_function(sequences)  # (B, 4)
            persona_rewards = compute_personalized_reward(properties, persona)  # (B,)
        
        # 3. Apply conditional reward structure
        # Invalid sequences get penalty, valid sequences get personalized reward
        rewards = torch.where(
            valid_mask,
            persona_rewards,  # Valid: use personalized reward
            torch.full_like(persona_rewards, penalty)  # Invalid: penalty
        )
        
        return rewards, valid_mask
    
    return reward_fn


def create_blended_hybrid_reward_fn(
    property_function: UnifiedPropertyFunction,
    persona: Persona,
    base_reward_fn: Callable,
    blend_weight: float = 0.5,
    penalty: float = -10.0,
    min_charge: float = 0.0,
    device: str = "cuda",
) -> Callable:
    """
    Create hybrid reward that blends base rewards with personalized rewards.
    
    Useful for gradually introducing personalization while maintaining
    compatibility with existing reward signals.
    
    Args:
        property_function: Unified property function
        persona: User persona
        base_reward_fn: Original reward function (e.g., classifier-based)
        blend_weight: Weight for personalized rewards (0 = all base, 1 = all personalized)
        penalty: Reward for invalid sequences
        min_charge: Minimum net charge requirement
        device: Computation device
    
    Returns:
        Blended reward function
    """
    def reward_fn(sequences: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute blended hybrid rewards."""
        if not sequences:
            return (
                torch.empty(0, device=device),
                torch.empty(0, dtype=torch.bool, device=device)
            )
        
        # Check validity
        valid_mask = validate_sequences(sequences, min_charge=min_charge)
        valid_mask = valid_mask.to(device)
        
        # Get base rewards
        base_rewards, base_mask = base_reward_fn(sequences)
        
        # Get personalized rewards
        with torch.no_grad():
            properties = property_function(sequences)
            persona_rewards = compute_personalized_reward(properties, persona)
        
        # Blend rewards for valid sequences
        blended_rewards = (
            (1 - blend_weight) * base_rewards +
            blend_weight * persona_rewards
        )
        
        # Apply penalty for invalid sequences
        final_rewards = torch.where(
            valid_mask,
            blended_rewards,
            torch.full_like(blended_rewards, penalty)
        )
        
        # Combine masks
        combined_mask = valid_mask & base_mask
        
        return final_rewards, combined_mask
    
    return reward_fn

