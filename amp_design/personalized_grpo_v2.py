"""
Personalized GRPO for AMP Design using Shared Property Model Architecture

This module integrates the new personalized RLHF architecture:
- Unified property function g(x) = [p_act, p_tox, p_stab, p_len]
- Lightweight persona weights w^(u)
- Simple dot product for rewards: R^(u)(x) = w^(u)^T · g(x)

No more per-user model training! Just load property heads once and use different weight vectors.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from personalization.unified_property_fn import create_unified_property_function, UnifiedPropertyFunction
from personalization.personas import get_persona, list_personas, compute_personalized_reward, Persona
from utils import clean_sequences


def create_personalized_reward_fn(
    property_function: UnifiedPropertyFunction,
    persona: Persona,
    device: str = "cuda",
):
    """
    Create a personalized reward function for GRPO/PPO training.
    
    Args:
        property_function: Unified property function g(x)
        persona: Persona defining user preferences w^(u)
        device: Device for computation
    
    Returns:
        Reward function compatible with GRPO/PPO
    """
    def reward_fn(sequences: list[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute personalized rewards for sequences.
        
        Args:
            sequences: List of protein sequences
        
        Returns:
            rewards: Tensor of shape (batch_size,)
            mask: Boolean tensor of shape (batch_size,) - all True
        """
        # Clean sequences
        sequences = clean_sequences(sequences)
        
        # Compute properties g(x)
        with torch.no_grad():
            properties = property_function(sequences)  # (batch_size, 4)
        
        # Compute personalized reward R^(u)(x) = w^(u)^T · g(x)
        rewards = compute_personalized_reward(properties, persona)
        
        # Create mask (all sequences are valid)
        mask = torch.ones(len(sequences), dtype=torch.bool, device=device)
        
        return rewards, mask
    
    return reward_fn


def create_blended_reward_fn(
    property_function: UnifiedPropertyFunction,
    persona: Persona,
    base_reward_fn,
    blend_weight: float = 0.5,
    device: str = "cuda",
):
    """
    Create a reward function that blends base rewards with personalized rewards.
    
    Useful for gradually introducing personalization:
    R_total = (1 - α) * R_base + α * R_personalized
    
    Args:
        property_function: Unified property function
        persona: Persona for personalization
        base_reward_fn: Original reward function (e.g., classifier-based)
        blend_weight: Weight for personalized rewards (0 = all base, 1 = all personalized)
        device: Device for computation
    
    Returns:
        Blended reward function
    """
    personalized_fn = create_personalized_reward_fn(property_function, persona, device)
    
    def blended_fn(sequences: list[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute blended rewards."""
        # Get base rewards
        base_rewards, base_mask = base_reward_fn(sequences)
        
        # Get personalized rewards
        personalized_rewards, personalized_mask = personalized_fn(sequences)
        
        # Blend rewards
        blended_rewards = (
            (1 - blend_weight) * base_rewards +
            blend_weight * personalized_rewards
        )
        
        # Combine masks
        combined_mask = base_mask & personalized_mask
        
        return blended_rewards, combined_mask
    
    return blended_fn


def analyze_sequences_with_persona(
    sequences: list[str],
    property_function: UnifiedPropertyFunction,
    persona_name: str,
):
    """
    Analyze how a persona would rank a set of sequences.
    
    Args:
        sequences: List of sequences to analyze
        property_function: Unified property function
        persona_name: Name of persona to use
    
    Returns:
        DataFrame with sequences, properties, and rewards
    """
    import pandas as pd
    
    # Get persona
    persona = get_persona(persona_name)
    
    # Compute properties
    with torch.no_grad():
        properties = property_function(sequences)
        properties_np = properties.cpu().numpy()
    
    # Compute rewards
    rewards = compute_personalized_reward(properties, persona)
    rewards_np = rewards.cpu().numpy()
    
    # Build DataFrame
    df = pd.DataFrame({
        'sequence': sequences,
        'activity': properties_np[:, 0],
        'toxicity': properties_np[:, 1],
        'stability': properties_np[:, 2],
        'length_norm': properties_np[:, 3],
        'reward': rewards_np,
    })
    
    # Sort by reward
    df = df.sort_values('reward', ascending=False)
    df = df.reset_index(drop=True)
    
    return df


def compare_personas_on_sequences(
    sequences: list[str],
    property_function: UnifiedPropertyFunction,
    persona_names: Optional[list[str]] = None,
):
    """
    Compare how different personas rank the same sequences.
    
    Args:
        sequences: List of sequences
        property_function: Unified property function
        persona_names: List of persona names (default: all)
    
    Returns:
        DataFrame with reward columns for each persona
    """
    import pandas as pd
    from personalization.personas import compute_multi_persona_rewards
    
    if persona_names is None:
        persona_names = list_personas()
    
    # Compute properties once
    with torch.no_grad():
        properties = property_function(sequences)
    
    # Compute rewards for all personas
    rewards_dict = compute_multi_persona_rewards(properties, persona_names)
    
    # Build DataFrame
    df = pd.DataFrame({'sequence': sequences})
    
    for name, rewards in rewards_dict.items():
        df[f'{name}_reward'] = rewards.cpu().numpy()
    
    return df


def main():
    """Demo: Setup personalized reward function."""
    parser = argparse.ArgumentParser(
        description="Personalized AMP Design with Shared Property Model"
    )
    parser.add_argument(
        "--activity-checkpoint",
        type=str,
        default="amp_design/best_new_4.pth",
        help="Path to activity classifier checkpoint"
    )
    parser.add_argument(
        "--toxicity-checkpoint",
        type=str,
        default="personalization/checkpoints/toxicity_head.pth",
        help="Path to toxicity head checkpoint"
    )
    parser.add_argument(
        "--stability-checkpoint",
        type=str,
        default="personalization/checkpoints/stability_head.pth",
        help="Path to stability head checkpoint"
    )
    parser.add_argument(
        "--persona",
        type=str,
        default="BalancedDesigner",
        choices=list_personas(),
        help="Persona to use for reward computation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo with example sequences"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Personalized AMP Design - Shared Property Model")
    print("=" * 80)
    
    # Create unified property function
    print("\nLoading property function...")
    property_fn = create_unified_property_function(
        activity_checkpoint=args.activity_checkpoint,
        toxicity_checkpoint=args.toxicity_checkpoint,
        stability_checkpoint=args.stability_checkpoint,
        esm_model_size="650M",
        device=args.device,
        max_length=100,
    )
    
    # Get persona
    print(f"\nUsing persona: {args.persona}")
    persona = get_persona(args.persona)
    print(persona.explain())
    
    # Create reward function
    print("\nCreating personalized reward function...")
    reward_fn = create_personalized_reward_fn(property_fn, persona, args.device)
    
    if args.demo:
        print("\n" + "=" * 80)
        print("Demo: Analyzing Example Sequences")
        print("=" * 80)
        
        # Example sequences
        sequences = [
            "GIGKFLHSAKKFGKAFVGEIMNS",  # Classic AMP
            "KKLLPIVKKK",                 # Short cationic
            "AAAAGGGGAAAA",               # Not AMP-like
            "DDDDDEEEDD",                 # Acidic
            "KWWKWWKKWW",                 # Hydrophobic + cationic
        ]
        
        print(f"\nAnalyzing {len(sequences)} sequences with {args.persona}...")
        df = analyze_sequences_with_persona(sequences, property_fn, args.persona)
        
        print("\nResults (sorted by reward):")
        print(df.to_string(index=False))
        
        # Compare with other personas
        print("\n" + "=" * 80)
        print("Comparing Multiple Personas")
        print("=" * 80)
        
        comparison_personas = ["PotencyMaximizer", "SafetyFirst", "BalancedDesigner"]
        df_compare = compare_personas_on_sequences(sequences, property_fn, comparison_personas)
        
        print("\nReward comparison across personas:")
        print(df_compare.to_string(index=False))
    
    print("\n✓ Personalized reward function ready for GRPO/PPO training!")
    print(f"  Use: reward_fn(sequences) -> (rewards, mask)")


if __name__ == "__main__":
    main()

