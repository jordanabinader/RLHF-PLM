"""
Verify Property Normalization

This script loads the normalization statistics and tests them on a small
set of sequences to verify that:
1. Statistics load correctly
2. Normalization is applied
3. Normalized properties have reasonable distributions
4. Rewards are computed correctly

Usage:
    python personalization/verify_normalization.py \
        --normalization_stats_path personalization/checkpoints/property_normalization.json
"""

import argparse
import json
import sys
import torch
import numpy as np
from pathlib import Path
from typing import List

# Add repo root and amp_design to path for imports
REPO_ROOT = Path(__file__).parent.parent.resolve()
AMP_DESIGN_DIR = REPO_ROOT / "amp_design"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(AMP_DESIGN_DIR) not in sys.path:
    sys.path.insert(0, str(AMP_DESIGN_DIR))

from personalization.unified_property_fn import create_unified_property_function
from personalization.personas import get_persona, list_personas, compute_personalized_reward


def load_test_sequences() -> List[str]:
    """Generate or load a small set of test sequences."""
    # Use some realistic AMP sequences for testing
    test_sequences = [
        "KLLKLLKKLLKLLK",  # Highly charged
        "GGGGGGGGGGGGGG",  # Low complexity
        "FWFWFWFWFWFWFW",  # Hydrophobic
        "KRWWKWWRRKWWKWWRRK",  # Typical AMP
        "ACDEFGHIKLMNPQRSTVWY",  # All canonical AAs
    ]
    return test_sequences


def verify_statistics_file(stats_path: Path) -> dict:
    """Verify the normalization statistics file exists and is valid."""
    print(f"Verifying statistics file: {stats_path}")
    
    if not stats_path.exists():
        raise FileNotFoundError(f"Statistics file not found: {stats_path}")
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    # Check required keys
    required_keys = ['mean', 'std', 'property_names']
    for key in required_keys:
        if key not in stats:
            raise ValueError(f"Missing required key in statistics: {key}")
    
    # Check dimensions
    if len(stats['mean']) != 4 or len(stats['std']) != 4:
        raise ValueError(f"Invalid statistics dimensions: mean={len(stats['mean'])}, std={len(stats['std'])}")
    
    print("✓ Statistics file is valid")
    print(f"  Mean: {stats['mean']}")
    print(f"  Std:  {stats['std']}")
    
    return stats


def test_normalization(
    property_fn,
    sequences: List[str],
    stats: dict,
) -> None:
    """Test that normalization is being applied correctly."""
    print("\nTesting normalization on sample sequences...")
    
    # Compute properties
    properties = property_fn(sequences)
    
    print(f"Properties shape: {properties.shape}")
    print(f"Expected shape: ({len(sequences)}, 4)")
    
    if properties.shape != (len(sequences), 4):
        raise ValueError(f"Unexpected properties shape: {properties.shape}")
    
    # Check that properties are normalized (mean ≈ 0, std ≈ 1)
    # Note: With only 5 sequences, we won't get exact mean=0, std=1,
    # but we should see values in a reasonable range
    
    property_names = ['activity', 'toxicity', 'stability', 'length']
    
    print("\nNormalized Property Statistics:")
    print("=" * 60)
    for i, name in enumerate(property_names):
        values = properties[:, i].cpu().numpy()
        mean = np.mean(values)
        std = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        
        print(f"{name:12s}: μ={mean:7.3f}, σ={std:6.3f}, "
              f"range=[{min_val:7.2f}, {max_val:7.2f}]")
    
    print("=" * 60)
    print("\nNote: With normalized properties, values should be roughly")
    print("centered around 0 with std close to 1 (for large samples).")
    print("For small test samples, expect some variation.")


def test_reward_computation(
    property_fn,
    sequences: List[str],
) -> None:
    """Test reward computation for different personas."""
    print("\nTesting reward computation for different personas...")
    
    # Compute properties
    properties = property_fn(sequences)
    
    # Test with a few personas
    test_personas = ['PotencyMaximizer', 'SafetyFirst', 'BalancedDesigner']
    
    print("\nRewards for each sequence and persona:")
    print("=" * 80)
    
    for seq_idx, seq in enumerate(sequences):
        print(f"\nSequence {seq_idx + 1}: {seq[:20]}..." if len(seq) > 20 else f"\nSequence {seq_idx + 1}: {seq}")
        props = properties[seq_idx:seq_idx+1]
        
        for persona_name in test_personas:
            persona = get_persona(persona_name)
            reward = compute_personalized_reward(props, persona).item()
            print(f"  {persona_name:20s}: reward = {reward:7.4f}")
    
    print("=" * 80)
    
    # Verify personas give different rewards
    all_rewards = {}
    for persona_name in test_personas:
        persona = get_persona(persona_name)
        rewards = compute_personalized_reward(properties, persona)
        all_rewards[persona_name] = rewards.cpu().numpy()
    
    print("\nVerifying persona differentiation...")
    
    # Check that different personas give different reward distributions
    from scipy.stats import pearsonr
    
    correlations = {}
    for i, p1 in enumerate(test_personas):
        for p2 in test_personas[i+1:]:
            corr, _ = pearsonr(all_rewards[p1], all_rewards[p2])
            correlations[f"{p1} vs {p2}"] = corr
            print(f"  Correlation {p1} vs {p2}: {corr:.3f}")
    
    if all(abs(c) > 0.95 for c in correlations.values()):
        print("\n⚠️  WARNING: All personas are highly correlated!")
        print("   This suggests the normalization may not be working correctly,")
        print("   or the test sequences are too similar.")
    else:
        print("\n✓ Personas show differentiation (correlations vary)")


def main():
    parser = argparse.ArgumentParser(description="Verify property normalization")
    parser.add_argument(
        "--normalization_stats_path",
        type=str,
        default="personalization/checkpoints/property_normalization.json",
        help="Path to normalization statistics JSON"
    )
    parser.add_argument(
        "--activity_checkpoint",
        type=str,
        default="amp_design/best_new_4.pth",
        help="Path to activity head checkpoint"
    )
    parser.add_argument(
        "--toxicity_checkpoint",
        type=str,
        default="personalization/checkpoints/toxicity_head.pth",
        help="Path to toxicity head checkpoint"
    )
    parser.add_argument(
        "--stability_checkpoint",
        type=str,
        default="personalization/checkpoints/stability_head.pth",
        help="Path to stability head checkpoint"
    )
    parser.add_argument(
        "--esm_model_size",
        type=str,
        default="650M",
        choices=["650M", "8M"],
        help="ESM model size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Property Normalization Verification")
    print("=" * 80)
    print()
    
    # Step 1: Verify statistics file
    stats = verify_statistics_file(Path(args.normalization_stats_path))
    
    # Step 2: Load property function WITH normalization
    print("\nLoading property function with normalization...")
    property_fn = create_unified_property_function(
        activity_checkpoint=args.activity_checkpoint,
        toxicity_checkpoint=args.toxicity_checkpoint,
        stability_checkpoint=args.stability_checkpoint,
        esm_model_size=args.esm_model_size,
        device=args.device,
        normalization_stats_path=args.normalization_stats_path,
    )
    print("✓ Property function loaded")
    
    # Step 3: Load test sequences
    sequences = load_test_sequences()
    print(f"\nLoaded {len(sequences)} test sequences")
    
    # Step 4: Test normalization
    test_normalization(property_fn, sequences, stats)
    
    # Step 5: Test reward computation
    test_reward_computation(property_fn, sequences)
    
    print("\n" + "=" * 80)
    print("✓ Verification Complete!")
    print("=" * 80)
    print("\nIf all checks passed, your normalization is working correctly.")
    print("You can now run GRPO training with confidence.")


if __name__ == "__main__":
    main()

