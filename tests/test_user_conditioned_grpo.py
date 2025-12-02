"""
Unit tests for user-conditioned GRPO components.

Tests validity constraints, hybrid rewards, and user context projection.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import pytest

from personalization.validity import is_sequence_valid, validate_sequences, calculate_net_charge
from personalization.hybrid_reward import create_hybrid_reward_fn
from personalization.user_conditioned_policy import UserContextProjector
from personalization.personas import get_persona


def test_validity_checker():
    """Test sequence validity constraints"""
    print("\n=== Testing Validity Checker ===")
    
    # Valid sequences
    assert is_sequence_valid("KKLLKWLKKLL"), "Should be valid: standard AMP-like sequence"
    assert is_sequence_valid("GIGKFLHSAKKFGKAFVGEIMNS"), "Should be valid: Magainin 2"
    
    # Invalid: too short
    assert not is_sequence_valid("KKK"), "Should be invalid: too short"
    
    # Invalid: too long
    assert not is_sequence_valid("A" * 60), "Should be invalid: too long"
    
    # Invalid: non-canonical AA
    assert not is_sequence_valid("KKLLBKWL"), "Should be invalid: contains B"
    
    # Invalid: too many repeats
    assert not is_sequence_valid("KKKKKKKKKKKK"), "Should be invalid: >50% repeats"
    
    # Invalid: insufficient charge
    assert not is_sequence_valid("DDDDDDDDDD", min_charge=0), "Should be invalid: negative charge"
    
    print("✓ All validity checks passed")


def test_net_charge_calculation():
    """Test charge calculation"""
    print("\n=== Testing Net Charge Calculation ===")
    
    assert calculate_net_charge("KKKKKK") == 6, "6 lysines = +6"
    assert calculate_net_charge("DDDDDD") == -6, "6 aspartates = -6"
    assert calculate_net_charge("KKDD") == 0, "2K + 2D = 0"
    assert calculate_net_charge("KKRRHH") == 6, "K+R+H all positive"
    assert calculate_net_charge("AAAAAA") == 0, "Alanines are neutral"
    
    print("✓ All charge calculations correct")


def test_validate_sequences_batch():
    """Test batch validation"""
    print("\n=== Testing Batch Validation ===")
    
    sequences = [
        "KKLLKWLKKLL",              # Valid
        "AAA",                       # Too short
        "GIGKFLHSAKKFGKAFVGEIMNS",  # Valid
        "KKXXKKLL",                  # Invalid AA
        "A" * 60,                    # Too long
        "KKKKKKKKKKKKKK",            # Too many repeats
    ]
    
    valid_mask = validate_sequences(sequences)
    
    assert valid_mask[0] == True, "First should be valid"
    assert valid_mask[1] == False, "Second should be invalid (too short)"
    assert valid_mask[2] == True, "Third should be valid"
    assert valid_mask[3] == False, "Fourth should be invalid (bad AA)"
    assert valid_mask[4] == False, "Fifth should be invalid (too long)"
    assert valid_mask[5] == False, "Sixth should be invalid (too many repeats)"
    
    print(f"✓ Batch validation correct: {valid_mask.sum()}/6 valid")


def test_hybrid_reward_function():
    """Test hybrid reward computation"""
    print("\n=== Testing Hybrid Reward Function ===")
    
    try:
        from personalization.unified_property_fn import create_unified_property_function
        
        # Create property function
        property_fn = create_unified_property_function(
            activity_checkpoint="amp_design/best_new_4.pth",
            toxicity_checkpoint="personalization/checkpoints/toxicity_head.pth",
            stability_checkpoint="personalization/checkpoints/stability_head.pth",
            device="cpu",
        )
        
        # Create hybrid reward
        persona = get_persona("BalancedDesigner")
        reward_fn = create_hybrid_reward_fn(
            property_function=property_fn,
            persona=persona,
            penalty=-10.0,
            device="cpu",
        )
        
        # Test sequences
        sequences = [
            "GIGKFLHSAKKFGKAFVGEIMNS",  # Valid AMP
            "AAA",                       # Invalid (too short)
        ]
        
        rewards, valid_mask = reward_fn(sequences)
        
        # Check that invalid sequence gets penalty
        assert rewards[1].item() == -10.0, f"Invalid sequence should get penalty, got {rewards[1].item()}"
        assert valid_mask[1] == False, "Invalid sequence should have False mask"
        
        # Check that valid sequence gets personalized reward
        assert rewards[0].item() > -10.0, f"Valid sequence should get >-10 reward, got {rewards[0].item()}"
        assert valid_mask[0] == True, "Valid sequence should have True mask"
        
        print(f"✓ Hybrid reward function working:")
        print(f"  Valid sequence reward: {rewards[0].item():.3f}")
        print(f"  Invalid sequence reward: {rewards[1].item():.3f}")
        
    except FileNotFoundError as e:
        print(f"⚠ Skipping hybrid reward test (checkpoint not found): {e}")
        pytest.skip(f"Checkpoint not found: {e}")


def test_user_context_projector():
    """Test user context projection"""
    print("\n=== Testing User Context Projector ===")
    
    projector = UserContextProjector(user_dim=4, output_dim=256)
    
    # Single user context
    user_weights = torch.tensor([1.0, -0.5, 0.3, -0.1])
    projected = projector(user_weights)
    
    assert projected.shape == (256,), f"Expected (256,), got {projected.shape}"
    
    # Batch of user contexts
    user_weights_batch = torch.tensor([
        [1.0, -0.5, 0.3, -0.1],
        [0.5, -1.0, 0.5, -0.2],
    ])
    projected_batch = projector(user_weights_batch)
    
    assert projected_batch.shape == (2, 256), f"Expected (2, 256), got {projected_batch.shape}"
    
    print("✓ User context projector working correctly")


def test_persona_weight_differences():
    """Test that different personas have different weights"""
    print("\n=== Testing Persona Weights ===")
    
    personas = ["PotencyMaximizer", "SafetyFirst", "BalancedDesigner"]
    
    weights = [get_persona(p).weights for p in personas]
    
    # All should be different
    for i in range(len(weights)):
        for j in range(i+1, len(weights)):
            assert not torch.allclose(weights[i], weights[j]), \
                f"{personas[i]} and {personas[j]} have identical weights"
    
    print("✓ All personas have unique weights:")
    for p, w in zip(personas, weights):
        print(f"  {p}: {w.tolist()}")


def test_persona_list():
    """Test that personas can be listed"""
    print("\n=== Testing Persona Listing ===")
    
    from personalization.personas import list_personas
    
    persona_names = list_personas()
    assert len(persona_names) > 0, "Should have at least one persona"
    assert "BalancedDesigner" in persona_names, "Should include BalancedDesigner"
    
    print(f"✓ Found {len(persona_names)} personas: {', '.join(persona_names)}")


if __name__ == "__main__":
    print("=" * 80)
    print("Running User-Conditioned GRPO Unit Tests")
    print("=" * 80)
    
    test_validity_checker()
    test_net_charge_calculation()
    test_validate_sequences_batch()
    test_user_context_projector()
    test_persona_weight_differences()
    test_persona_list()
    test_hybrid_reward_function()
    
    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)

