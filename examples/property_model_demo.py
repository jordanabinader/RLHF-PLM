"""
Complete Demo: Shared Property Model for Personalized AMP Design

This script demonstrates the complete workflow:
1. Setup unified property function
2. Define/select personas
3. Compute properties and rewards
4. Compare different personas
5. Explain individual decisions
"""

import sys
from pathlib import Path
import torch
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from personalization import (
    create_unified_property_function,
    get_persona,
    list_personas,
    compute_personalized_reward,
    explain_reward,
    create_custom_persona,
)


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


def demo_property_function():
    """Demo 1: Create and use property function."""
    print_section("Demo 1: Property Function Setup")
    
    print("Creating unified property function...")
    print("  - Activity: Using best_new_4.pth (pre-trained AMP classifier)")
    print("  - Toxicity: Using toxicity_head.pth (trained on ToxDL2)")
    print("  - Stability: Using stability_head.pth (placeholder or trained)")
    print("  - Length: Computed directly from sequence")
    
    try:
        property_fn = create_unified_property_function(
            activity_checkpoint="amp_design/best_new_4.pth",
            toxicity_checkpoint="personalization/checkpoints/toxicity_head.pth",
            stability_checkpoint="personalization/checkpoints/stability_head.pth",
            esm_model_size="650M",
            device="cuda" if torch.cuda.is_available() else "cpu",
            max_length=100,
        )
        print("✓ Property function created successfully!\n")
        return property_fn
    
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("\nPlease ensure checkpoints exist:")
        print("  1. amp_design/best_new_4.pth (should already exist)")
        print("  2. Run: python personalization/train_toxicity.py")
        print("  3. Run: python personalization/train_stability.py --mode placeholder")
        sys.exit(1)


def demo_personas():
    """Demo 2: Show available personas."""
    print_section("Demo 2: Available Personas")
    
    print(f"Found {len(list_personas())} pre-defined personas:\n")
    
    for name in list_personas():
        persona = get_persona(name)
        print(f"• {name}")
        print(f"  {persona.description}")
        weights = persona.get_weight_dict()
        print(f"  Weights: act={weights['activity']:.1f}, tox={weights['toxicity']:.1f}, "
              f"stab={weights['stability']:.1f}, len={weights['length']:.1f}\n")


def demo_property_computation(property_fn):
    """Demo 3: Compute properties for example sequences."""
    print_section("Demo 3: Property Computation")
    
    # Example AMP sequences
    sequences = [
        "GIGKFLHSAKKFGKAFVGEIMNS",  # Classic AMP (Magainin-like)
        "KKLLPIVKKK",                 # Short cationic AMP
        "AAAAGGGGAAAA",               # Non-AMP control
        "KWWKWWKKWW",                 # Hydrophobic + cationic
        "DDDDDEEEDD",                 # Acidic (unlikely AMP)
    ]
    
    print("Computing properties for 5 example sequences...\n")
    
    with torch.no_grad():
        properties = property_fn(sequences)
    
    # Convert to DataFrame for nice display
    df = pd.DataFrame({
        'sequence': sequences,
        'activity': properties[:, 0].cpu().numpy(),
        'toxicity': properties[:, 1].cpu().numpy(),
        'stability': properties[:, 2].cpu().numpy(),
        'length_norm': properties[:, 3].cpu().numpy(),
    })
    
    print(df.to_string(index=False))
    print(f"\nProperty vector shape: {properties.shape}")
    print("✓ Properties computed successfully!\n")
    
    return sequences, properties


def demo_personalized_rewards(sequences, properties):
    """Demo 4: Compute rewards for different personas."""
    print_section("Demo 4: Personalized Rewards")
    
    # Select 3 personas with different preferences
    persona_names = ["PotencyMaximizer", "SafetyFirst", "BalancedDesigner"]
    
    print(f"Computing rewards for {len(persona_names)} personas...\n")
    
    results = {'sequence': sequences}
    
    for name in persona_names:
        persona = get_persona(name)
        rewards = compute_personalized_reward(properties, persona)
        results[f'{name}_reward'] = rewards.cpu().numpy()
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    print("\nObservations:")
    print("  • PotencyMaximizer: Highest rewards for sequences with high activity")
    print("  • SafetyFirst: Penalizes toxic sequences heavily")
    print("  • BalancedDesigner: Balanced trade-off between properties")
    print("\n✓ Different personas rank sequences differently!")


def demo_interpretability(sequences, properties):
    """Demo 5: Explain individual rewards."""
    print_section("Demo 5: Interpretability - Explain Rewards")
    
    # Take first sequence
    seq = sequences[0]
    prop = properties[0]
    
    print(f"Explaining why SafetyFirst persona ranks this sequence:\n")
    
    persona = get_persona("SafetyFirst")
    explanation = explain_reward(prop, persona, seq)
    
    print(explanation)
    
    print("\n✓ Reward is fully interpretable as weighted combination of properties!")


def demo_custom_persona(sequences, properties):
    """Demo 6: Create and use custom persona."""
    print_section("Demo 6: Custom Persona Creation")
    
    print("Creating custom persona 'TherapeuticFocused'...")
    print("  Requirements:")
    print("    - High activity (0.7)")
    print("    - Very low toxicity (-0.9)")
    print("    - High stability (0.8)")
    print("    - Moderate length preference (-0.3)\n")
    
    custom_persona = create_custom_persona(
        name="TherapeuticFocused",
        activity_weight=0.7,
        toxicity_weight=-0.9,
        stability_weight=0.8,
        length_weight=-0.3,
        description="Optimized for therapeutic applications"
    )
    
    print(custom_persona.explain())
    
    # Compute rewards
    rewards = compute_personalized_reward(properties, custom_persona)
    
    print("\nRewards for custom persona:")
    for seq, reward in zip(sequences, rewards):
        print(f"  {seq[:20]:20s} ... reward = {reward.item():.4f}")
    
    print("\n✓ Easy to create custom personas for specific needs!")


def demo_comparison_table(sequences, properties):
    """Demo 7: Comprehensive comparison table."""
    print_section("Demo 7: Comprehensive Persona Comparison")
    
    # All pre-defined personas
    persona_names = list_personas()
    
    print(f"Comparing all {len(persona_names)} personas on {len(sequences)} sequences...\n")
    
    results = []
    
    for seq, prop in zip(sequences, properties):
        row = {
            'sequence': seq[:25] + '...' if len(seq) > 25 else seq,
            'activity': f"{prop[0].item():.3f}",
            'toxicity': f"{prop[1].item():.3f}",
            'stability': f"{prop[2].item():.3f}",
        }
        
        for name in persona_names:
            persona = get_persona(name)
            reward = compute_personalized_reward(prop.unsqueeze(0), persona)
            row[name] = f"{reward.item():.3f}"
        
        results.append(row)
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    print("\n✓ Easy to compare how different users would rank sequences!")


def main():
    """Run all demos."""
    print("=" * 80)
    print("Shared Property Model Demo")
    print("Personalized RLHF for AMP Design")
    print("=" * 80)
    
    # Demo 1: Setup
    property_fn = demo_property_function()
    
    # Demo 2: Show personas
    demo_personas()
    
    # Demo 3: Compute properties
    sequences, properties = demo_property_computation(property_fn)
    
    # Demo 4: Personalized rewards
    demo_personalized_rewards(sequences, properties)
    
    # Demo 5: Interpretability
    demo_interpretability(sequences, properties)
    
    # Demo 6: Custom persona
    demo_custom_persona(sequences, properties)
    
    # Demo 7: Comparison table
    demo_comparison_table(sequences, properties)
    
    # Summary
    print_section("Summary")
    print("✓ Unified property function computes g(x) = [p_act, p_tox, p_stab, p_len]")
    print("✓ Personas are just 4-dimensional weight vectors w^(u)")
    print("✓ Personalized rewards: R^(u)(x) = w^(u)^T · g(x) (simple dot product!)")
    print("✓ Fully interpretable: can explain every reward")
    print("✓ Efficient: train property models once, define users with 4 weights")
    print("\nNext steps:")
    print("  1. Integrate with GRPO: Use create_personalized_reward_fn()")
    print("  2. Train with different personas")
    print("  3. Compare generated sequences across personas")
    print("  4. Create custom personas for your specific needs")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

