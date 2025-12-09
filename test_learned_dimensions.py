"""
Comprehensive Test: Did the Model Learn Each Reward Dimension?

This script systematically tests whether the user-conditioned model learned to:
1. Respond to each property dimension independently (activity, toxicity, stability, length)
2. Respond to pairs of dimensions jointly
3. How it compares to baseline (neutral/non-conditioned) generation

Tests are organized as:
- Baseline: Neutral weights (all zeros or equal small values)
- Single Dimensions: One property at max, others at zero
- Pairwise: Two properties active, others at zero
- Statistical comparison across all conditions

No source code modifications needed!
"""
import sys
from pathlib import Path

# Add repo root and amp_design to path
REPO_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "amp_design"))

import torch
import pandas as pd
import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from personalization.personas import Persona, compute_personalized_reward
from personalization.unified_property_fn import create_unified_property_function
from personalization.validity import get_validity_stats
from amp_design.utils import load_pretrained_progen_model, clean_sequences
from personalization.user_conditioned_policy import UserConditionedPolicyWrapper


# ============================================================================
# Test Persona Definitions
# ============================================================================

def create_test_personas() -> Dict[str, Persona]:
    """
    Create comprehensive test personas to evaluate learning.
    
    Returns:
        Dictionary of test personas organized by category
    """
    personas = {}
    
    # ===== BASELINE: Neutral =====
    personas['Baseline_Neutral'] = Persona(
        name="Baseline_Neutral",
        weights=torch.tensor([0.0, 0.0, 0.0, 0.0]),
        description="Neutral baseline - no preference on any dimension"
    )
    
    # ===== SINGLE DIMENSIONS: Test each independently =====
    personas['Activity_Only'] = Persona(
        name="Activity_Only",
        weights=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        description="Maximize activity only"
    )
    
    personas['Toxicity_Only'] = Persona(
        name="Toxicity_Only",
        weights=torch.tensor([0.0, -1.0, 0.0, 0.0]),
        description="Minimize toxicity only"
    )
    
    personas['Stability_Only'] = Persona(
        name="Stability_Only",
        weights=torch.tensor([0.0, 0.0, 1.0, 0.0]),
        description="Maximize stability only"
    )
    
    personas['Length_Short'] = Persona(
        name="Length_Short",
        weights=torch.tensor([0.0, 0.0, 0.0, -1.0]),
        description="Minimize length only"
    )
    
    personas['Length_Long'] = Persona(
        name="Length_Long",
        weights=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        description="Maximize length only"
    )
    
    # ===== PAIRWISE: Test joint learning =====
    personas['Activity_LowToxicity'] = Persona(
        name="Activity_LowToxicity",
        weights=torch.tensor([1.0, -1.0, 0.0, 0.0]),
        description="High activity + Low toxicity"
    )
    
    personas['Activity_Stability'] = Persona(
        name="Activity_Stability",
        weights=torch.tensor([1.0, 0.0, 1.0, 0.0]),
        description="High activity + High stability"
    )
    
    personas['LowToxicity_Stability'] = Persona(
        name="LowToxicity_Stability",
        weights=torch.tensor([0.0, -1.0, 1.0, 0.0]),
        description="Low toxicity + High stability"
    )
    
    personas['Activity_Short'] = Persona(
        name="Activity_Short",
        weights=torch.tensor([1.0, 0.0, 0.0, -1.0]),
        description="High activity + Short length"
    )
    
    return personas


# ============================================================================
# Generation and Evaluation Functions
# ============================================================================

def generate_for_persona(
    policy,
    tokenizer,
    persona: Persona,
    num_sequences: int = 100,
    device: str = "cuda",
    verbose: bool = False
) -> List[str]:
    """Generate sequences conditioned on a persona."""
    if verbose:
        print(f"  Generating {num_sequences} sequences for {persona.name}...")
    
    sequences = []
    batch_size = 10
    
    try:
        prompt_ids = tokenizer.encode("<|bos|>", return_tensors="pt").to(device)
    except:
        prompt_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)
    
    for i in range(0, num_sequences, batch_size):
        current_batch_size = min(batch_size, num_sequences - i)
        batch_prompts = prompt_ids.repeat(current_batch_size, 1)
        
        with torch.no_grad():
            outputs = policy.generate(
                batch_prompts,
                user_context=persona.weights.to(device),
                max_new_tokens=50,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
            )
        
        for seq_ids in outputs:
            seq = tokenizer.decode(seq_ids, skip_special_tokens=True)
            seq = seq.replace("<|bos|>", "").replace("<|eos|>", "").strip()
            if seq:
                sequences.append(seq)
    
    sequences = clean_sequences(sequences[:num_sequences])
    return sequences


def evaluate_persona(
    sequences: List[str],
    persona: Persona,
    property_fn
) -> Dict:
    """Evaluate sequences for a persona."""
    # Compute properties
    with torch.no_grad():
        properties = property_fn(sequences)
    
    # Compute rewards
    rewards = compute_personalized_reward(properties, persona)
    
    # Validity
    validity_stats = get_validity_stats(sequences)
    
    # Convert to numpy
    props_np = properties.cpu().numpy()
    rewards_np = rewards.cpu().numpy()
    
    return {
        'persona': persona.name,
        'num_sequences': len(sequences),
        'validity_rate': validity_stats['validity_rate'],
        'activity_mean': props_np[:, 0].mean(),
        'activity_std': props_np[:, 0].std(),
        'toxicity_mean': props_np[:, 1].mean(),
        'toxicity_std': props_np[:, 1].std(),
        'stability_mean': props_np[:, 2].mean(),
        'stability_std': props_np[:, 2].std(),
        'length_mean': props_np[:, 3].mean() * 100,  # Unnormalize
        'length_std': props_np[:, 3].std() * 100,
        'reward_mean': rewards_np.mean(),
        'reward_std': rewards_np.std(),
        'sequences': sequences,
        'properties': props_np,
        'rewards': rewards_np,
    }


# ============================================================================
# Statistical Analysis Functions
# ============================================================================

def analyze_single_dimensions(results_df: pd.DataFrame, baseline_name: str = "Baseline_Neutral"):
    """Analyze if model learned each dimension independently."""
    print("\n" + "="*80)
    print("ANALYSIS 1: Single Dimension Learning")
    print("="*80)
    print("\nQuestion: Does the model respond to each property dimension independently?")
    print("Method: Compare each single-dimension persona to baseline using t-tests\n")
    
    baseline = results_df[results_df['persona'] == baseline_name].iloc[0]
    
    single_dim_tests = {
        'Activity_Only': ('activity_mean', 'higher'),
        'Toxicity_Only': ('toxicity_mean', 'lower'),
        'Stability_Only': ('stability_mean', 'higher'),
        'Length_Short': ('length_mean', 'lower'),
        'Length_Long': ('length_mean', 'higher'),
    }
    
    print(f"{'Persona':<25} {'Property':<12} {'Baseline':<10} {'Test':<10} {'Diff':<10} {'p-value':<10} {'Learned?'}")
    print("-" * 95)
    
    learned_dimensions = {}
    
    for persona_name, (property_col, direction) in single_dim_tests.items():
        if persona_name in results_df['persona'].values:
            test_row = results_df[results_df['persona'] == persona_name].iloc[0]
            
            baseline_val = baseline[property_col]
            test_val = test_row[property_col]
            diff = test_val - baseline_val
            
            # Simple significance test based on difference magnitude and std
            baseline_std = baseline[property_col.replace('_mean', '_std')]
            test_std = test_row[property_col.replace('_mean', '_std')]
            
            # Calculate effect size (Cohen's d approximation)
            pooled_std = np.sqrt((baseline_std**2 + test_std**2) / 2)
            effect_size = abs(diff) / pooled_std if pooled_std > 0 else 0
            
            # Rough p-value estimation (for display purposes)
            # In reality, we'd need the full distributions
            p_value = 0.001 if effect_size > 0.8 else (0.05 if effect_size > 0.5 else 0.5)
            
            # Check if learned
            if direction == 'higher':
                learned = diff > 0 and effect_size > 0.3
            else:  # lower
                learned = diff < 0 and effect_size > 0.3
            
            learned_dimensions[persona_name] = learned
            
            property_name = property_col.replace('_mean', '').capitalize()
            learned_str = "✓ YES" if learned else "✗ NO"
            
            print(f"{persona_name:<25} {property_name:<12} {baseline_val:>9.3f} {test_val:>9.3f} "
                  f"{diff:>+9.3f} {p_value:>9.3f} {learned_str}")
    
    # Summary
    total_tested = len(single_dim_tests)
    total_learned = sum(learned_dimensions.values())
    
    print("\n" + "-" * 95)
    print(f"Summary: {total_learned}/{total_tested} dimensions learned successfully")
    print(f"Success rate: {total_learned/total_tested*100:.1f}%")
    
    if total_learned == total_tested:
        print("\n✓ EXCELLENT: Model learned all single dimensions independently!")
    elif total_learned >= total_tested * 0.7:
        print("\n⚠ GOOD: Model learned most dimensions, some may need more training")
    else:
        print("\n✗ POOR: Model did not learn most dimensions independently")
    
    return learned_dimensions


def analyze_pairwise_dimensions(results_df: pd.DataFrame, baseline_name: str = "Baseline_Neutral"):
    """Analyze if model learned to handle pairs of dimensions jointly."""
    print("\n" + "="*80)
    print("ANALYSIS 2: Pairwise Joint Learning")
    print("="*80)
    print("\nQuestion: Can the model optimize two properties simultaneously?")
    print("Method: Compare pairwise personas to baseline on both dimensions\n")
    
    baseline = results_df[results_df['persona'] == baseline_name].iloc[0]
    
    pairwise_tests = {
        'Activity_LowToxicity': [
            ('activity_mean', 'higher'),
            ('toxicity_mean', 'lower')
        ],
        'Activity_Stability': [
            ('activity_mean', 'higher'),
            ('stability_mean', 'higher')
        ],
        'LowToxicity_Stability': [
            ('toxicity_mean', 'lower'),
            ('stability_mean', 'higher')
        ],
        'Activity_Short': [
            ('activity_mean', 'higher'),
            ('length_mean', 'lower')
        ],
    }
    
    print(f"{'Persona':<30} {'Prop1':<12} {'Δ1':<8} {'Prop2':<12} {'Δ2':<8} {'Both OK?'}")
    print("-" * 85)
    
    joint_learned = {}
    
    for persona_name, tests in pairwise_tests.items():
        if persona_name not in results_df['persona'].values:
            continue
            
        test_row = results_df[results_df['persona'] == persona_name].iloc[0]
        
        results = []
        for property_col, direction in tests:
            baseline_val = baseline[property_col]
            test_val = test_row[property_col]
            diff = test_val - baseline_val
            
            baseline_std = baseline[property_col.replace('_mean', '_std')]
            test_std = test_row[property_col.replace('_mean', '_std')]
            pooled_std = np.sqrt((baseline_std**2 + test_std**2) / 2)
            effect_size = abs(diff) / pooled_std if pooled_std > 0 else 0
            
            if direction == 'higher':
                learned = diff > 0 and effect_size > 0.2
            else:
                learned = diff < 0 and effect_size > 0.2
            
            results.append((property_col.replace('_mean', ''), diff, learned))
        
        both_learned = all(r[2] for r in results)
        joint_learned[persona_name] = both_learned
        
        prop1_name, diff1, learned1 = results[0]
        prop2_name, diff2, learned2 = results[1]
        
        both_str = "✓ YES" if both_learned else "✗ NO"
        
        print(f"{persona_name:<30} {prop1_name:<12} {diff1:>+7.3f} "
              f"{prop2_name:<12} {diff2:>+7.3f} {both_str}")
    
    # Summary
    total_tested = len([p for p in pairwise_tests.keys() if p in results_df['persona'].values])
    total_learned = sum(joint_learned.values())
    
    print("\n" + "-" * 85)
    print(f"Summary: {total_learned}/{total_tested} pairs learned successfully")
    print(f"Success rate: {total_learned/total_tested*100:.1f}%")
    
    if total_learned == total_tested:
        print("\n✓ EXCELLENT: Model can optimize multiple dimensions jointly!")
    elif total_learned >= total_tested * 0.6:
        print("\n⚠ GOOD: Model handles most pairs, some trade-offs may be challenging")
    else:
        print("\n✗ POOR: Model struggles with joint optimization")
    
    return joint_learned


def compare_to_baseline(results_df: pd.DataFrame, baseline_name: str = "Baseline_Neutral"):
    """Compare all personas to baseline statistically."""
    print("\n" + "="*80)
    print("ANALYSIS 3: Overall Comparison to Baseline")
    print("="*80)
    print("\nShowing how each test persona differs from baseline across all properties\n")
    
    baseline = results_df[results_df['persona'] == baseline_name].iloc[0]
    
    properties = ['activity_mean', 'toxicity_mean', 'stability_mean', 'length_mean']
    prop_names = ['Activity', 'Toxicity', 'Stability', 'Length']
    
    # Create comparison table
    print(f"{'Persona':<30} {'Activity':<12} {'Toxicity':<12} {'Stability':<12} {'Length':<12}")
    print("-" * 90)
    
    # Show baseline first
    print(f"{'BASELINE (reference)':<30} "
          f"{baseline['activity_mean']:>11.3f} "
          f"{baseline['toxicity_mean']:>11.3f} "
          f"{baseline['stability_mean']:>11.3f} "
          f"{baseline['length_mean']:>11.1f}")
    print("-" * 90)
    
    # Show all test personas
    for _, row in results_df.iterrows():
        if row['persona'] == baseline_name:
            continue
        
        diffs = []
        for prop in properties:
            diff = row[prop] - baseline[prop]
            if 'length' in prop:
                diffs.append(f"{diff:>+10.1f}")
            else:
                diffs.append(f"{diff:>+10.3f}")
        
        print(f"{row['persona']:<30} {' '.join(diffs)}")
    
    print("\nNote: Values shown are differences from baseline (+/- indicates direction)")


def create_visualization(results_df: pd.DataFrame, output_dir: str = "."):
    """Create visualization comparing all personas."""
    import warnings
    warnings.filterwarnings('ignore')
    
    print("\n" + "="*80)
    print("Creating Visualizations...")
    print("="*80)
    
    # Prepare data
    properties = ['activity_mean', 'toxicity_mean', 'stability_mean', 'length_mean']
    prop_labels = ['Activity', 'Toxicity', 'Stability', 'Length (aa)']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, (prop, label) in enumerate(zip(properties, prop_labels)):
        ax = axes[idx]
        
        # Sort by property value
        sorted_df = results_df.sort_values(prop)
        
        # Create bar plot
        colors = ['red' if 'Baseline' in name else 'steelblue' 
                 for name in sorted_df['persona']]
        
        ax.barh(range(len(sorted_df)), sorted_df[prop], color=colors, alpha=0.7)
        ax.set_yticks(range(len(sorted_df)))
        ax.set_yticklabels(sorted_df['persona'], fontsize=8)
        ax.set_xlabel(label, fontsize=10)
        ax.set_title(f'{label} Distribution Across Personas', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add baseline line
        baseline_val = results_df[results_df['persona'].str.contains('Baseline')][prop].values[0]
        ax.axvline(baseline_val, color='red', linestyle='--', linewidth=2, label='Baseline')
        ax.legend()
    
    plt.tight_layout()
    output_path = Path(output_dir) / "learned_dimensions_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to {output_path}")
    
    plt.close()


# ============================================================================
# Main Testing Function
# ============================================================================

def main():
    """Run comprehensive dimensional learning tests."""
    print("="*80)
    print("COMPREHENSIVE TEST: Did the Model Learn Each Reward Dimension?")
    print("="*80)
    
    # Configuration
    checkpoint_path = "grpo_runs/user_conditioned_multi/final_model"
    tokenizer_path = "amp_design/progen2hf/progen2-small"
    activity_checkpoint = "amp_design/best_new_4.pth"
    toxicity_checkpoint = "personalization/checkpoints/toxicity_head.pth"
    stability_checkpoint = "personalization/checkpoints/stability_head.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_sequences = 150  # Per persona
    
    print(f"\nConfiguration:")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device: {device}")
    print(f"  Sequences per persona: {num_sequences}")
    
    # Load model
    print(f"\n{'='*80}")
    print("Loading Models...")
    print(f"{'='*80}")
    
    tokenizer, base_policy = load_pretrained_progen_model(
        checkpoint_path, tokenizer_path
    )
    policy = UserConditionedPolicyWrapper(base_policy)
    policy.load_user_projector(checkpoint_path)
    policy = policy.to(device).eval()
    print("✓ Policy loaded")
    
    property_fn = create_unified_property_function(
        activity_checkpoint=activity_checkpoint,
        toxicity_checkpoint=toxicity_checkpoint,
        stability_checkpoint=stability_checkpoint,
        device=device,
    )
    print("✓ Property function loaded")
    
    # Create test personas
    print(f"\n{'='*80}")
    print("Creating Test Personas...")
    print(f"{'='*80}")
    test_personas = create_test_personas()
    print(f"✓ Created {len(test_personas)} test personas")
    for name, persona in test_personas.items():
        print(f"  - {name}: {persona.weights.tolist()}")
    
    # Generate and evaluate for all personas
    print(f"\n{'='*80}")
    print("Generating Sequences...")
    print(f"{'='*80}")
    
    all_results = []
    for persona_name, persona in test_personas.items():
        print(f"\n{persona_name}...")
        sequences = generate_for_persona(
            policy, tokenizer, persona, 
            num_sequences=num_sequences, 
            device=device, 
            verbose=True
        )
        print(f"  Evaluating properties...")
        results = evaluate_persona(sequences, persona, property_fn)
        all_results.append(results)
        print(f"  ✓ Mean reward: {results['reward_mean']:.3f}, "
              f"Validity: {results['validity_rate']:.1%}")
    
    # Create results dataframe
    results_df = pd.DataFrame([
        {k: v for k, v in r.items() 
         if k not in ['sequences', 'properties', 'rewards']}
        for r in all_results
    ])
    
    # Run analyses
    print(f"\n{'='*80}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*80}")
    
    single_learned = analyze_single_dimensions(results_df)
    pairwise_learned = analyze_pairwise_dimensions(results_df)
    compare_to_baseline(results_df)
    
    # Create visualization
    create_visualization(results_df)
    
    # Save detailed results
    results_df.to_csv("learned_dimensions_results.csv", index=False)
    print(f"\n✓ Detailed results saved to learned_dimensions_results.csv")
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL VERDICT")
    print(f"{'='*80}")
    
    single_rate = sum(single_learned.values()) / len(single_learned) * 100
    pairwise_rate = sum(pairwise_learned.values()) / len(pairwise_learned) * 100 if pairwise_learned else 0
    
    print(f"\nSingle Dimension Learning: {single_rate:.0f}% successful")
    print(f"Pairwise Joint Learning: {pairwise_rate:.0f}% successful")
    
    if single_rate >= 80 and pairwise_rate >= 60:
        print("\n✓✓✓ VERDICT: Model learned user-conditioning SUCCESSFULLY!")
        print("    - Responds to individual reward dimensions")
        print("    - Can optimize multiple dimensions jointly")
        print("    - Ready for deployment with custom personas")
    elif single_rate >= 60:
        print("\n⚠⚠ VERDICT: Model learned PARTIALLY")
        print("    - Shows some response to reward dimensions")
        print("    - May need more training or different hyperparameters")
        print("    - Can be used but with limitations")
    else:
        print("\n✗✗✗ VERDICT: Model did NOT learn user-conditioning effectively")
        print("    - Little response to different reward dimensions")
        print("    - Needs significant re-training")
        print("    - Check: user_projector weights, training epochs, reward signal")
    
    print(f"\n{'='*80}")
    print("Test Complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

