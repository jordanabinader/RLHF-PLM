"""
Test Custom Reward Function by Creating a Custom Persona

This script demonstrates the SIMPLEST way to test a custom reward function:
1. Define a custom persona with your desired weights
2. Generate sequences using the trained user-conditioned model
3. Evaluate properties and rewards
4. Analyze results

No training needed - works immediately with your existing model!
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import pandas as pd
import numpy as np
from personalization.personas import create_custom_persona, Persona
from personalization.unified_property_fn import create_unified_property_function
from personalization.evaluate_user_conditioned_policy import generate_sequences_for_persona
from personalization.validity import validate_sequences, get_validity_stats
from amp_design.utils import load_pretrained_progen_model, clean_sequences
from personalization.user_conditioned_policy import UserConditionedPolicyWrapper
from personalization.personas import compute_personalized_reward


def create_my_custom_persona() -> Persona:
    """
    Define your custom reward function here by setting the weights.
    
    The weights define how much you care about each property:
    - activity_weight: Higher = prefer more antimicrobial activity (typically 0.5-1.0)
    - toxicity_weight: Negative = avoid toxicity (typically -1.0 to 0.0)
    - stability_weight: Higher = prefer more stable sequences (typically 0.3-1.0)
    - length_weight: Negative = prefer shorter sequences (typically -0.8 to 0.0)
    
    Returns:
        Custom Persona object
    """
    # CUSTOMIZE THESE WEIGHTS FOR YOUR REWARD FUNCTION!
    custom_persona = create_custom_persona(
        name="MyCustomReward",
        activity_weight=0.9,      # Strong preference for activity
        toxicity_weight=-0.6,     # Moderate penalty for toxicity
        stability_weight=0.4,     # Some preference for stability
        length_weight=-0.3,       # Slight preference for shorter sequences
        description="My custom reward function for testing"
    )
    
    # Alternative: Create directly with a tensor for more control
    # custom_persona = Persona(
    #     name="MyCustomReward",
    #     weights=torch.tensor([0.9, -0.6, 0.4, -0.3]),  # [act, tox, stab, len]
    #     description="My custom reward function"
    # )
    
    return custom_persona


def load_user_conditioned_model(checkpoint_path: str, tokenizer_path: str, device: str = "cuda"):
    """Load the trained user-conditioned model."""
    print(f"\nLoading model from {checkpoint_path}...")
    
    tokenizer, base_policy = load_pretrained_progen_model(
        checkpoint_path, tokenizer_path
    )
    
    # Wrap with user conditioning
    policy = UserConditionedPolicyWrapper(base_policy)
    
    # Load the trained user projector
    projector_path = Path(checkpoint_path) / "user_projector.pt"
    if not projector_path.exists():
        raise FileNotFoundError(
            f"user_projector.pt not found at {checkpoint_path}. "
            f"Make sure this is a user-conditioned checkpoint!"
        )
    
    policy.load_user_projector(checkpoint_path)
    policy = policy.to(device).eval()
    
    print("✓ Model loaded successfully!")
    return tokenizer, policy


def generate_and_evaluate(
    policy,
    tokenizer,
    persona: Persona,
    property_fn,
    num_sequences: int = 200,
    device: str = "cuda"
):
    """Generate sequences and evaluate them with the custom reward."""
    print(f"\n{'='*80}")
    print(f"Generating sequences with: {persona.name}")
    print(f"{'='*80}")
    print(f"Weights: {persona.get_weight_dict()}")
    print(f"Description: {persona.description}")
    
    # Generate sequences
    print(f"\nGenerating {num_sequences} sequences...")
    sequences = []
    batch_size = 10
    
    # Create prompt
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
        
        # Decode sequences
        for seq_ids in outputs:
            seq = tokenizer.decode(seq_ids, skip_special_tokens=True)
            seq = seq.replace("<|bos|>", "").replace("<|eos|>", "").strip()
            if seq:
                sequences.append(seq)
        
        if (i // batch_size + 1) % 5 == 0:
            print(f"  Generated {len(sequences)}/{num_sequences}...")
    
    sequences = clean_sequences(sequences[:num_sequences])
    print(f"✓ Generated {len(sequences)} sequences")
    
    # Compute properties
    print("\nComputing properties...")
    with torch.no_grad():
        properties = property_fn(sequences)
    
    # Compute rewards
    rewards = compute_personalized_reward(properties, persona)
    
    # Validity statistics
    validity_stats = get_validity_stats(sequences)
    
    # Convert to numpy for analysis
    properties_np = properties.cpu().numpy()
    rewards_np = rewards.cpu().numpy()
    
    # Calculate statistics
    results = {
        'persona': persona.name,
        'num_sequences': len(sequences),
        'validity_rate': validity_stats['validity_rate'],
        'mean_activity': properties_np[:, 0].mean(),
        'std_activity': properties_np[:, 0].std(),
        'mean_toxicity': properties_np[:, 1].mean(),
        'std_toxicity': properties_np[:, 1].std(),
        'mean_stability': properties_np[:, 2].mean(),
        'std_stability': properties_np[:, 2].std(),
        'mean_length': properties_np[:, 3].mean() * 100,  # Unnormalize
        'std_length': properties_np[:, 3].std() * 100,
        'mean_reward': rewards_np.mean(),
        'std_reward': rewards_np.std(),
        'max_reward': rewards_np.max(),
        'min_reward': rewards_np.min(),
    }
    
    return sequences, properties_np, rewards_np, results


def print_results(results: dict, sequences: list, rewards_np: np.ndarray, top_n: int = 5):
    """Print detailed results."""
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nGeneration Statistics:")
    print(f"  Total sequences: {results['num_sequences']}")
    print(f"  Validity rate: {results['validity_rate']:.1%}")
    
    print(f"\nProperty Statistics:")
    print(f"  Activity:  {results['mean_activity']:.3f} ± {results['std_activity']:.3f}")
    print(f"  Toxicity:  {results['mean_toxicity']:.3f} ± {results['std_toxicity']:.3f}")
    print(f"  Stability: {results['mean_stability']:.3f} ± {results['std_stability']:.3f}")
    print(f"  Length:    {results['mean_length']:.1f} ± {results['std_length']:.1f} aa")
    
    print(f"\nReward Statistics:")
    print(f"  Mean:   {results['mean_reward']:.3f}")
    print(f"  Std:    {results['std_reward']:.3f}")
    print(f"  Max:    {results['max_reward']:.3f}")
    print(f"  Min:    {results['min_reward']:.3f}")
    
    # Show top sequences
    print(f"\n{'='*80}")
    print(f"Top {top_n} Sequences by Reward:")
    print(f"{'='*80}")
    top_indices = np.argsort(rewards_np)[-top_n:][::-1]
    
    for i, idx in enumerate(top_indices):
        seq = sequences[idx]
        reward = rewards_np[idx]
        print(f"\n{i+1}. Reward: {reward:.3f}")
        print(f"   Sequence ({len(seq)} aa): {seq[:60]}{'...' if len(seq) > 60 else ''}")
        if len(seq) > 60:
            print(f"                           {seq[60:]}")


def save_results(sequences: list, properties_np: np.ndarray, rewards_np: np.ndarray, 
                 output_path: str = "custom_reward_results.csv"):
    """Save results to CSV."""
    df = pd.DataFrame({
        'sequence': sequences,
        'activity': properties_np[:, 0],
        'toxicity': properties_np[:, 1],
        'stability': properties_np[:, 2],
        'length': properties_np[:, 3] * 100,  # Unnormalize
        'reward': rewards_np,
    })
    
    df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to {output_path}")


def main():
    """Main testing function."""
    print("="*80)
    print("Custom Reward Function Testing")
    print("="*80)
    
    # Configuration
    checkpoint_path = "grpo_runs/user_conditioned_multi/final_model"
    tokenizer_path = "amp_design/progen2hf/progen2-small"
    activity_checkpoint = "amp_design/best_new_4.pth"
    toxicity_checkpoint = "personalization/checkpoints/toxicity_head.pth"
    stability_checkpoint = "personalization/checkpoints/stability_head.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_sequences = 200
    
    print(f"\nConfiguration:")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device: {device}")
    print(f"  Sequences to generate: {num_sequences}")
    
    # Step 1: Create custom persona
    print(f"\n{'='*80}")
    print("Step 1: Creating Custom Persona")
    print(f"{'='*80}")
    custom_persona = create_my_custom_persona()
    print(f"✓ Created persona: {custom_persona.name}")
    print(f"  Weights: {custom_persona.get_weight_dict()}")
    print(custom_persona.explain())
    
    # Step 2: Load model
    print(f"\n{'='*80}")
    print("Step 2: Loading User-Conditioned Model")
    print(f"{'='*80}")
    tokenizer, policy = load_user_conditioned_model(
        checkpoint_path, tokenizer_path, device
    )
    
    # Step 3: Load property function
    print(f"\n{'='*80}")
    print("Step 3: Loading Property Function")
    print(f"{'='*80}")
    property_fn = create_unified_property_function(
        activity_checkpoint=activity_checkpoint,
        toxicity_checkpoint=toxicity_checkpoint,
        stability_checkpoint=stability_checkpoint,
        device=device,
    )
    print("✓ Property function loaded")
    
    # Step 4: Generate and evaluate
    print(f"\n{'='*80}")
    print("Step 4: Generating and Evaluating Sequences")
    print(f"{'='*80}")
    sequences, properties, rewards, results = generate_and_evaluate(
        policy=policy,
        tokenizer=tokenizer,
        persona=custom_persona,
        property_fn=property_fn,
        num_sequences=num_sequences,
        device=device
    )
    
    # Step 5: Display results
    print_results(results, sequences, rewards, top_n=10)
    
    # Step 6: Save results
    output_path = "custom_reward_results.csv"
    save_results(sequences, properties, rewards, output_path)
    
    print(f"\n{'='*80}")
    print("Testing Complete!")
    print(f"{'='*80}")
    print("\nNext steps:")
    print("  1. Modify the weights in create_my_custom_persona()")
    print("  2. Run again: python test_custom_reward.py")
    print("  3. Compare results to find your optimal reward function")
    print(f"  4. Check {output_path} for detailed results")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

