"""
Evaluate user-conditioned GRPO policy across multiple personas.

This script:
1. Generates sequences from the trained policy for each persona
2. Computes property distributions per persona
3. Evaluates validity rates
4. Measures reward correlation with persona weights
5. Assesses diversity of generated sequences
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import argparse
from typing import List, Dict
import pandas as pd
import numpy as np
from collections import Counter

from personalization.unified_property_fn import create_unified_property_function
from personalization.personas import get_persona, list_personas, compute_personalized_reward
from personalization.validity import validate_sequences, get_validity_stats
from personalization.hybrid_reward import create_hybrid_reward_fn
from amp_design.utils import load_pretrained_progen_model, clean_sequences


def generate_sequences_for_persona(
    policy,
    tokenizer,
    persona_name: str,
    num_sequences: int = 100,
    device: str = "cuda",
) -> List[str]:
    """Generate sequences using policy conditioned on persona"""
    persona = get_persona(persona_name)
    user_context = persona.weights.to(device)
    
    # Create prompt
    try:
        prompt_ids = tokenizer.encode("<|bos|>", return_tensors="pt").to(device)
    except:
        prompt_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)
    
    sequences = []
    batch_size = 10
    
    for i in range(0, num_sequences, batch_size):
        current_batch_size = min(batch_size, num_sequences - i)
        batch_prompts = prompt_ids.repeat(current_batch_size, 1)
        
        with torch.no_grad():
            try:
                if hasattr(policy, 'generate') and hasattr(policy, 'user_projector'):
                    # User-conditioned policy
                    outputs = policy.generate(
                        batch_prompts,
                        user_context=user_context,
                        max_new_tokens=50,
                        do_sample=True,
                        top_p=0.9,
                        temperature=0.8,
                    )
                else:
                    # Standard policy
                    outputs = policy.generate(
                        batch_prompts,
                        max_new_tokens=50,
                        do_sample=True,
                        top_p=0.9,
                        temperature=0.8,
                    )
            except Exception as e:
                print(f"Error generating batch {i}: {e}")
                continue
        
        # Decode sequences
        for seq_ids in outputs:
            seq = tokenizer.decode(seq_ids, skip_special_tokens=True)
            # Remove special tokens
            seq = seq.replace("<|bos|>", "").replace("<|eos|>", "").strip()
            if seq:
                sequences.append(seq)
    
    # Clean all sequences (remove non-alphabetic characters, uppercase)
    sequences = clean_sequences(sequences)
    
    return sequences[:num_sequences]


def evaluate_persona_properties(
    sequences: List[str],
    property_fn,
    persona_name: str,
) -> Dict:
    """Evaluate properties and rewards for sequences"""
    persona = get_persona(persona_name)
    
    # Compute properties
    with torch.no_grad():
        properties = property_fn(sequences)  # (N, 4)
        
    # Compute rewards
    rewards = compute_personalized_reward(properties, persona)
    
    # Validity
    valid_mask = validate_sequences(sequences)
    validity_stats = get_validity_stats(sequences)
    
    # Statistics
    properties_np = properties.cpu().numpy()
    rewards_np = rewards.cpu().numpy()
    
    stats = {
        'persona': persona_name,
        'num_sequences': len(sequences),
        'validity_rate': validity_stats['validity_rate'],
        'invalid_aa': validity_stats['invalid_aa'],
        'invalid_length': validity_stats['invalid_length'],
        'invalid_charge': validity_stats['invalid_charge'],
        'invalid_repeats': validity_stats['invalid_repeats'],
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
    
    return stats, properties_np, rewards_np, valid_mask


def calculate_diversity(sequences: List[str]) -> Dict:
    """Calculate sequence diversity metrics"""
    if not sequences:
        return {
            'uniqueness': 0,
            'exact_duplicates': 0,
            'length_std': 0,
            'aa_entropy': 0,
        }
    
    # Unique sequences
    unique_seqs = set(sequences)
    uniqueness = len(unique_seqs) / len(sequences)
    
    # Exact duplicates
    exact_matches = len(sequences) - len(unique_seqs)
    
    # Length diversity
    lengths = [len(s) for s in sequences]
    length_std = np.std(lengths) if lengths else 0
    
    # Amino acid composition diversity
    all_aas = ''.join(sequences)
    if all_aas:
        aa_counts = Counter(all_aas)
        total_aa = sum(aa_counts.values())
        aa_entropy = -sum((c/total_aa) * np.log2(c/total_aa) 
                         for c in aa_counts.values() if c > 0)
    else:
        aa_entropy = 0
    
    return {
        'uniqueness': uniqueness,
        'exact_duplicates': exact_matches,
        'length_std': length_std,
        'aa_entropy': aa_entropy,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate user-conditioned policy")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained policy checkpoint")
    parser.add_argument("--tokenizer-path", type=Path, required=True, help="Path to tokenizer")
    parser.add_argument("--activity-checkpoint", type=Path, required=True, help="Path to activity checkpoint")
    parser.add_argument("--toxicity-checkpoint", type=Path, required=True, help="Path to toxicity checkpoint")
    parser.add_argument("--stability-checkpoint", type=Path, required=True, help="Path to stability checkpoint")
    parser.add_argument("--num-sequences", type=int, default=100, help="Number of sequences per persona")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation_results"), help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--personas", nargs="+", help="Specific personas to evaluate (default: all)")
    args = parser.parse_args()
    
    args.output_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 80)
    print("User-Conditioned Policy Evaluation")
    print("=" * 80)
    
    # Load policy
    print("\n1. Loading policy...")
    try:
        tokenizer, policy = load_pretrained_progen_model(
            str(args.checkpoint),
            str(args.tokenizer_path)
        )
        policy = policy.to(args.device).eval()
        print(f"   ✓ Policy loaded from {args.checkpoint}")
    except Exception as e:
        print(f"   ✗ Error loading policy: {e}")
        return
    
    # Load property function
    print("\n2. Loading property function...")
    try:
        property_fn = create_unified_property_function(
            activity_checkpoint=args.activity_checkpoint,
            toxicity_checkpoint=args.toxicity_checkpoint,
            stability_checkpoint=args.stability_checkpoint,
            device=args.device,
        )
        print("   ✓ Property function loaded")
    except Exception as e:
        print(f"   ✗ Error loading property function: {e}")
        return
    
    # Determine personas to evaluate
    if args.personas:
        personas_to_eval = args.personas
    else:
        personas_to_eval = list_personas()
    
    # Evaluate each persona
    print(f"\n3. Generating and evaluating sequences for {len(personas_to_eval)} personas...")
    all_stats = []
    all_sequences = {}
    
    for persona_name in personas_to_eval:
        print(f"\n   Persona: {persona_name}")
        
        # Generate sequences
        try:
            sequences = generate_sequences_for_persona(
                policy=policy,
                tokenizer=tokenizer,
                persona_name=persona_name,
                num_sequences=args.num_sequences,
                device=args.device,
            )
            all_sequences[persona_name] = sequences
            print(f"     Generated: {len(sequences)} sequences")
        except Exception as e:
            print(f"     ✗ Error generating sequences: {e}")
            continue
        
        # Evaluate properties
        try:
            stats, properties, rewards, valid_mask = evaluate_persona_properties(
                sequences=sequences,
                property_fn=property_fn,
                persona_name=persona_name,
            )
            
            # Calculate diversity
            diversity = calculate_diversity(sequences)
            stats.update(diversity)
            
            all_stats.append(stats)
            
            print(f"     Validity: {stats['validity_rate']:.2%}")
            print(f"     Mean Reward: {stats['mean_reward']:.3f}")
            print(f"     Uniqueness: {stats['uniqueness']:.2%}")
        except Exception as e:
            print(f"     ✗ Error evaluating properties: {type(e).__name__}: {e}")
            import traceback
            print("     Traceback:")
            traceback.print_exc()
            continue
    
    # Create summary DataFrame
    if all_stats:
        df = pd.DataFrame(all_stats)
        
        print("\n" + "=" * 80)
        print("Summary Statistics")
        print("=" * 80)
        print(df.to_string(index=False))
        
        # Save results
        df.to_csv(args.output_dir / "persona_evaluation.csv", index=False)
        
        # Save sequences
        for persona_name, sequences in all_sequences.items():
            seq_file = args.output_dir / f"{persona_name}_sequences.txt"
            with open(seq_file, 'w') as f:
                f.write('\n'.join(sequences))
        
        print(f"\n✓ Results saved to {args.output_dir}")
    else:
        print("\n✗ No results to save")


if __name__ == "__main__":
    main()

