"""
Calculate Normalization Statistics for Property Function

This script computes mean and standard deviation for all four properties
(activity, toxicity, stability, length) on a representative corpus of AMP sequences.

These statistics are critical for preventing objective inversion in the reward function.
Without normalization, properties with larger scales (like stability: [-10, 10])
dominate the reward signal over properties with smaller scales (activity: [0, 1]).

Usage:
    python personalization/calculate_normalization_stats.py \
        --activity_checkpoint amp_design/best_new_4.pth \
        --toxicity_checkpoint personalization/checkpoints/toxicity_head.pth \
        --stability_checkpoint personalization/checkpoints/stability_head.pth \
        --output_path personalization/checkpoints/property_normalization.json \
        --num_sequences 5000
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from personalization.unified_property_fn import create_unified_property_function


def generate_representative_sequences(
    model_name: str = "hugohrban/progen2-small",
    num_sequences: int = 5000,
    max_length: int = 100,
    device: str = "cuda",
    batch_size: int = 16,
) -> List[str]:
    """
    Generate representative AMP sequences from the base PLM.
    
    Args:
        model_name: Name or path of the PLM
        num_sequences: Number of sequences to generate
        max_length: Maximum sequence length
        device: Device to run on
        batch_size: Generation batch size
    
    Returns:
        List of generated sequences
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from amp_design.utils import clean_sequences
    
    print(f"Loading base model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()
    
    # Set up generation config
    bos_token = tokenizer.bos_token or "<|bos|>"
    
    # Get bad_words_ids for non-canonical amino acids
    canonical_aa = set("ACDEFGHIKLMNPQRSTVWY")
    bad_chars = []
    for token, token_id in tokenizer.get_vocab().items():
        if len(token) == 1 and token.upper() not in canonical_aa:
            if token not in [bos_token, tokenizer.eos_token, tokenizer.pad_token]:
                bad_chars.append(token_id)
    
    print(f"Generating {num_sequences} sequences...")
    sequences = []
    
    num_batches = (num_sequences + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc="Generating"):
            # Encode prompt
            current_batch = min(batch_size, num_sequences - len(sequences))
            inputs = tokenizer(
                [bos_token] * current_batch,
                return_tensors="pt",
                padding=True
            ).to(device)
            
            # Generate
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                bad_words_ids=[[token_id] for token_id in bad_chars] if bad_chars else None,
            )
            
            # Decode
            batch_seqs = [
                tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
            batch_seqs = clean_sequences(batch_seqs)
            sequences.extend(batch_seqs)
            
            if len(sequences) >= num_sequences:
                break
    
    sequences = sequences[:num_sequences]
    print(f"Generated {len(sequences)} sequences")
    
    # Basic statistics
    lengths = [len(seq) for seq in sequences]
    print(f"  Mean length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    print(f"  Length range: [{min(lengths)}, {max(lengths)}]")
    
    return sequences


def calculate_property_statistics(
    sequences: List[str],
    property_fn,
    batch_size: int = 32,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate mean and std for all properties.
    
    Args:
        sequences: List of protein sequences
        property_fn: UnifiedPropertyFunction instance
        batch_size: Batch size for property computation
    
    Returns:
        Dictionary with statistics for each property
    """
    print(f"\nCalculating properties for {len(sequences)} sequences...")
    
    all_properties = []
    
    # Process in batches
    num_batches = (len(sequences) + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc="Computing properties"):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(sequences))
        batch = sequences[start_idx:end_idx]
        
        try:
            # Get properties (batch_size, 4) - WITHOUT normalization
            # We need raw values to calculate stats
            props = property_fn(batch)
            all_properties.append(props.cpu())
        except Exception as e:
            print(f"Warning: Batch {i} failed: {e}")
            continue
    
    # Concatenate all properties
    all_properties = torch.cat(all_properties, dim=0)  # (N, 4)
    
    print(f"Computed properties for {all_properties.shape[0]} sequences")
    
    # Calculate statistics per property
    property_names = ['activity', 'toxicity', 'stability', 'length']
    stats = {}
    
    print("\nProperty Statistics (Raw Values):")
    print("=" * 60)
    
    for i, name in enumerate(property_names):
        values = all_properties[:, i].numpy()
        
        mean = float(np.mean(values))
        std = float(np.std(values))
        min_val = float(np.min(values))
        max_val = float(np.max(values))
        
        stats[name] = {
            'mean': mean,
            'std': std,
            'min': min_val,
            'max': max_val,
        }
        
        print(f"{name:12s}: μ={mean:8.4f}, σ={std:8.4f}, "
              f"range=[{min_val:7.2f}, {max_val:7.2f}]")
    
    print("=" * 60)
    
    return stats


def save_normalization_stats(
    stats: Dict[str, Dict[str, float]],
    output_path: Path,
    metadata: Dict = None,
) -> None:
    """
    Save normalization statistics to JSON file.
    
    The saved format matches what UnifiedPropertyFunction expects:
    {
        "mean": [mean_act, mean_tox, mean_stab, mean_len],
        "std": [std_act, std_tox, std_stab, std_len],
        "metadata": {...}
    }
    
    Args:
        stats: Statistics dictionary from calculate_property_statistics
        output_path: Path to save JSON file
        metadata: Optional metadata to include
    """
    property_names = ['activity', 'toxicity', 'stability', 'length']
    
    # Extract mean and std arrays in correct order
    mean_values = [stats[name]['mean'] for name in property_names]
    std_values = [stats[name]['std'] for name in property_names]
    
    output_data = {
        'mean': mean_values,
        'std': std_values,
        'property_names': property_names,
        'full_stats': stats,  # Include full stats for reference
    }
    
    if metadata:
        output_data['metadata'] = metadata
    
    # Save to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Saved normalization statistics to {output_path}")
    print(f"  Mean vector: {mean_values}")
    print(f"  Std vector:  {std_values}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate normalization statistics for property function"
    )
    parser.add_argument(
        "--activity_checkpoint",
        type=str,
        required=True,
        help="Path to activity head checkpoint (best_new_4.pth)"
    )
    parser.add_argument(
        "--toxicity_checkpoint",
        type=str,
        required=True,
        help="Path to toxicity head checkpoint"
    )
    parser.add_argument(
        "--stability_checkpoint",
        type=str,
        required=True,
        help="Path to stability head checkpoint"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="personalization/checkpoints/property_normalization.json",
        help="Path to save normalization statistics JSON"
    )
    parser.add_argument(
        "--num_sequences",
        type=int,
        default=5000,
        help="Number of sequences to use for statistics (default: 5000)"
    )
    parser.add_argument(
        "--sequences_file",
        type=str,
        default=None,
        help="Optional: Load sequences from file instead of generating"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="hugohrban/progen2-small",
        help="Base model for sequence generation"
    )
    parser.add_argument(
        "--esm_model_size",
        type=str,
        default="650M",
        choices=["650M", "8M"],
        help="ESM model size for property computation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for generation and property computation"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=100,
        help="Maximum sequence length"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Property Normalization Statistics Calculator")
    print("=" * 80)
    
    # Step 1: Get representative sequences
    if args.sequences_file:
        print(f"Loading sequences from {args.sequences_file}...")
        with open(args.sequences_file, 'r') as f:
            sequences = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(sequences)} sequences")
    else:
        sequences = generate_representative_sequences(
            model_name=args.model_name,
            num_sequences=args.num_sequences,
            max_length=args.max_length,
            device=args.device,
            batch_size=args.batch_size,
        )
    
    # Step 2: Create property function WITHOUT normalization
    # (we need raw values to calculate the stats)
    print("\nLoading property function...")
    property_fn = create_unified_property_function(
        activity_checkpoint=args.activity_checkpoint,
        toxicity_checkpoint=args.toxicity_checkpoint,
        stability_checkpoint=args.stability_checkpoint,
        esm_model_size=args.esm_model_size,
        device=args.device,
        max_length=args.max_length,
        normalization_stats_path=None,  # Don't load stats yet!
    )
    
    # Step 3: Calculate statistics
    stats = calculate_property_statistics(
        sequences=sequences,
        property_fn=property_fn,
        batch_size=args.batch_size,
    )
    
    # Step 4: Save statistics
    metadata = {
        'num_sequences': len(sequences),
        'model_name': args.model_name,
        'esm_model_size': args.esm_model_size,
        'max_length': args.max_length,
        'generated_from': 'base_model' if not args.sequences_file else args.sequences_file,
    }
    
    save_normalization_stats(
        stats=stats,
        output_path=Path(args.output_path),
        metadata=metadata,
    )
    
    print("\n" + "=" * 80)
    print("✓ Normalization statistics calculation complete!")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"1. Pass --normalization_stats_path {args.output_path} to GRPO training")
    print(f"2. The unified property function will automatically apply z-score normalization")
    print(f"3. This should fix the objective inversion issue")


if __name__ == "__main__":
    main()

