"""
Compute property normalization statistics for fair reward scaling.

This script computes mean and standard deviation for all properties
from the training data, enabling Z-score normalization during GRPO.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List
import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from personalization.unified_property_fn import create_unified_property_function


AMINOACID = 'ACDEFGHIKLMNPQRSTVWY'


def load_sequences_from_fasta(fasta_path: Path) -> List[str]:
    """Load protein sequences from FASTA file."""
    sequences = []
    current_seq = None
    
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Save previous sequence
                if current_seq:
                    sequences.append(current_seq)
                current_seq = None
            elif sum([char in AMINOACID for char in line]) == len(line) and len(line) > 0:
                # This is a sequence line
                current_seq = line
    
    # Save last sequence
    if current_seq:
        sequences.append(current_seq)
    
    return sequences


def load_sequences_from_domain_file(domain_path: Path) -> List[str]:
    """Load sequences from ToxDL2 domain file format."""
    sequences = []
    
    with open(domain_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        # Check if this is a sequence line (all amino acids)
        if sum([char in AMINOACID for char in line]) == len(line) and len(line) > 0:
            sequences.append(line)
    
    return sequences


def compute_normalization_stats(
    sequences: List[str],
    property_fn,
    batch_size: int = 32,
    device: str = "cuda"
) -> dict:
    """
    Compute mean and std for each property across all sequences.
    
    Args:
        sequences: List of protein sequences
        property_fn: UnifiedPropertyFunction instance
        batch_size: Batch size for processing
        device: Device for computation
    
    Returns:
        Dictionary with 'mean' and 'std' lists (4 values each)
    """
    all_properties = []
    
    print(f"Computing properties for {len(sequences)} sequences...")
    
    # Process in batches
    for i in tqdm(range(0, len(sequences), batch_size)):
        batch = sequences[i:i + batch_size]
        
        try:
            with torch.no_grad():
                props = property_fn(batch)  # (batch_size, 4)
                all_properties.append(props.cpu())
        except Exception as e:
            print(f"Warning: Failed to process batch {i//batch_size}: {e}")
            continue
    
    if not all_properties:
        raise ValueError("No properties could be computed!")
    
    # Concatenate all properties
    all_properties = torch.cat(all_properties, dim=0)  # (num_sequences, 4)
    
    print(f"Successfully computed properties for {all_properties.shape[0]} sequences")
    
    # Compute statistics
    mean = all_properties.mean(dim=0)  # (4,)
    std = all_properties.std(dim=0)    # (4,)
    
    # Print statistics for review
    property_names = ['activity', 'toxicity', 'stability', 'length']
    print("\nProperty Statistics:")
    print("-" * 60)
    for i, name in enumerate(property_names):
        print(f"{name:12s}: mean={mean[i]:.4f}, std={std[i]:.4f}")
    
    return {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'property_names': property_names,
        'num_sequences': all_properties.shape[0]
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute property normalization statistics from training data"
    )
    parser.add_argument(
        "--data-source",
        type=str,
        required=True,
        help="Path to training data (domain file or fasta file)"
    )
    parser.add_argument(
        "--data-format",
        type=str,
        choices=["domain", "fasta"],
        default="domain",
        help="Format of data file"
    )
    parser.add_argument(
        "--activity-checkpoint",
        type=str,
        default="amp_design/best_new_4.pth",
        help="Path to activity checkpoint"
    )
    parser.add_argument(
        "--toxicity-checkpoint",
        type=str,
        default="personalization/checkpoints/toxicity_head.pth",
        help="Path to toxicity checkpoint"
    )
    parser.add_argument(
        "--stability-checkpoint",
        type=str,
        default="personalization/checkpoints/stability_head.pth",
        help="Path to stability checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="personalization/checkpoints/property_normalization.json",
        help="Output path for statistics JSON"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)"
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Maximum number of sequences to process (for testing)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Computing Property Normalization Statistics")
    print("=" * 80)
    
    # Load sequences
    data_path = Path(args.data_source)
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)
    
    print(f"\nLoading sequences from {data_path}...")
    if args.data_format == "domain":
        sequences = load_sequences_from_domain_file(data_path)
    else:
        sequences = load_sequences_from_fasta(data_path)
    
    if args.max_sequences:
        sequences = sequences[:args.max_sequences]
    
    print(f"Loaded {len(sequences)} sequences")
    
    # Create property function
    print("\nCreating unified property function...")
    property_fn = create_unified_property_function(
        activity_checkpoint=args.activity_checkpoint,
        toxicity_checkpoint=args.toxicity_checkpoint,
        stability_checkpoint=args.stability_checkpoint,
        device=args.device,
    )
    print("✓ Property function created")
    
    # Compute statistics
    stats = compute_normalization_stats(
        sequences=sequences,
        property_fn=property_fn,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✓ Statistics saved to: {output_path}")
    print("\nUsage:")
    print(f"  Pass normalization_stats_path='{output_path}' to create_unified_property_function()")


if __name__ == "__main__":
    main()

