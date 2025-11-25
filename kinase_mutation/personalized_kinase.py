"""
Personalized RL for Kinase Mutation Design

This module integrates personalized preference-based reward models
into kinase mutation RL training.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

# Add parent directory for personalization imports
sys.path.append(str(Path(__file__).parent.parent))

from personalization import (
    SyntheticUser,
    generate_pairwise_preferences,
    PreferenceDataset,
    PreferenceRewardModel,
    train_preference_reward_model,
)

from personalization.synthetic_users import define_synthetic_users_kinase


def build_kinase_feature_matrix(
    sequences: List[str],
    tokenizer,
    model,
    device: str = "cuda",
) -> np.ndarray:
    """
    Build feature matrix for kinase sequences using ESM.
    
    Args:
        sequences: List of kinase sequences
        tokenizer: ESM tokenizer
        model: ESM model
        device: Device
        
    Returns:
        Feature matrix of shape (N, feature_dim)
    """
    device = torch.device(device)
    embeddings = []
    
    with torch.no_grad():
        for seq in sequences:
            tokens = tokenizer(seq, return_tensors="pt")
            tokens = {k: v.to(device) for k, v in tokens.items()}
            
            outputs = model(**tokens, output_hidden_states=True)
            # Mean pooling
            hidden = outputs.hidden_states[-1]
            embedding = hidden.mean(dim=1).squeeze(0)
            embeddings.append(embedding.cpu().numpy())
    
    embeddings = np.array(embeddings, dtype=np.float32)
    
    # Add sequence-level features
    property_features = []
    for seq in sequences:
        feats = [
            len(seq),
            seq.count('P') / len(seq) if len(seq) > 0 else 0,  # Proline content
            sum(1 for aa in seq if aa in 'KR') / len(seq) if len(seq) > 0 else 0,  # Basic residues
        ]
        property_features.append(feats)
    
    property_features = np.array(property_features, dtype=np.float32)
    features = np.concatenate([embeddings, property_features], axis=1)
    
    return features


def train_personalized_kinase_reward_model(
    sequences: List[str],
    fitness_scores: np.ndarray,
    user: SyntheticUser,
    tokenizer,
    model,
    num_pairs: int = 2000,
    device: str = "cuda",
    output_path: Path | None = None,
) -> PreferenceRewardModel:
    """
    Train personalized reward model for kinase mutation.
    
    Args:
        sequences: List of kinase sequences
        fitness_scores: Array of fitness scores
        user: SyntheticUser
        tokenizer: Tokenizer
        model: ESM model
        num_pairs: Number of preference pairs
        device: Device
        output_path: Optional save path
        
    Returns:
        Trained PreferenceRewardModel
    """
    print(f"[PersonalizedKinase] Training reward model for user: {user.name}")
    
    # Build property dataframe
    df = pd.DataFrame({
        "sequence": sequences,
        "fitness": fitness_scores,
        "stability": np.random.rand(len(sequences)),  # Mock - use real predictor
        "num_mutations": [np.random.randint(0, 5) for _ in sequences],  # Mock
        "conservation": np.random.rand(len(sequences)),  # Mock - use conservation scores
    })
    
    print(f"[PersonalizedKinase] Properties computed for {len(df)} sequences")
    
    # Generate preferences
    print(f"[PersonalizedKinase] Generating {num_pairs} preference pairs...")
    prefs_df = generate_pairwise_preferences(
        df=df,
        user=user,
        num_pairs=num_pairs,
        noise_flip_prob=0.05,
    )
    
    # Build features
    print("[PersonalizedKinase] Building feature matrix...")
    features = build_kinase_feature_matrix(
        sequences=sequences,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )
    print(f"[PersonalizedKinase] Feature shape: {features.shape}")
    
    # Train model
    dataset = PreferenceDataset(features=features, preferences=prefs_df)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    reward_model = PreferenceRewardModel(
        input_dim=features.shape[1],
        hidden_dims=[256, 128],
        dropout=0.1,
    )
    
    print("[PersonalizedKinase] Training preference reward model...")
    reward_model, _ = train_preference_reward_model(
        model=reward_model,
        dataloader=dataloader,
        num_epochs=10,
        lr=1e-3,
        device=device,
        verbose=True,
    )
    
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(reward_model.state_dict(), output_path)
        print(f"[PersonalizedKinase] Saved to {output_path}")
    
    return reward_model


def create_personalized_kinase_reward_fn(
    preference_model: PreferenceRewardModel,
    tokenizer,
    model,
    device: str = "cuda",
):
    """
    Create personalized reward function for kinase sequences.
    
    Args:
        preference_model: Trained preference model
        tokenizer: Tokenizer
        model: ESM model
        device: Device
        
    Returns:
        Reward function
    """
    device = torch.device(device)
    preference_model.eval()
    
    def reward_fn(sequences: List[str]) -> torch.Tensor:
        """Compute rewards for kinase sequences."""
        features = build_kinase_feature_matrix(
            sequences=sequences,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )
        features = torch.tensor(features, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            rewards = preference_model(features)
        
        return rewards
    
    return reward_fn


# Example usage
def main():
    """Example: Train personalized kinase mutation reward models."""
    parser = argparse.ArgumentParser(description="Train personalized kinase reward models")
    parser.add_argument("--data-file", type=Path, default=Path("data/PhoQ.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("esm_8m"))
    parser.add_argument("--output-dir", type=Path, default=Path("personalized_kinase_rewards"))
    parser.add_argument("--num-pairs", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    # Load data
    print("[Main] Loading kinase data...")
    fitness_dict = {}
    if args.data_file.exists():
        for row in csv.reader(open(args.data_file)):
            if row[0] == 'AACombo':
                continue
            fitness_dict[row[0]] = float(row[1])
    else:
        # Mock data
        fitness_dict = {f"AAAA{i:02d}": np.random.rand() * 100 for i in range(100)}
    
    sequences = list(fitness_dict.keys())
    fitness_scores = np.array(list(fitness_dict.values()))
    
    print(f"[Main] Loaded {len(sequences)} sequences")
    
    # Load model
    print("[Main] Loading ESM model...")
    if args.model_dir.exists():
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model = AutoModel.from_pretrained(args.model_dir).to(args.device)
    else:
        print("[Main] Model directory not found, using pretrained ESM")
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D").to(args.device)
    
    model.eval()
    
    # Define users
    users = define_synthetic_users_kinase()
    
    # Train models
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    for user in users:
        print(f"\n{'='*60}")
        print(f"Training for: {user.name}")
        print(f"Description: {user.description}")
        print(f"{'='*60}\n")
        
        output_path = args.output_dir / f"reward_model_{user.name.lower().replace(' ', '_')}.pt"
        
        train_personalized_kinase_reward_model(
            sequences=sequences,
            fitness_scores=fitness_scores,
            user=user,
            tokenizer=tokenizer,
            model=model,
            num_pairs=args.num_pairs,
            device=args.device,
            output_path=output_path,
        )
    
    print("\n[Main] All kinase models trained!")


if __name__ == "__main__":
    main()

