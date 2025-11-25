"""
Personalized RL for Antibody Mutation Design

This module integrates personalized preference-based reward models
into the antibody mutation policy training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add parent directory to path for personalization imports
sys.path.append(str(Path(__file__).parent.parent))

from personalization import (
    SyntheticUser,
    define_synthetic_users,
    compute_property_scores,
    generate_pairwise_preferences,
    PreferenceDataset,
    PreferenceRewardModel,
    train_preference_reward_model,
)

from personalization.synthetic_users import define_synthetic_users_antibody


def build_antibody_feature_matrix(
    sequences: List[str],
    model,
    tokenizer,
    device: str = "cuda",
) -> np.ndarray:
    """
    Build feature matrix for antibody sequences.
    
    Args:
        sequences: List of antibody sequences
        model: ESM or other protein model
        tokenizer: Tokenizer
        device: Device for computation
        
    Returns:
        Numpy array of shape (N, feature_dim)
    """
    device = torch.device(device)
    embeddings = []
    
    with torch.no_grad():
        for seq in sequences:
            inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs, output_hidden_states=True)
            # Use mean pooling over sequence length
            hidden = outputs.hidden_states[-1]  # Last layer
            embedding = hidden.mean(dim=1).squeeze(0)
            embeddings.append(embedding.cpu().numpy())
    
    embeddings = np.array(embeddings, dtype=np.float32)
    
    # Add simple sequence features
    property_features = []
    for seq in sequences:
        feats = [
            len(seq),  # Length
            seq.count('C') / len(seq) if len(seq) > 0 else 0,  # Cysteine content
            sum(1 for aa in seq if aa in 'KRH') / len(seq) if len(seq) > 0 else 0,  # Charge
        ]
        property_features.append(feats)
    
    property_features = np.array(property_features, dtype=np.float32)
    features = np.concatenate([embeddings, property_features], axis=1)
    
    return features


def train_personalized_antibody_reward_model(
    sequences: List[str],
    user: SyntheticUser,
    model,
    tokenizer,
    base_reward_model,
    num_pairs: int = 3000,
    device: str = "cuda",
    output_path: Path | None = None,
) -> PreferenceRewardModel:
    """
    Train a personalized reward model for antibody mutation.
    
    Args:
        sequences: List of antibody sequences
        user: SyntheticUser defining preferences
        model: Base protein model
        tokenizer: Tokenizer
        base_reward_model: Base reward model for computing properties
        num_pairs: Number of preference pairs
        device: Device
        output_path: Optional save path
        
    Returns:
        Trained PreferenceRewardModel
    """
    print(f"[PersonalizedAntibody] Training reward model for user: {user.name}")
    
    # Step 1: Compute properties
    print("[PersonalizedAntibody] Computing sequence properties...")
    
    # Get binding affinity predictions from base model
    with torch.no_grad():
        binding_scores = []
        for seq in sequences:
            # Mock computation - replace with actual model inference
            # For now, use random scores
            score = np.random.rand()
            binding_scores.append(score)
        binding_scores = np.array(binding_scores)
    
    # Create property dataframe
    df = pd.DataFrame({
        "sequence": sequences,
        "binding_affinity": binding_scores,
        "stability": np.random.rand(len(sequences)),  # Mock - use real predictor
        "immunogenicity": np.random.rand(len(sequences)) * 0.5,  # Mock
        "num_mutations": [np.random.randint(0, 10) for _ in sequences],  # Mock
        "developability": np.random.rand(len(sequences)),  # Mock
    })
    
    print(f"[PersonalizedAntibody] Generated properties for {len(df)} sequences")
    
    # Step 2: Generate preferences
    print(f"[PersonalizedAntibody] Generating {num_pairs} preference pairs...")
    prefs_df = generate_pairwise_preferences(
        df=df,
        user=user,
        num_pairs=num_pairs,
        noise_flip_prob=0.05,
    )
    
    # Step 3: Build feature matrix
    print("[PersonalizedAntibody] Building feature matrix...")
    features = build_antibody_feature_matrix(
        sequences=sequences,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    print(f"[PersonalizedAntibody] Feature matrix shape: {features.shape}")
    
    # Step 4: Train model
    dataset = PreferenceDataset(features=features, preferences=prefs_df)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    reward_model = PreferenceRewardModel(
        input_dim=features.shape[1],
        hidden_dims=[256, 128],
        dropout=0.1,
    )
    
    print("[PersonalizedAntibody] Training preference reward model...")
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
        print(f"[PersonalizedAntibody] Saved to {output_path}")
    
    return reward_model


def create_personalized_antibody_reward_fn(
    user: SyntheticUser,
    preference_model: PreferenceRewardModel,
    base_model,
    tokenizer,
    blend_weight: float = 0.5,
    device: str = "cuda",
):
    """
    Create personalized reward function for antibody mutation.
    
    Args:
        user: SyntheticUser
        preference_model: Trained preference reward model
        base_model: Base protein model
        tokenizer: Tokenizer
        blend_weight: Blending weight
        device: Device
        
    Returns:
        Reward function
    """
    device = torch.device(device)
    preference_model.eval()
    
    def reward_fn(sequences: List[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute personalized rewards for antibody sequences."""
        # Extract features
        features = build_antibody_feature_matrix(
            sequences=sequences,
            model=base_model,
            tokenizer=tokenizer,
            device=device,
        )
        features = torch.tensor(features, dtype=torch.float32, device=device)
        
        # Get preference-based rewards
        with torch.no_grad():
            pref_rewards = preference_model(features)
        
        # Create mask (all valid for now)
        mask = torch.ones_like(pref_rewards, dtype=torch.bool)
        
        return pref_rewards, mask
    
    return reward_fn


# Example usage
def main():
    """Example: Train personalized antibody mutation reward models."""
    parser = argparse.ArgumentParser(description="Train personalized antibody reward models")
    parser.add_argument("--data-file", type=Path, help="CSV with antibody sequences")
    parser.add_argument("--model-path", type=str, default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--output-dir", type=Path, default=Path("personalized_antibody_rewards"))
    parser.add_argument("--num-pairs", type=int, default=3000)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    # Load model
    from transformers import AutoTokenizer, AutoModel
    
    print("[Main] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path).to(args.device)
    model.eval()
    
    # Load sequences
    if args.data_file and args.data_file.exists():
        df = pd.read_csv(args.data_file)
        sequences = df["sequence"].tolist()[:1000]  # Limit for example
    else:
        # Mock sequences
        print("[Main] No data file, using mock sequences")
        sequences = ["EVQLVESGGGLVQPGGSLRLSCAASGFTFS"] * 100
    
    print(f"[Main] Using {len(sequences)} sequences")
    
    # Define users
    users = define_synthetic_users_antibody()
    
    # Train models
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    for user in users:
        print(f"\n{'='*60}")
        print(f"Training for: {user.name}")
        print(f"Description: {user.description}")
        print(f"{'='*60}\n")
        
        output_path = args.output_dir / f"reward_model_{user.name.lower().replace(' ', '_')}.pt"
        
        train_personalized_antibody_reward_model(
            sequences=sequences,
            user=user,
            model=model,
            tokenizer=tokenizer,
            base_reward_model=None,
            num_pairs=args.num_pairs,
            device=args.device,
            output_path=output_path,
        )
    
    print("\n[Main] All models trained!")


if __name__ == "__main__":
    main()

