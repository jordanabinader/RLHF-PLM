"""
Personalized GRPO for Antimicrobial Peptide Design

This module integrates personalized preference-based reward models
into the GRPO training loop for AMP generation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

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
    PersonalizedRewardWrapper,
)

from reward import reward_amp_cls, encode
from utils import clean_sequences, load_esm
from mlp import MLP


def build_amp_feature_matrix(
    sequences: list[str],
    esm_model,
    batch_converter,
    alphabet,
    device: str = "cuda",
) -> np.ndarray:
    """
    Build feature matrix for AMP sequences using ESM embeddings + simple properties.
    
    Args:
        sequences: List of peptide sequences
        esm_model: ESM model for embeddings
        batch_converter: ESM batch converter
        alphabet: ESM alphabet
        device: Device for computation
        
    Returns:
        Numpy array of shape (N, feature_dim)
    """
    # Get ESM embeddings
    with torch.no_grad():
        embeddings = encode(
            sequences,
            esm_model=esm_model,
            batch_converter=batch_converter,
            alphabet=alphabet,
            device=device,
        )
        embeddings = embeddings.cpu().numpy()
    
    # Compute simple property features
    property_features = []
    for seq in sequences:
        from personalization.synthetic_users import (
            compute_length,
            compute_hydrophobicity,
            compute_net_charge,
        )
        
        feats = [
            compute_length(seq),
            compute_hydrophobicity(seq),
            compute_net_charge(seq),
        ]
        property_features.append(feats)
    
    property_features = np.array(property_features, dtype=np.float32)
    
    # Concatenate ESM embeddings with properties
    features = np.concatenate([embeddings, property_features], axis=1)
    return features


def train_personalized_amp_reward_model(
    sequences: list[str],
    user: SyntheticUser,
    esm_model,
    batch_converter,
    alphabet,
    classifier,
    num_pairs: int = 5000,
    device: str = "cuda",
    output_path: Path | None = None,
) -> PreferenceRewardModel:
    """
    Train a personalized reward model for AMP design.
    
    Args:
        sequences: List of training sequences
        user: SyntheticUser defining preferences
        esm_model: ESM model for embeddings
        batch_converter: ESM batch converter
        alphabet: ESM alphabet
        classifier: AMP activity classifier
        num_pairs: Number of preference pairs to generate
        device: Device for training
        output_path: Optional path to save the trained model
        
    Returns:
        Trained PreferenceRewardModel
    """
    from personalization.synthetic_users import define_synthetic_users_amp
    
    print(f"[PersonalizedAMP] Training reward model for user: {user.name}")
    
    # Step 1: Compute properties for sequences
    print("[PersonalizedAMP] Computing sequence properties...")
    
    # Get activity scores from classifier
    with torch.no_grad():
        rewards, _ = reward_amp_cls(
            sequences=clean_sequences(sequences),
            esm_model=esm_model,
            batch_converter=batch_converter,
            alphabet=alphabet,
            classifier=classifier,
            device=device,
        )
        activity_scores = rewards.cpu().numpy()
    
    # Compute additional properties
    property_data = {
        "sequence": sequences,
        "activity_score": activity_scores,
    }
    
    from personalization.synthetic_users import (
        compute_length,
        compute_hydrophobicity,
        compute_net_charge,
        compute_stability_proxy,
    )
    
    df = pd.DataFrame(property_data)
    df["length"] = df["sequence"].apply(compute_length)
    df["hydrophobicity"] = df["sequence"].apply(compute_hydrophobicity)
    df["charge"] = df["sequence"].apply(compute_net_charge)
    df["stability_score"] = df["sequence"].apply(compute_stability_proxy)
    
    # Mock toxicity scores (in practice, use a real toxicity predictor)
    df["toxicity_score"] = np.random.rand(len(df)) * 0.5
    
    print(f"[PersonalizedAMP] Generated properties for {len(df)} sequences")
    
    # Step 2: Generate pairwise preferences
    print(f"[PersonalizedAMP] Generating {num_pairs} preference pairs...")
    prefs_df = generate_pairwise_preferences(
        df=df,
        user=user,
        num_pairs=num_pairs,
        noise_flip_prob=0.05,
    )
    print(f"[PersonalizedAMP] Generated {len(prefs_df)} preferences")
    
    # Step 3: Build feature matrix
    print("[PersonalizedAMP] Building feature matrix...")
    features = build_amp_feature_matrix(
        sequences=sequences,
        esm_model=esm_model,
        batch_converter=batch_converter,
        alphabet=alphabet,
        device=device,
    )
    print(f"[PersonalizedAMP] Feature matrix shape: {features.shape}")
    
    # Step 4: Create dataset and dataloader
    dataset = PreferenceDataset(features=features, preferences=prefs_df)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # Step 5: Create and train reward model
    input_dim = features.shape[1]
    reward_model = PreferenceRewardModel(
        input_dim=input_dim,
        hidden_dims=[256, 128, 64],
        dropout=0.1,
    )
    
    print("[PersonalizedAMP] Training preference reward model...")
    reward_model, loss_history = train_preference_reward_model(
        model=reward_model,
        dataloader=dataloader,
        num_epochs=10,
        lr=1e-3,
        device=device,
        verbose=True,
    )
    
    # Save model if path provided
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(reward_model.state_dict(), output_path)
        print(f"[PersonalizedAMP] Saved reward model to {output_path}")
    
    return reward_model


def create_personalized_amp_reward_fn(
    user: SyntheticUser,
    reward_model: PreferenceRewardModel,
    esm_model,
    batch_converter,
    alphabet,
    classifier,
    blend_weight: float = 0.5,
    device: str = "cuda",
):
    """
    Create a blended reward function combining classifier and preferences.
    
    Args:
        user: SyntheticUser
        reward_model: Trained preference reward model
        esm_model: ESM model
        batch_converter: ESM batch converter
        alphabet: ESM alphabet
        classifier: AMP classifier
        blend_weight: Weight for blending (0=only classifier, 1=only preferences)
        device: Device
        
    Returns:
        Callable reward function
    """
    device = torch.device(device)
    
    # Base reward function using classifier
    def base_reward_fn(sequences: Iterable[str]):
        return reward_amp_cls(
            sequences=sequences,
            esm_model=esm_model,
            batch_converter=batch_converter,
            alphabet=alphabet,
            classifier=classifier,
            device=device,
        )
    
    # Feature extractor for preference model
    def feature_extractor(sequences: list[str]) -> torch.Tensor:
        features = build_amp_feature_matrix(
            sequences=sequences,
            esm_model=esm_model,
            batch_converter=batch_converter,
            alphabet=alphabet,
            device=device,
        )
        return torch.tensor(features, dtype=torch.float32, device=device)
    
    # Create wrapper
    wrapper = PersonalizedRewardWrapper(
        base_reward_fn=base_reward_fn,
        preference_reward_model=reward_model,
        feature_extractor=feature_extractor,
        blend_weight=blend_weight,
        device=device,
    )
    
    return wrapper


# Example usage script
def main():
    """
    Example: Train personalized reward models for different AMP design users.
    """
    parser = argparse.ArgumentParser(description="Train personalized AMP reward models")
    parser.add_argument("--data-file", type=Path, help="CSV file with AMP sequences")
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("personalized_rewards"))
    parser.add_argument("--esm-mode", type=str, default="8M", choices=["8M", "650M"])
    parser.add_argument("--num-pairs", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    # Load ESM and classifier
    print("[Main] Loading ESM and classifier...")
    from utils import load_esm
    
    batch_converter, esm_model, alphabet = load_esm(args.esm_mode, device=args.device)
    classifier = MLP(input_dim=320 if args.esm_mode == "8M" else 1280, hidden_dim=128)
    classifier.load_state_dict(torch.load(args.classifier_checkpoint, map_location="cpu"))
    classifier = classifier.to(args.device).eval()
    
    # Load or generate sequences
    if args.data_file and args.data_file.exists():
        df = pd.read_csv(args.data_file)
        sequences = df["sequence"].tolist()
    else:
        # Generate some example sequences
        print("[Main] No data file provided, using example sequences")
        sequences = [
            "GIGKFLHSAKKFGKAFVGEIMNS",
            "KKLLPIVKKK",
            "KWWKWWKKWW",
            "GLFDIVKKVVGALG",
            "FLGALFKVASKLF",
        ] * 50  # Replicate for training
    
    print(f"[Main] Using {len(sequences)} sequences")
    
    # Define synthetic users
    from personalization.synthetic_users import define_synthetic_users_amp
    
    users = define_synthetic_users_amp()
    
    # Train reward models for each user
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    for user in users:
        print(f"\n{'='*60}")
        print(f"Training reward model for: {user.name}")
        print(f"Description: {user.description}")
        print(f"{'='*60}\n")
        
        output_path = args.output_dir / f"reward_model_{user.name.lower().replace(' ', '_')}.pt"
        
        reward_model = train_personalized_amp_reward_model(
            sequences=sequences,
            user=user,
            esm_model=esm_model,
            batch_converter=batch_converter,
            alphabet=alphabet,
            classifier=classifier,
            num_pairs=args.num_pairs,
            device=args.device,
            output_path=output_path,
        )
    
    print("\n[Main] All personalized reward models trained successfully!")


if __name__ == "__main__":
    main()

