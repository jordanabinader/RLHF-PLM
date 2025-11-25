"""
Complete end-to-end example of personalized RLHF for AMP design.

This script demonstrates:
1. Loading/generating AMP sequences
2. Defining synthetic users
3. Computing properties
4. Generating pairwise preferences
5. Training preference reward models
6. Using personalized rewards in RL (conceptual)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from personalization import (
    SyntheticUser,
    generate_pairwise_preferences,
    PreferenceDataset,
    PreferenceRewardModel,
    train_preference_reward_model,
)
from personalization.synthetic_users import (
    define_synthetic_users_amp,
    compute_length,
    compute_hydrophobicity,
    compute_net_charge,
    compute_stability_proxy,
)


def generate_example_amp_data(n_sequences: int = 500) -> pd.DataFrame:
    """
    Generate example AMP sequences with properties.
    
    In practice, replace this with real data.
    """
    # Example AMP sequences (from literature/databases)
    base_sequences = [
        "GIGKFLHSAKKFGKAFVGEIMNS",
        "KKLLPIVKKK",
        "KWWKWWKKWW",
        "GLFDIVKKVVGALG",
        "FLGALFKVASKLF",
        "GLLSSLGRKF",
        "KWKSFIKKLTSVGKVLKK",
        "ILPWKWPWWPWRR",
        "GIGAVLKVLTTGLPALISWIKRKRQQ",
        "FFGHLFKLATKIIPSLFQ",
    ]
    
    # Replicate and add variations
    sequences = []
    for i in range(n_sequences):
        base = base_sequences[i % len(base_sequences)]
        # Simple variation (in practice, use more sophisticated generation)
        sequences.append(base)
    
    # Compute properties
    df = pd.DataFrame({"sequence": sequences})
    df["length"] = df["sequence"].apply(compute_length)
    df["hydrophobicity"] = df["sequence"].apply(compute_hydrophobicity)
    df["charge"] = df["sequence"].apply(compute_net_charge)
    df["stability_score"] = df["sequence"].apply(compute_stability_proxy)
    
    # Mock activity and toxicity scores (replace with real predictors)
    # Simulate: longer, more hydrophobic = more active but also more toxic
    df["activity_score"] = (
        0.5 + 0.3 * (df["hydrophobicity"] / 5.0)
        + 0.2 * (df["charge"] / 10.0)
        + np.random.rand(len(df)) * 0.3
    )
    df["toxicity_score"] = (
        0.3 + 0.4 * (df["hydrophobicity"] / 5.0)
        + 0.1 * (df["length"] / 30.0)
        + np.random.rand(len(df)) * 0.2
    )
    
    # Clip to [0, 1]
    df["activity_score"] = df["activity_score"].clip(0, 1)
    df["toxicity_score"] = df["toxicity_score"].clip(0, 1)
    
    return df


def build_simple_features(df: pd.DataFrame) -> np.ndarray:
    """
    Build a simple feature matrix from properties.
    
    In practice, concatenate ESM embeddings with these features.
    """
    feature_cols = [
        "length",
        "hydrophobicity",
        "charge",
        "stability_score",
        "activity_score",
        "toxicity_score",
    ]
    features = df[feature_cols].values.astype(np.float32)
    return features


def main():
    print("="*70)
    print("Personalized RLHF for AMP Design - Complete Example")
    print("="*70)
    
    # Step 1: Generate/load data
    print("\n[Step 1] Generating example AMP data...")
    df = generate_example_amp_data(n_sequences=500)
    print(f"Generated {len(df)} sequences")
    print(f"Property ranges:")
    print(f"  Activity: [{df['activity_score'].min():.2f}, {df['activity_score'].max():.2f}]")
    print(f"  Toxicity: [{df['toxicity_score'].min():.2f}, {df['toxicity_score'].max():.2f}]")
    print(f"  Length: [{df['length'].min():.0f}, {df['length'].max():.0f}]")
    
    # Step 2: Define synthetic users
    print("\n[Step 2] Defining synthetic users...")
    users = define_synthetic_users_amp()
    print(f"Defined {len(users)} synthetic users:")
    for user in users:
        print(f"  - {user.name}: {user.description}")
    
    # Step 3: Train reward models for each user
    print("\n[Step 3] Training personalized reward models...")
    
    trained_models = {}
    
    for user in users[:2]:  # Train for first 2 users as example
        print(f"\n{'-'*60}")
        print(f"Training for: {user.name}")
        print(f"Weights: {user.weights}")
        print(f"{'-'*60}")
        
        # 3a. Generate preferences
        print(f"  Generating preferences...")
        prefs_df = generate_pairwise_preferences(
            df=df,
            user=user,
            num_pairs=3000,
            noise_flip_prob=0.05,
            seed=42,
        )
        print(f"  Generated {len(prefs_df)} preference pairs")
        
        # 3b. Build features
        print(f"  Building feature matrix...")
        features = build_simple_features(df)
        print(f"  Feature shape: {features.shape}")
        
        # 3c. Create dataset
        dataset = PreferenceDataset(features=features, preferences=prefs_df)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=128, shuffle=True
        )
        
        # 3d. Train model
        print(f"  Training preference reward model...")
        reward_model = PreferenceRewardModel(
            input_dim=features.shape[1],
            hidden_dims=[128, 64],
            dropout=0.1,
        )
        
        trained_model, loss_history = train_preference_reward_model(
            model=reward_model,
            dataloader=dataloader,
            num_epochs=5,
            lr=1e-3,
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=True,
        )
        
        trained_models[user.name] = trained_model
        print(f"  ✓ Training complete. Final loss: {loss_history[-1]:.4f}")
    
    # Step 4: Compare reward predictions
    print("\n[Step 4] Comparing reward predictions across users...")
    
    # Select a few test sequences
    test_sequences = [
        df.iloc[0]["sequence"],  # First sequence
        df.iloc[len(df)//2]["sequence"],  # Middle sequence
        df.iloc[-1]["sequence"],  # Last sequence
    ]
    
    test_indices = [0, len(df)//2, len(df)-1]
    test_features = features[test_indices]
    test_features_tensor = torch.tensor(test_features, dtype=torch.float32)
    
    print(f"\nReward predictions for example sequences:")
    print(f"{'Sequence':<30} {'Activity':<10} {'Toxicity':<10}", end="")
    for user_name in trained_models.keys():
        print(f" {user_name:<20}", end="")
    print()
    print("-" * 100)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for i, seq in enumerate(test_sequences):
        activity = df.iloc[test_indices[i]]["activity_score"]
        toxicity = df.iloc[test_indices[i]]["toxicity_score"]
        
        print(f"{seq[:28]:<30} {activity:<10.3f} {toxicity:<10.3f}", end="")
        
        for user_name, model in trained_models.items():
            model = model.to(device).eval()
            with torch.no_grad():
                reward = model(test_features_tensor[i:i+1].to(device))
                print(f" {reward.item():<20.3f}", end="")
        print()
    
    # Step 5: Conceptual RL integration
    print("\n[Step 5] Conceptual RL integration...")
    print("""
    To use these personalized reward models in RL training:
    
    1. In your RL loop (PPO/GRPO/DPO), replace the base reward function:
       
       # Before:
       rewards, mask = base_reward_fn(sequences)
       
       # After:
       def personalized_reward_fn(sequences):
           features = extract_features(sequences)
           rewards = reward_model(features)
           return rewards, mask
    
    2. Or use the PersonalizedRewardWrapper for blending:
       
       from personalization import PersonalizedRewardWrapper
       
       wrapper = PersonalizedRewardWrapper(
           base_reward_fn=amp_classifier_reward,
           preference_reward_model=trained_model,
           feature_extractor=extract_features,
           blend_weight=0.5,
       )
       
       rewards, mask = wrapper(sequences)
    
    3. Train separate policies per user, or train one policy with user conditioning.
    """)
    
    # Save models
    output_dir = Path("personalized_amp_models")
    output_dir.mkdir(exist_ok=True)
    
    for user_name, model in trained_models.items():
        save_path = output_dir / f"{user_name.lower().replace(' ', '_')}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"Saved {user_name} model to {save_path}")
    
    print("\n" + "="*70)
    print("Example complete! Models saved to:", output_dir)
    print("="*70)


if __name__ == "__main__":
    main()

