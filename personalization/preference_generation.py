"""
Pairwise preference generation from synthetic users.

This module generates preference pairs that can be used to train
Bradley-Terry style reward models for personalized RLHF.
"""

from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .synthetic_users import SyntheticUser


def generate_pairwise_preferences(
    df: pd.DataFrame,
    user: SyntheticUser,
    num_pairs: int,
    noise_flip_prob: float = 0.05,
    property_columns: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate pairwise preference data for a synthetic user.
    
    Args:
        df: DataFrame with sequences and their properties
        user: SyntheticUser defining the preference weights
        num_pairs: Number of pairwise comparisons to generate
        noise_flip_prob: Probability of flipping a preference label (simulates imperfect feedback)
        property_columns: List of property column names to use (if None, uses user.weights.keys())
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with columns: idx_a, idx_b, pref (1 if A preferred, 0 if B preferred), reward_diff
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Compute rewards for each sequence based on user preferences
    property_cols = property_columns if property_columns is not None else list(user.weights.keys())
    
    rewards = []
    for _, row in df.iterrows():
        properties = {col: row[col] for col in property_cols if col in df.columns}
        rewards.append(user.compute_reward(properties))
    
    rewards = np.array(rewards)
    n = len(df)
    
    idx_a_list = []
    idx_b_list = []
    pref_list = []
    reward_diff_list = []
    
    for _ in range(num_pairs):
        # Sample two different sequences
        i, j = np.random.randint(0, n), np.random.randint(0, n)
        while i == j:
            j = np.random.randint(0, n)
        
        r_i, r_j = rewards[i], rewards[j]
        reward_diff = r_i - r_j
        
        # Determine preference
        if abs(reward_diff) < 1e-8:
            # Break ties randomly
            pref = np.random.randint(0, 2)
        else:
            pref = 1 if r_i > r_j else 0
        
        # Add noise: flip label with some probability
        if np.random.rand() < noise_flip_prob:
            pref = 1 - pref
        
        idx_a_list.append(i)
        idx_b_list.append(j)
        pref_list.append(pref)
        reward_diff_list.append(float(reward_diff))
    
    prefs_df = pd.DataFrame({
        "idx_a": idx_a_list,
        "idx_b": idx_b_list,
        "pref": pref_list,
        "reward_diff": reward_diff_list,
    })
    
    return prefs_df


def generate_multi_user_preferences(
    df: pd.DataFrame,
    users: List[SyntheticUser],
    num_pairs_per_user: int,
    noise_flip_prob: float = 0.05,
    property_columns: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate preferences for multiple users.
    
    Args:
        df: DataFrame with sequences and properties
        users: List of SyntheticUser objects
        num_pairs_per_user: Number of preference pairs per user
        noise_flip_prob: Probability of noisy labels
        property_columns: Property columns to use
        seed: Random seed
        
    Returns:
        Tuple of (combined_preferences_df, user_index_df) where the second
        DataFrame maps user_id to user names
    """
    all_prefs = []
    user_index = []
    
    for user_id, user in enumerate(users):
        prefs = generate_pairwise_preferences(
            df=df,
            user=user,
            num_pairs=num_pairs_per_user,
            noise_flip_prob=noise_flip_prob,
            property_columns=property_columns,
            seed=seed + user_id if seed is not None else None,
        )
        prefs["user_id"] = user_id
        all_prefs.append(prefs)
        user_index.append({"user_id": user_id, "user_name": user.name})
    
    combined_prefs = pd.concat(all_prefs, ignore_index=True)
    user_index_df = pd.DataFrame(user_index)
    
    return combined_prefs, user_index_df


class PreferenceDataset(Dataset):
    """
    PyTorch Dataset for pairwise preferences.
    
    Args:
        features: Numpy array of shape (N, D) where N is number of sequences
        preferences: DataFrame with columns idx_a, idx_b, pref
        user_embeddings: Optional user embeddings of shape (num_users, user_dim)
        include_user_id: Whether to include user_id in the batch
    """
    
    def __init__(
        self,
        features: np.ndarray,
        preferences: pd.DataFrame,
        user_embeddings: Optional[np.ndarray] = None,
        include_user_id: bool = False,
    ):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.idx_a = torch.tensor(preferences["idx_a"].values, dtype=torch.long)
        self.idx_b = torch.tensor(preferences["idx_b"].values, dtype=torch.long)
        self.pref = torch.tensor(preferences["pref"].values, dtype=torch.float32)
        
        self.include_user_id = include_user_id
        if include_user_id and "user_id" in preferences.columns:
            self.user_ids = torch.tensor(preferences["user_id"].values, dtype=torch.long)
        else:
            self.user_ids = None
        
        self.user_embeddings = None
        if user_embeddings is not None:
            self.user_embeddings = torch.tensor(user_embeddings, dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.pref)
    
    def __getitem__(self, idx: int) -> dict:
        batch = {
            "feat_a": self.features[self.idx_a[idx]],
            "feat_b": self.features[self.idx_b[idx]],
            "pref": self.pref[idx],
        }
        
        if self.include_user_id and self.user_ids is not None:
            batch["user_id"] = self.user_ids[idx]
            if self.user_embeddings is not None:
                batch["user_embed"] = self.user_embeddings[self.user_ids[idx]]
        
        return batch

