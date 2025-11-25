"""
Base class for personalized RL training.

This module provides a template for integrating personalized reward models
into RL training loops across different domains.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import torch
import torch.nn as nn

from .synthetic_users import SyntheticUser
from .preference_reward_model import PreferenceRewardModel


class PersonalizedRLTrainer(ABC):
    """
    Abstract base class for personalized RL training.
    
    Subclasses should implement task-specific methods for:
    - Feature extraction from sequences
    - Reward computation
    - Policy updates
    
    Attributes:
        user: The current SyntheticUser defining preferences
        reward_model: Trained PreferenceRewardModel
        device: Torch device
    """
    
    def __init__(
        self,
        user: SyntheticUser,
        reward_model: PreferenceRewardModel,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.user = user
        self.reward_model = reward_model
        self.device = torch.device(device)
        self.reward_model = self.reward_model.to(self.device)
        self.reward_model.eval()
    
    @abstractmethod
    def extract_features(self, sequences: list[str]) -> torch.Tensor:
        """
        Extract features from sequences for reward model.
        
        Args:
            sequences: List of sequence strings
            
        Returns:
            Tensor of shape (len(sequences), feature_dim)
        """
        pass
    
    def compute_personalized_rewards(
        self,
        sequences: list[str],
        user_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute personalized rewards for sequences.
        
        Args:
            sequences: List of sequence strings
            user_id: Optional user ID (if using multi-user reward model)
            
        Returns:
            Tensor of rewards of shape (len(sequences),)
        """
        self.reward_model.eval()
        with torch.no_grad():
            features = self.extract_features(sequences)
            
            user_ids = None
            if self.reward_model.use_user_conditioning and user_id is not None:
                user_ids = torch.tensor([user_id] * len(sequences), dtype=torch.long, device=self.device)
            
            rewards = self.reward_model(features, user_ids=user_ids)
        
        return rewards
    
    @abstractmethod
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            batch: Batch of training data
            
        Returns:
            Dictionary of metrics
        """
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: Path) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: Path) -> None:
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        pass


class PersonalizedRewardWrapper:
    """
    Wrapper that combines a base reward function with a personalized reward model.
    
    This allows you to blend task-specific rewards (e.g., from a classifier)
    with user-specific preferences.
    
    Args:
        base_reward_fn: Base reward function (e.g., activity classifier)
        preference_reward_model: Trained preference reward model
        feature_extractor: Function to extract features for preference model
        blend_weight: Weight for blending (0=only base, 1=only preference)
        device: Torch device
    """
    
    def __init__(
        self,
        base_reward_fn: Callable,
        preference_reward_model: PreferenceRewardModel,
        feature_extractor: Callable,
        blend_weight: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.base_reward_fn = base_reward_fn
        self.preference_reward_model = preference_reward_model
        self.feature_extractor = feature_extractor
        self.blend_weight = blend_weight
        self.device = torch.device(device)
        self.preference_reward_model.eval()
    
    def __call__(
        self,
        sequences: list[str],
        user_id: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute blended rewards.
        
        Args:
            sequences: List of sequences
            user_id: Optional user ID
            
        Returns:
            Tuple of (rewards, mask) matching the base_reward_fn signature
        """
        # Get base rewards
        base_rewards, mask = self.base_reward_fn(sequences)
        
        # Get preference-based rewards
        with torch.no_grad():
            features = self.feature_extractor(sequences)
            
            user_ids = None
            if self.preference_reward_model.use_user_conditioning and user_id is not None:
                user_ids = torch.tensor([user_id] * len(sequences), dtype=torch.long, device=self.device)
            
            pref_rewards = self.preference_reward_model(features, user_ids=user_ids)
        
        # Blend rewards
        blended_rewards = (1 - self.blend_weight) * base_rewards + self.blend_weight * pref_rewards
        
        return blended_rewards, mask


def create_user_conditioned_reward_fn(
    preference_reward_model: PreferenceRewardModel,
    feature_extractor: Callable,
    user_id: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Callable:
    """
    Create a reward function conditioned on a specific user.
    
    Args:
        preference_reward_model: Trained preference reward model
        feature_extractor: Function to extract features
        user_id: ID of the user to condition on
        device: Torch device
        
    Returns:
        Callable reward function
    """
    device = torch.device(device)
    preference_reward_model.eval()
    
    def reward_fn(sequences: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            features = feature_extractor(sequences)
            
            user_ids = None
            if preference_reward_model.use_user_conditioning:
                user_ids = torch.tensor([user_id] * len(sequences), dtype=torch.long, device=device)
            
            rewards = preference_reward_model(features, user_ids=user_ids)
            mask = torch.ones_like(rewards, dtype=torch.bool)
        
        return rewards, mask
    
    return reward_fn

