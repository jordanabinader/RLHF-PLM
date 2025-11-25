"""
Personalized RLHF Module for Protein Design

This module provides infrastructure for:
- Synthetic user preference generation
- Preference-based reward model training
- Personalized RL training loops
"""

from .synthetic_users import SyntheticUser, define_synthetic_users, compute_property_scores
from .preference_generation import generate_pairwise_preferences, PreferenceDataset
from .preference_reward_model import PreferenceRewardModel, train_preference_reward_model
from .personalized_trainer import PersonalizedRLTrainer

__all__ = [
    "SyntheticUser",
    "define_synthetic_users",
    "compute_property_scores",
    "generate_pairwise_preferences",
    "PreferenceDataset",
    "PreferenceRewardModel",
    "train_preference_reward_model",
    "PersonalizedRLTrainer",
]

