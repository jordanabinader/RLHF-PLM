"""
Personalized RLHF Module for Protein Design

This module provides infrastructure for:
- Synthetic user preference generation (old)
- Preference-based reward model training (old)
- Personalized RL training loops (old)

NEW: Shared Property Model Architecture
- property_models: Neural networks for biological properties
- unified_property_fn: g(x) = [p_act, p_tox, p_stab, p_len]
- personas: Lightweight weight vectors w^(u)
- Reward: R^(u)(x) = w^(u)^T · g(x)
"""

# Old architecture (backwards compatibility)
from .synthetic_users import SyntheticUser, compute_property_scores
from .preference_generation import generate_pairwise_preferences, PreferenceDataset
from .preference_reward_model import PreferenceRewardModel, train_preference_reward_model
from .personalized_trainer import PersonalizedRLTrainer

# New shared property model architecture
from .property_models import (
    ActivityHead,
    ToxicityHead,
    StabilityHead,
    load_activity_head,
    load_toxicity_head,
    load_stability_head,
)
from .unified_property_fn import (
    UnifiedPropertyFunction,
    create_unified_property_function,
)
from .personas import (
    Persona,
    get_persona,
    list_personas,
    compute_personalized_reward,
    create_custom_persona,
    explain_reward,
)
from .validity import (
    is_sequence_valid,
    validate_sequences,
    calculate_net_charge,
    get_validity_stats,
)
from .hybrid_reward import (
    create_hybrid_reward_fn,
    create_blended_hybrid_reward_fn,
)
from .user_conditioned_policy import (
    UserContextProjector,
    UserConditionedPolicyWrapper,
)

__all__ = [
    # Old architecture
    "SyntheticUser",
    "define_synthetic_users",
    "compute_property_scores",
    "generate_pairwise_preferences",
    "PreferenceDataset",
    "PreferenceRewardModel",
    "train_preference_reward_model",
    "PersonalizedRLTrainer",
    # New shared property model
    "ActivityHead",
    "ToxicityHead",
    "StabilityHead",
    "load_activity_head",
    "load_toxicity_head",
    "load_stability_head",
    "UnifiedPropertyFunction",
    "create_unified_property_function",
    "Persona",
    "get_persona",
    "list_personas",
    "compute_personalized_reward",
    "create_custom_persona",
    "explain_reward",
    # User-conditioned policy
    "is_sequence_valid",
    "validate_sequences",
    "calculate_net_charge",
    "get_validity_stats",
    "create_hybrid_reward_fn",
    "create_blended_hybrid_reward_fn",
    "UserContextProjector",
    "UserConditionedPolicyWrapper",
]

