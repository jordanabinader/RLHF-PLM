"""
Synthetic user definitions and property score computation.

This module allows you to define virtual users with different property trade-offs
(e.g., prioritizing activity vs. safety, short vs. long sequences, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Callable, Optional
import numpy as np
import pandas as pd


@dataclass
class SyntheticUser:
    """
    Represents a virtual user with specific property preferences.
    
    Attributes:
        name: Human-readable identifier for the user
        weights: Dictionary mapping property names to preference weights
        description: Optional description of the user's preferences
    """
    name: str
    weights: Dict[str, float]
    description: str = ""
    
    def compute_reward(self, properties: Dict[str, float]) -> float:
        """
        Compute the reward for a sequence based on user preferences.
        
        Args:
            properties: Dictionary of property values for a sequence
            
        Returns:
            Scalar reward value
        """
        reward = 0.0
        for prop_name, weight in self.weights.items():
            if prop_name in properties:
                reward += weight * properties[prop_name]
        return reward


def define_synthetic_users_amp() -> List[SyntheticUser]:
    """
    Define synthetic users for antimicrobial peptide design.
    
    Returns:
        List of SyntheticUser objects with different trade-offs
    """
    users = [
        SyntheticUser(
            name="PotencyMaximizer",
            weights={
                "activity_score": +1.0,
                "toxicity_score": 0.0,
                "stability_score": +0.3,
                "length": 0.0,
                "hydrophobicity": +0.2,
                "charge": +0.1,
            },
            description="Prioritizes antimicrobial activity above all else",
        ),
        SyntheticUser(
            name="SafetyFirst",
            weights={
                "activity_score": +0.5,
                "toxicity_score": -1.0,
                "stability_score": +0.5,
                "length": -0.2,
                "hydrophobicity": 0.0,
                "charge": +0.2,
            },
            description="Prioritizes low toxicity and safety",
        ),
        SyntheticUser(
            name="BalancedDesigner",
            weights={
                "activity_score": +0.7,
                "toxicity_score": -0.5,
                "stability_score": +0.6,
                "length": -0.1,
                "hydrophobicity": 0.0,
                "charge": +0.15,
            },
            description="Balanced approach to multiple objectives",
        ),
        SyntheticUser(
            name="ShortPeptideFan",
            weights={
                "activity_score": +0.7,
                "toxicity_score": -0.5,
                "stability_score": 0.0,
                "length": -0.8,
                "hydrophobicity": +0.1,
                "charge": +0.2,
            },
            description="Prefers shorter peptides with good activity",
        ),
        SyntheticUser(
            name="StabilityFocused",
            weights={
                "activity_score": +0.4,
                "toxicity_score": -0.3,
                "stability_score": +1.0,
                "length": 0.0,
                "hydrophobicity": +0.3,
                "charge": 0.0,
            },
            description="Prioritizes stability and manufacturability",
        ),
    ]
    return users


def define_synthetic_users_antibody() -> List[SyntheticUser]:
    """
    Define synthetic users for antibody mutation design.
    
    Returns:
        List of SyntheticUser objects with different trade-offs
    """
    users = [
        SyntheticUser(
            name="AffinityMaximizer",
            weights={
                "binding_affinity": +1.0,
                "stability": +0.2,
                "immunogenicity": -0.1,
                "num_mutations": 0.0,
                "developability": +0.3,
            },
            description="Maximizes binding affinity",
        ),
        SyntheticUser(
            name="ConservativeMutator",
            weights={
                "binding_affinity": +0.6,
                "stability": +0.5,
                "immunogenicity": -0.5,
                "num_mutations": -0.8,
                "developability": +0.6,
            },
            description="Prefers fewer mutations with good properties",
        ),
        SyntheticUser(
            name="DevelopabilityFocused",
            weights={
                "binding_affinity": +0.5,
                "stability": +0.7,
                "immunogenicity": -0.9,
                "num_mutations": -0.3,
                "developability": +1.0,
            },
            description="Optimizes for clinical developability",
        ),
        SyntheticUser(
            name="StabilityEnhancer",
            weights={
                "binding_affinity": +0.4,
                "stability": +1.0,
                "immunogenicity": -0.4,
                "num_mutations": -0.2,
                "developability": +0.5,
            },
            description="Focuses on improving stability",
        ),
    ]
    return users


def define_synthetic_users_kinase() -> List[SyntheticUser]:
    """
    Define synthetic users for kinase mutation design.
    
    Returns:
        List of SyntheticUser objects with different trade-offs
    """
    users = [
        SyntheticUser(
            name="ActivityBooster",
            weights={
                "fitness": +1.0,
                "stability": +0.2,
                "num_mutations": 0.0,
                "conservation": -0.1,
            },
            description="Maximizes kinase activity/fitness",
        ),
        SyntheticUser(
            name="ConservedEngineer",
            weights={
                "fitness": +0.6,
                "stability": +0.5,
                "num_mutations": -0.7,
                "conservation": +0.8,
            },
            description="Prefers conservative mutations",
        ),
        SyntheticUser(
            name="BalancedOptimizer",
            weights={
                "fitness": +0.7,
                "stability": +0.6,
                "num_mutations": -0.3,
                "conservation": +0.2,
            },
            description="Balanced fitness and stability",
        ),
    ]
    return users


def compute_property_scores(
    sequences: List[str],
    property_functions: Dict[str, Callable],
    base_properties: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute property scores for a list of sequences.
    
    Args:
        sequences: List of protein/peptide sequences
        property_functions: Dictionary mapping property names to computation functions.
                          Each function should take a sequence string and return a float.
        base_properties: Optional DataFrame with pre-computed properties (e.g., from ML models)
        
    Returns:
        DataFrame with sequences and their property scores
    """
    df = pd.DataFrame({"sequence": sequences})
    
    # Merge with base properties if provided
    if base_properties is not None:
        df = df.merge(base_properties, on="sequence", how="left")
    
    # Compute additional properties
    for prop_name, prop_fn in property_functions.items():
        if prop_name not in df.columns:
            df[prop_name] = df["sequence"].apply(prop_fn)
    
    return df


# Example property computation functions (can be replaced with real models)
def compute_length(seq: str) -> float:
    """Return sequence length."""
    return float(len(seq))


def compute_hydrophobicity(seq: str) -> float:
    """Compute average hydrophobicity using Kyte-Doolittle scale."""
    KYTE_DOOLITTLE = {
        'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
        'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
        'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
        'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
    }
    if not seq:
        return 0.0
    values = [KYTE_DOOLITTLE.get(aa, 0.0) for aa in seq]
    return float(np.mean(values)) if values else 0.0


def compute_net_charge(seq: str) -> float:
    """Compute net charge at pH 7."""
    CHARGE_DICT = {
        'K': 1.0, 'R': 1.0, 'H': 0.5,   # basic
        'D': -1.0, 'E': -1.0             # acidic
    }
    return float(sum(CHARGE_DICT.get(aa, 0.0) for aa in seq))


def compute_stability_proxy(seq: str) -> float:
    """
    Simple stability proxy based on hydrophobicity and length.
    Replace with real stability prediction models.
    """
    hydro = compute_hydrophobicity(seq)
    length = len(seq)
    # Favor moderate hydrophobicity and moderate length
    return -(abs(hydro - 0.5) + abs(length - 20) / 20.0)

