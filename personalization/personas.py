"""
Persona definitions for personalized protein design.

Each persona is defined by a weight vector w^(u) that specifies preferences
over biological properties: [activity, toxicity, stability, length].

Personalized rewards are computed as a simple dot product:
    R^(u)(x) = w^(u)^T · g(x)

where g(x) = [p_act, p_tox, p_stab, p_len] is the property vector.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import torch
import numpy as np
import pandas as pd


@dataclass
class Persona:
    """
    A persona representing a specific user preference profile.
    
    Attributes:
        name: Unique identifier for the persona
        weights: Weight vector [w_act, w_tox, w_stab, w_len]
        description: Human-readable description of preferences
        properties_explained: Optional dict explaining what each weight means
    """
    name: str
    weights: torch.Tensor  # Shape: (4,) for [activity, toxicity, stability, length]
    description: str
    properties_explained: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        """Ensure weights is a tensor."""
        if not isinstance(self.weights, torch.Tensor):
            self.weights = torch.tensor(self.weights, dtype=torch.float32)
        
        if self.weights.shape != (4,):
            raise ValueError(f"Weights must have shape (4,), got {self.weights.shape}")
    
    def get_weight_dict(self) -> Dict[str, float]:
        """Return weights as a dictionary for interpretability."""
        return {
            'activity': float(self.weights[0]),
            'toxicity': float(self.weights[1]),
            'stability': float(self.weights[2]),
            'length': float(self.weights[3]),
        }
    
    def explain(self) -> str:
        """Generate an explanation of this persona's preferences."""
        weights = self.get_weight_dict()
        
        explanation = f"Persona: {self.name}\n"
        explanation += f"Description: {self.description}\n\n"
        explanation += "Property Weights:\n"
        
        for prop, weight in weights.items():
            sign = "+" if weight >= 0 else ""
            explanation += f"  {prop:12s}: {sign}{weight:6.2f}"
            
            if weight > 0.7:
                explanation += " (strongly prefers high)"
            elif weight > 0.3:
                explanation += " (prefers high)"
            elif weight > -0.3:
                explanation += " (neutral)"
            elif weight > -0.7:
                explanation += " (avoids high)"
            else:
                explanation += " (strongly avoids high)"
            
            explanation += "\n"
        
        return explanation


# ============================================================================
# Pre-defined Personas for AMP Design
# ============================================================================

PERSONAS: Dict[str, Persona] = {
    "PotencyMaximizer": Persona(
        name="PotencyMaximizer",
        weights=torch.tensor([1.0, 0.0, 0.3, 0.0]),
        description="Prioritizes maximum antimicrobial activity above all else. "
                   "Neutral on toxicity and slightly favors stability.",
        properties_explained={
            'activity': "Maximize: highest weight (1.0)",
            'toxicity': "Ignore: zero weight",
            'stability': "Moderate preference: 0.3",
            'length': "Neutral: 0.0",
        }
    ),
    
    "SafetyFirst": Persona(
        name="SafetyFirst",
        weights=torch.tensor([0.5, -1.0, 0.5, -0.2]),
        description="Strongly emphasizes safety (low toxicity) while maintaining "
                   "moderate activity. Prefers stable and shorter peptides.",
        properties_explained={
            'activity': "Moderate requirement: 0.5",
            'toxicity': "Strongly penalize: -1.0",
            'stability': "Prefer stable: 0.5",
            'length': "Slight preference for shorter: -0.2",
        }
    ),
    
    "BalancedDesigner": Persona(
        name="BalancedDesigner",
        weights=torch.tensor([0.7, -0.5, 0.6, -0.1]),
        description="Seeks a balanced trade-off between activity, safety, and stability. "
                   "The default choice for most applications.",
        properties_explained={
            'activity': "High preference: 0.7",
            'toxicity': "Moderate penalty: -0.5",
            'stability': "High preference: 0.6",
            'length': "Slight preference for shorter: -0.1",
        }
    ),
    
    "StabilityFocused": Persona(
        name="StabilityFocused",
        weights=torch.tensor([0.4, -0.3, 1.0, 0.0]),
        description="Maximizes protein stability with moderate activity requirements. "
                   "Good for therapeutic applications requiring long shelf life.",
        properties_explained={
            'activity': "Moderate requirement: 0.4",
            'toxicity': "Moderate penalty: -0.3",
            'stability': "Maximize: 1.0",
            'length': "Neutral: 0.0",
        }
    ),
    
    "ShortPeptideFan": Persona(
        name="ShortPeptideFan",
        weights=torch.tensor([0.6, -0.4, 0.3, -0.8]),
        description="Strong preference for short sequences (easier synthesis, lower cost). "
                   "Maintains good activity while avoiding toxicity.",
        properties_explained={
            'activity': "Good activity needed: 0.6",
            'toxicity': "Avoid: -0.4",
            'stability': "Some stability: 0.3",
            'length': "Strongly prefer short: -0.8",
        }
    ),
    
    "TherapeuticOptimizer": Persona(
        name="TherapeuticOptimizer",
        weights=torch.tensor([0.6, -0.9, 0.8, -0.3]),
        description="Optimized for therapeutic use: strong activity, very low toxicity, "
                   "high stability, moderate length preference.",
        properties_explained={
            'activity': "Strong activity: 0.6",
            'toxicity': "Very low toxicity critical: -0.9",
            'stability': "High stability needed: 0.8",
            'length': "Prefer shorter for formulation: -0.3",
        }
    ),
    
    "ResearchExplorer": Persona(
        name="ResearchExplorer",
        weights=torch.tensor([0.8, 0.0, 0.2, 0.0]),
        description="For research purposes: prioritize activity discovery, "
                   "neutral on other properties to explore diverse space.",
        properties_explained={
            'activity': "High activity for discovery: 0.8",
            'toxicity': "Neutral for exploration: 0.0",
            'stability': "Minor consideration: 0.2",
            'length': "No constraint: 0.0",
        }
    ),
}


def get_persona(name: str) -> Persona:
    """
    Retrieve a persona by name.
    
    Args:
        name: Persona name (case-sensitive)
    
    Returns:
        Persona object
    
    Raises:
        KeyError: If persona name not found
    """
    if name not in PERSONAS:
        available = ", ".join(PERSONAS.keys())
        raise KeyError(
            f"Persona '{name}' not found. Available personas: {available}"
        )
    return PERSONAS[name]


def list_personas() -> list[str]:
    """Return list of available persona names."""
    return list(PERSONAS.keys())


def compute_personalized_reward(
    properties: torch.Tensor,
    persona: Persona,
) -> torch.Tensor:
    """
    Compute personalized reward as R^(u)(x) = w^(u)^T · g(x).
    
    This is the core personalization function: a simple dot product
    between user weights and property predictions.
    
    Args:
        properties: Tensor of shape (batch_size, 4) with property vectors
                   [p_act, p_tox, p_stab, p_len]
        persona: Persona object containing weight vector w^(u)
    
    Returns:
        rewards: Tensor of shape (batch_size,) with personalized rewards
    
    Example:
        >>> properties = torch.tensor([[0.9, 0.3, 0.7, 0.5]])  # 1 sequence
        >>> persona = get_persona("SafetyFirst")
        >>> reward = compute_personalized_reward(properties, persona)
        >>> # reward = 0.5 * 0.9 + (-1.0) * 0.3 + 0.5 * 0.7 + (-0.2) * 0.5
        >>> # reward = 0.45 - 0.3 + 0.35 - 0.1 = 0.4
    """
    if properties.shape[1] != 4:
        raise ValueError(
            f"Properties must have shape (batch_size, 4), got {properties.shape}"
        )
    
    # Move weights to same device as properties
    weights = persona.weights.to(properties.device)
    
    # Compute dot product: R = w^T · g
    rewards = torch.matmul(properties, weights)
    
    return rewards


def compute_multi_persona_rewards(
    properties: torch.Tensor,
    persona_names: list[str],
) -> Dict[str, torch.Tensor]:
    """
    Compute rewards for multiple personas at once.
    
    Useful for comparing how different personas would rank sequences.
    
    Args:
        properties: Tensor of shape (batch_size, 4)
        persona_names: List of persona names
    
    Returns:
        Dictionary mapping persona names to reward tensors
    """
    rewards_dict = {}
    
    for name in persona_names:
        persona = get_persona(name)
        rewards = compute_personalized_reward(properties, persona)
        rewards_dict[name] = rewards
    
    return rewards_dict


def explain_reward(
    properties: torch.Tensor,
    persona: Persona,
    sequence: Optional[str] = None,
) -> str:
    """
    Generate a human-readable explanation of why a sequence got its reward.
    
    Args:
        properties: Property vector (4,) for a single sequence
        persona: Persona used for reward computation
        sequence: Optional sequence string for display
    
    Returns:
        Explanation string
    """
    if properties.dim() > 1:
        properties = properties.squeeze()
    
    if properties.shape[0] != 4:
        raise ValueError(f"Expected 4 properties, got {properties.shape[0]}")
    
    # Compute reward
    reward = compute_personalized_reward(properties.unsqueeze(0), persona).item()
    
    # Build explanation
    explanation = []
    
    if sequence:
        explanation.append(f"Sequence: {sequence}\n")
    
    explanation.append(f"Persona: {persona.name}\n")
    explanation.append(f"Total Reward: {reward:.4f}\n\n")
    explanation.append("Property Contributions:\n")
    
    property_names = ['Activity', 'Toxicity', 'Stability', 'Length']
    weights = persona.weights.cpu().numpy()
    props = properties.cpu().numpy()
    
    for i, (name, weight, value) in enumerate(zip(property_names, weights, props)):
        contribution = weight * value
        explanation.append(
            f"  {name:12s}: {value:6.3f} × {weight:6.2f} = {contribution:7.4f}\n"
        )
    
    explanation.append(f"\n  {'Total':12s}:               = {reward:7.4f}")
    
    return "".join(explanation)


def create_custom_persona(
    name: str,
    activity_weight: float = 0.5,
    toxicity_weight: float = -0.5,
    stability_weight: float = 0.5,
    length_weight: float = 0.0,
    description: str = "Custom persona",
) -> Persona:
    """
    Create a custom persona with specified weights.
    
    Args:
        name: Unique name for the persona
        activity_weight: Weight for activity (typically positive)
        toxicity_weight: Weight for toxicity (typically negative)
        stability_weight: Weight for stability (typically positive)
        length_weight: Weight for length (negative = prefer shorter)
        description: Description of the persona's preferences
    
    Returns:
        Custom Persona object
    """
    weights = torch.tensor([
        activity_weight,
        toxicity_weight,
        stability_weight,
        length_weight,
    ], dtype=torch.float32)
    
    return Persona(
        name=name,
        weights=weights,
        description=description,
    )


def register_persona(persona: Persona) -> None:
    """
    Register a custom persona to the global PERSONAS dictionary.
    
    Args:
        persona: Persona to register
    """
    if persona.name in PERSONAS:
        print(f"Warning: Overwriting existing persona '{persona.name}'")
    
    PERSONAS[persona.name] = persona


# ============================================================================
# Utility Functions for Analysis
# ============================================================================

def compare_personas_on_sequence(
    properties: torch.Tensor,
    persona_names: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Compare how different personas would rank a set of sequences.
    
    Args:
        properties: Tensor of shape (batch_size, 4)
        persona_names: List of persona names to compare (default: all)
    
    Returns:
        DataFrame with columns for each persona's rewards
    """
    import pandas as pd
    
    if persona_names is None:
        persona_names = list_personas()
    
    rewards_dict = compute_multi_persona_rewards(properties, persona_names)
    
    # Convert to DataFrame
    df = pd.DataFrame(rewards_dict)
    df.index.name = 'sequence_idx'
    
    return df


def visualize_persona_weights(persona_names: Optional[list[str]] = None):
    """
    Visualize weight vectors for multiple personas.
    
    Requires matplotlib. Useful for understanding persona differences.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib required for visualization")
        return
    
    if persona_names is None:
        persona_names = list_personas()
    
    property_names = ['Activity', 'Toxicity', 'Stability', 'Length']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(property_names))
    width = 0.8 / len(persona_names)
    
    for i, name in enumerate(persona_names):
        persona = get_persona(name)
        weights = persona.weights.cpu().numpy()
        
        offset = (i - len(persona_names) / 2) * width
        ax.bar(x + offset, weights, width, label=name)
    
    ax.set_xlabel('Property')
    ax.set_ylabel('Weight')
    ax.set_title('Persona Weight Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(property_names)
    ax.legend()
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Demo: print all personas
    print("=" * 80)
    print("Available Personas for AMP Design")
    print("=" * 80)
    print()
    
    for name in list_personas():
        persona = get_persona(name)
        print(persona.explain())
        print()

