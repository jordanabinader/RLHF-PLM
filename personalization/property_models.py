"""
Property Head Models for Shared Property Architecture

This module defines the property prediction heads:
- ActivityHead: Predicts antimicrobial activity (p_act)
- ToxicityHead: Predicts toxicity probability (p_tox)
- StabilityHead: Predicts stability score (p_stab)

Each head operates on ESM-2 embeddings as input.
"""

import torch
import torch.nn as nn
from pathlib import Path


class ActivityHead(nn.Module):
    """
    Wrapper for existing AMP activity classifier (best_new_4.pth).
    
    Predicts p_act ∈ [0, 1]: probability of antimicrobial activity.
    """
    def __init__(self, checkpoint_path: str):
        super().__init__()
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Activity classifier not found at {checkpoint_path}")
        
        # Load the pre-trained classifier
        self.classifier = torch.load(checkpoint_path, map_location='cpu')
        self.classifier.eval()
        
        # Freeze parameters - this is a fixed property predictor
        for param in self.classifier.parameters():
            param.requires_grad = False
    
    def forward(self, esm_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            esm_embedding: (batch_size, esm_dim) tensor
        
        Returns:
            p_act: (batch_size,) tensor with activity probabilities
        """
        with torch.no_grad():
            output = self.classifier(esm_embedding)
            if output.dim() > 1:
                output = output.squeeze(-1)
            # Ensure output is in [0, 1] range
            output = torch.sigmoid(output) if not torch.all((output >= 0) & (output <= 1)) else output
            return output


class ToxicityHead(nn.Module):
    """
    MLP classifier for predicting protein toxicity.
    
    Takes ESM embeddings + domain vectors as input.
    Predicts p_tox ∈ [0, 1]: probability of toxicity.
    
    Architecture matches ToxDL2's dense module but without GCN component.
    """
    def __init__(self, esm_dim: int = 1280, domain_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.esm_dim = esm_dim
        self.domain_dim = domain_dim
        
        # MLP similar to ToxDL2's combine module
        self.mlp = nn.Sequential(
            nn.Linear(esm_dim + domain_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, esm_embedding: torch.Tensor, domain_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            esm_embedding: (batch_size, esm_dim) tensor
            domain_vector: (batch_size, domain_dim) tensor
        
        Returns:
            p_tox: (batch_size,) tensor with toxicity probabilities
        """
        # Concatenate ESM embedding with domain vector
        combined = torch.cat([esm_embedding, domain_vector], dim=-1)
        output = self.mlp(combined)
        return output.squeeze(-1)


class StabilityHead(nn.Module):
    """
    MLP regressor for predicting protein stability.
    
    Takes ESM embeddings as input.
    Predicts p_stab: continuous stability score (typically ΔΔG or similar).
    
    Architecture follows EsmTherm's MLP design.
    """
    def __init__(self, esm_dim: int = 1280, hidden_dim: int = 512, n_layers: int = 2):
        super().__init__()
        self.esm_dim = esm_dim
        
        # Build MLP layers
        layers = []
        layers.append(nn.Linear(esm_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, esm_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            esm_embedding: (batch_size, esm_dim) tensor
        
        Returns:
            p_stab: (batch_size,) tensor with stability scores
        """
        output = self.mlp(esm_embedding)
        return output.squeeze(-1)


def load_activity_head(checkpoint_path: str, device: str = "cuda") -> ActivityHead:
    """Load pre-trained activity head."""
    model = ActivityHead(checkpoint_path)
    model = model.to(device)
    model.eval()
    return model


def load_toxicity_head(checkpoint_path: str, device: str = "cuda") -> ToxicityHead:
    """Load trained toxicity head."""
    model = ToxicityHead()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle both direct state dict and wrapped checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    return model


def load_stability_head(checkpoint_path: str, device: str = "cuda") -> StabilityHead:
    """Load trained stability head."""
    model = StabilityHead()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle both direct state dict and wrapped checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    return model

