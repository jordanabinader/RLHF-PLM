"""
Preference-based reward models using Bradley-Terry framework.

This module implements reward models that learn from pairwise preferences
rather than direct reward labels, enabling personalized RLHF.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class PreferenceRewardModel(nn.Module):
    """
    Neural network that predicts scalar rewards from sequence features.
    
    Can be trained using Bradley-Terry loss on pairwise preferences.
    Supports optional user conditioning for personalized rewards.
    
    Args:
        input_dim: Dimension of input features
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout probability
        num_users: Number of users (for user-conditioned reward model)
        user_embed_dim: Dimension of user embeddings
        use_user_conditioning: Whether to condition rewards on user identity
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [128, 64],
        dropout: float = 0.1,
        num_users: Optional[int] = None,
        user_embed_dim: int = 32,
        use_user_conditioning: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.use_user_conditioning = use_user_conditioning
        
        # User embeddings (if personalized)
        if use_user_conditioning and num_users is not None:
            self.user_embeddings = nn.Embedding(num_users, user_embed_dim)
            effective_input = input_dim + user_embed_dim
        else:
            self.user_embeddings = None
            effective_input = input_dim
        
        # Build MLP
        layers = []
        prev_dim = effective_input
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(
        self,
        features: torch.Tensor,
        user_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute reward scores.
        
        Args:
            features: Tensor of shape (batch_size, input_dim)
            user_ids: Optional tensor of shape (batch_size,) with user IDs
            
        Returns:
            Tensor of shape (batch_size,) with reward scores
        """
        x = features
        
        # Concatenate user embeddings if using personalization
        if self.use_user_conditioning:
            if user_ids is None:
                raise ValueError("user_ids must be provided when use_user_conditioning=True")
            user_embeds = self.user_embeddings(user_ids)
            x = torch.cat([x, user_embeds], dim=-1)
        
        return self.network(x).squeeze(-1)


def bradley_terry_loss(
    r_a: torch.Tensor,
    r_b: torch.Tensor,
    pref: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Bradley-Terry preference loss.
    
    For a pair (a, b) with label pref:
        p = sigmoid(r_a - r_b)
        loss = -[pref * log(p) + (1-pref) * log(1-p)]
    
    Args:
        r_a: Rewards for option A, shape (batch_size,)
        r_b: Rewards for option B, shape (batch_size,)
        pref: Preference labels (1 if A preferred, 0 if B preferred), shape (batch_size,)
        
    Returns:
        Scalar loss
    """
    logits = r_a - r_b
    p = torch.sigmoid(logits)
    eps = 1e-8
    loss = -torch.mean(
        pref * torch.log(p + eps) + (1 - pref) * torch.log(1 - p + eps)
    )
    return loss


def train_preference_reward_model(
    model: PreferenceRewardModel,
    dataloader: DataLoader,
    num_epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
) -> Tuple[PreferenceRewardModel, list]:
    """
    Train a preference reward model using Bradley-Terry loss.
    
    Args:
        model: PreferenceRewardModel to train
        dataloader: DataLoader yielding batches with keys:
                   'feat_a', 'feat_b', 'pref', optionally 'user_id'
        num_epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for regularization
        device: Device to train on
        verbose: Whether to print progress
        
    Returns:
        Tuple of (trained_model, loss_history)
    """
    device = torch.device(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    loss_history = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") if verbose else dataloader
        
        for batch in pbar:
            feat_a = batch["feat_a"].to(device)
            feat_b = batch["feat_b"].to(device)
            pref = batch["pref"].to(device)
            
            user_ids = batch.get("user_id")
            if user_ids is not None:
                user_ids = user_ids.to(device)
            
            # Forward pass
            r_a = model(feat_a, user_ids=user_ids)
            r_b = model(feat_b, user_ids=user_ids)
            
            # Compute loss
            loss = bradley_terry_loss(r_a, r_b, pref)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            if verbose and isinstance(pbar, tqdm):
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = np.mean(epoch_losses)
        loss_history.append(avg_loss)
        
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")
    
    return model, loss_history


def evaluate_reward_model(
    model: PreferenceRewardModel,
    dataloader: DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """
    Evaluate a preference reward model.
    
    Args:
        model: Trained PreferenceRewardModel
        dataloader: Evaluation DataLoader
        device: Device for evaluation
        
    Returns:
        Dictionary with evaluation metrics
    """
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            feat_a = batch["feat_a"].to(device)
            feat_b = batch["feat_b"].to(device)
            pref = batch["pref"].to(device)
            
            user_ids = batch.get("user_id")
            if user_ids is not None:
                user_ids = user_ids.to(device)
            
            r_a = model(feat_a, user_ids=user_ids)
            r_b = model(feat_b, user_ids=user_ids)
            
            loss = bradley_terry_loss(r_a, r_b, pref)
            total_loss += loss.item() * len(pref)
            
            # Compute accuracy
            predicted_pref = (r_a > r_b).float()
            correct += (predicted_pref == pref).sum().item()
            total += len(pref)
    
    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


class EnsembleRewardModel(nn.Module):
    """
    Ensemble of multiple reward models for uncertainty estimation.
    
    Args:
        models: List of PreferenceRewardModel instances
    """
    
    def __init__(self, models: list[PreferenceRewardModel]):
        super().__init__()
        self.models = nn.ModuleList(models)
    
    def forward(
        self,
        features: torch.Tensor,
        user_ids: Optional[torch.Tensor] = None,
        return_uncertainty: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute ensemble reward predictions.
        
        Args:
            features: Input features
            user_ids: Optional user IDs
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Mean rewards (and optionally standard deviations)
        """
        rewards = []
        for model in self.models:
            r = model(features, user_ids=user_ids)
            rewards.append(r)
        
        rewards = torch.stack(rewards, dim=0)  # (num_models, batch_size)
        mean_rewards = rewards.mean(dim=0)
        
        if return_uncertainty:
            std_rewards = rewards.std(dim=0)
            return mean_rewards, std_rewards
        
        return mean_rewards

