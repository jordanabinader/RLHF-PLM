"""
User-conditioned policy wrapper for GRPO training.

This module implements a wrapper around the base policy that projects
user preference weights w^(u) and conditions sequence generation on them.
"""
import torch
import torch.nn as nn
from typing import Optional


class UserContextProjector(nn.Module):
    """
    Projects 4D user weights w^(u) to high-dimensional embedding space.
    
    The projector is a small MLP that learns to map user preferences
    into a representation compatible with the policy's hidden states.
    """
    
    def __init__(self, user_dim: int = 4, hidden_dim: int = 128, output_dim: int = 256):
        """
        Initialize user context projector.
        
        Args:
            user_dim: Dimension of user weight vector (default: 4 for [act, tox, stab, len])
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (projection space)
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(user_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, user_weights: torch.Tensor) -> torch.Tensor:
        """
        Project user weights to embedding space.
        
        Args:
            user_weights: (batch_size, 4) or (4,) tensor of user weights
        
        Returns:
            projected: (batch_size, output_dim) or (output_dim,) tensor
        """
        return self.mlp(user_weights)


class UserConditionedPolicyWrapper(nn.Module):
    """
    Wraps a base policy to accept user context during generation.
    
    This is a simplified implementation that stores user context
    and makes it available to the policy. For full integration,
    this would be modified to inject user embeddings into the
    attention mechanism or as prepended tokens.
    """
    
    def __init__(
        self, 
        base_policy,
        user_dim: int = 4,
        projection_dim: int = 256,
    ):
        """
        Initialize user-conditioned policy wrapper.
        
        Args:
            base_policy: Base language model (ProGen, ESM, etc.)
            user_dim: Dimension of user weight vector
            projection_dim: Dimension of projected user embeddings
        """
        super().__init__()
        self.base_policy = base_policy
        self.user_projector = UserContextProjector(
            user_dim=user_dim,
            hidden_dim=128,
            output_dim=projection_dim
        )
        self.projection_dim = projection_dim
        self.current_user_context = None
    
    def forward(self, input_ids, user_context: Optional[torch.Tensor] = None, **kwargs):
        """
        Forward pass with optional user context.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            user_context: Optional user weights (batch_size, 4) or (4,)
            **kwargs: Additional arguments for base policy
        
        Returns:
            Model outputs from base policy
        """
        # If no user context provided, use default (balanced designer)
        if user_context is None:
            batch_size = input_ids.shape[0]
            # Default to BalancedDesigner weights: [0.7, -0.5, 0.6, -0.1]
            user_context = torch.tensor(
                [0.7, -0.5, 0.6, -0.1], 
                device=input_ids.device
            ).repeat(batch_size, 1)
        
        # Project user context: (batch_size, 4) -> (batch_size, projection_dim)
        user_embed = self.user_projector(user_context)
        
        # Store user context for generation (accessed during sampling)
        self.current_user_context = user_embed
        
        # Forward through base policy
        # Note: In a full implementation, user_embed would be injected
        # into the model's attention or prepended as special tokens
        return self.base_policy(input_ids, **kwargs)
    
    def generate(
        self,
        input_ids,
        user_context: torch.Tensor,
        max_length: int = None,
        max_new_tokens: int = None,
        **generation_kwargs
    ):
        """
        Generate sequences conditioned on user context.
        
        Args:
            input_ids: Starting tokens (batch_size, seq_len)
            user_context: User weights (batch_size, 4) or (4,)
            max_length: Maximum total length (deprecated, use max_new_tokens)
            max_new_tokens: Maximum number of new tokens to generate
            **generation_kwargs: Additional args for base_policy.generate()
        
        Returns:
            Generated token IDs
        """
        # Handle single user context for batch
        if user_context.dim() == 1:
            batch_size = input_ids.shape[0]
            user_context = user_context.unsqueeze(0).repeat(batch_size, 1)
        
        # Project user context
        user_embed = self.user_projector(user_context)
        
        # Store for forward pass
        self.current_user_context = user_embed
        
        # Generate using base policy
        # Note: This is a simplified version. For full integration, we would
        # modify the generation loop to inject user_embed at each step or
        # prepend it as special tokens that the model can attend to
        
        gen_args = generation_kwargs.copy()
        if max_new_tokens is not None:
            gen_args['max_new_tokens'] = max_new_tokens
        elif max_length is not None:
            gen_args['max_length'] = max_length
        
        return self.base_policy.generate(input_ids, **gen_args)
    
    def save_pretrained(self, save_directory, **kwargs):
        """
        Save both base policy and user projector.
        
        Args:
            save_directory: Directory to save model
            **kwargs: Additional arguments for saving
        """
        # Save base policy
        self.base_policy.save_pretrained(save_directory, **kwargs)
        
        # Save user projector separately
        import os
        projector_path = os.path.join(save_directory, "user_projector.pt")
        torch.save(self.user_projector.state_dict(), projector_path)
    
    def load_user_projector(self, load_directory):
        """
        Load user projector weights.
        
        Args:
            load_directory: Directory containing user_projector.pt
        """
        import os
        projector_path = os.path.join(load_directory, "user_projector.pt")
        if os.path.exists(projector_path):
            self.user_projector.load_state_dict(torch.load(projector_path))
    
    @property
    def module(self):
        """For compatibility with DDP."""
        return self
    
    def train(self, mode: bool = True):
        """Set training mode."""
        super().train(mode)
        self.base_policy.train(mode)
        self.user_projector.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode."""
        return self.train(False)

