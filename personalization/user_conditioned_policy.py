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
    Wraps a base policy to inject user context into generation.
    
    User embeddings are prepended to input token embeddings, allowing
    the model to condition on user preferences throughout generation.
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
    
    def _get_embedding_layer(self):
        """Get the base policy's token embedding layer."""
        if hasattr(self.base_policy, 'transformer'):
            return self.base_policy.transformer.wte
        elif hasattr(self.base_policy, 'model'):
            return self.base_policy.model.get_input_embeddings()
        else:
            return self.base_policy.get_input_embeddings()
    
    def forward(self, input_ids, user_context: Optional[torch.Tensor] = None, **kwargs):
        """
        Forward pass with user context injected into embeddings.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            user_context: User weights (batch_size, 4) or (4,) - REQUIRED
            **kwargs: Additional arguments for base policy
        
        Returns:
            Model outputs from base policy
        """
        if user_context is None:
            raise ValueError(
                "user_context is required for UserConditionedPolicyWrapper. "
                "Pass persona weights as user_context parameter."
            )
        
        # Project user context: (batch_size, 4) -> (batch_size, projection_dim)
        if user_context.dim() == 1:
            batch_size = input_ids.shape[0]
            user_context = user_context.unsqueeze(0).repeat(batch_size, 1)
        user_embed = self.user_projector(user_context)
        
        # Get token embeddings from base policy
        embed_layer = self._get_embedding_layer()
        token_embeds = embed_layer(input_ids)
        
        # Prepend user embedding as first "token"
        user_token = user_embed.unsqueeze(1)  # (batch, 1, projection_dim)
        combined = torch.cat([user_token, token_embeds], dim=1)
        
        # Adjust attention mask to account for prepended user token
        if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
            mask = kwargs['attention_mask']
            user_mask = torch.ones((mask.shape[0], 1), device=mask.device, dtype=mask.dtype)
            kwargs['attention_mask'] = torch.cat([user_mask, mask], dim=1)
        
        return self.base_policy(inputs_embeds=combined, **kwargs)
    
    def generate(
        self,
        input_ids,
        user_context: torch.Tensor,
        max_length: int = None,
        max_new_tokens: int = None,
        **generation_kwargs
    ):
        """
        Generate sequences conditioned on user context with length control.
        
        Args:
            input_ids: Starting tokens (batch_size, seq_len)
            user_context: User weights (batch_size, 4) or (4,) - REQUIRED
            max_length: Maximum total length (deprecated, use max_new_tokens)
            max_new_tokens: Maximum number of new tokens to generate
            **generation_kwargs: Additional args for base_policy.generate()
        
        Returns:
            Generated token IDs
        """
        from personalization.length_logit_bias import LengthLogitBias
        
        # Project user context
        if user_context.dim() == 1:
            batch_size = input_ids.shape[0]
            user_context = user_context.unsqueeze(0).repeat(batch_size, 1)
        user_embed = self.user_projector(user_context)
        
        # Get token embeddings and prepend user embedding
        embed_layer = self._get_embedding_layer()
        token_embeds = embed_layer(input_ids)
        user_token = user_embed.unsqueeze(1)  # (batch, 1, projection_dim)
        combined = torch.cat([user_token, token_embeds], dim=1)
        
        # Adjust attention mask to account for prepended user token
        if 'attention_mask' in generation_kwargs:
            mask = generation_kwargs['attention_mask']
            user_mask = torch.ones((mask.shape[0], 1), device=mask.device, dtype=mask.dtype)
            generation_kwargs['attention_mask'] = torch.cat([user_mask, mask], dim=1)
        
        # Add length biasing based on w_len (4th component of user weights)
        # Formula: target_length = max_length * (1 - w_len)
        # Negative w_len → longer sequences, positive w_len → shorter sequences
        gen_args = generation_kwargs.copy()
        if max_new_tokens is not None:
            base_length = max_new_tokens
            gen_args['max_new_tokens'] = max_new_tokens
        elif max_length is not None:
            base_length = max_length
            gen_args['max_length'] = max_length
        else:
            base_length = 50  # Default
            gen_args['max_new_tokens'] = 50
        
        # Get tokenizer for EOS token (try multiple sources)
        tokenizer = None
        if hasattr(self, 'tok'):
            tokenizer = self.tok
        elif hasattr(self.base_policy, 'config') and hasattr(self.base_policy.config, 'eos_token_id'):
            # Create minimal tokenizer-like object
            class MinimalTokenizer:
                def __init__(self, eos_id):
                    self.eos_token_id = eos_id
            tokenizer = MinimalTokenizer(self.base_policy.config.eos_token_id)
        
        if tokenizer is not None:
            w_len = user_context[0, 3].item()  # Length weight
            target_length = int(base_length * (1 - w_len))
            target_length = max(10, target_length)  # Ensure minimum length
            
            length_bias = LengthLogitBias(
                eos_token_id=tokenizer.eos_token_id,
                target_length=target_length,
                bias_strength=5.0
            )
            
            if 'logits_processor' in gen_args:
                gen_args['logits_processor'].append(length_bias)
            else:
                gen_args['logits_processor'] = [length_bias]
        
        # Generate with combined embeddings
        outputs = self.base_policy.generate(inputs_embeds=combined, **gen_args)
        
        return outputs
    
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

