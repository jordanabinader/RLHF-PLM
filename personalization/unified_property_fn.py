"""
Unified Property Function: g(x) = [p_act, p_tox, p_stab, p_len]

This module provides the single source of truth for computing biological
properties for any protein sequence. All property predictions go through
this unified function to ensure consistency.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import esm


def encode_sequences(
    sequences: List[str],
    esm_model: nn.Module,
    batch_converter,
    alphabet,
    device: str = "cuda",
    representation_layer: int = 33,
) -> torch.Tensor:
    """
    Encode sequences with ESM model and return per-sequence embeddings.
    
    Based on amp_design/reward.py encode() function.
    
    Args:
        sequences: List of protein sequences
        esm_model: ESM-2 model
        batch_converter: ESM batch converter
        alphabet: ESM alphabet
        device: Device to run on
        representation_layer: Which layer to extract (33 for ESM2-650M)
    
    Returns:
        Tensor of shape (batch_size, esm_dim) with mean-pooled embeddings
    """
    device = torch.device(device)
    esm_model = esm_model.to(device).eval()
    
    # Prepare data for batch conversion
    data = [(f"protein_{idx}", seq) for idx, seq in enumerate(sequences)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device, non_blocking=True)
    
    # Create mask for actual sequence tokens (excluding BOS, EOS, padding)
    lengths = [len(seq) for seq in sequences]
    batch_size = len(sequences)
    max_length = max(lengths) if lengths else 0
    mask = torch.zeros((batch_size, max_length), dtype=torch.bool, device=device)
    for idx, length in enumerate(lengths):
        if length:
            mask[idx, :length] = 1
    
    # Get ESM representations
    with torch.no_grad():
        output = esm_model(batch_tokens, repr_layers=[representation_layer], return_contacts=False)
        token_reps = output["representations"][representation_layer]
        # Remove BOS and EOS tokens
        token_reps = token_reps[:, 1:-1, :]
    
    # Mean pooling over valid tokens
    mask = mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
    masked_embedding = token_reps * mask
    seq_lengths = mask.sum(dim=1).clamp_min(1)
    pooled_embeddings = masked_embedding.sum(dim=1) / seq_lengths
    
    return pooled_embeddings


class UnifiedPropertyFunction:
    """
    Unified property function that computes g(x) = [p_act, p_tox, p_stab, p_len].
    
    This is the single point of truth for all property predictions. It:
    1. Computes ESM embeddings once for all properties
    2. Runs each property head (activity, toxicity, stability)
    3. Computes normalized length
    4. Returns a standardized property vector
    
    Usage:
        property_fn = UnifiedPropertyFunction(...)
        properties = property_fn(["ACDEFGHIKLM", "MKLLPGK"])
        # properties shape: (2, 4) with [p_act, p_tox, p_stab, p_len]
    """
    
    def __init__(
        self,
        esm_model: nn.Module,
        batch_converter,
        alphabet,
        activity_head: nn.Module,
        toxicity_head: nn.Module,
        stability_head: nn.Module,
        domain_encoder: Optional[nn.Module] = None,
        device: str = "cuda",
        max_length: int = 100,
        representation_layer: int = 33,
    ):
        """
        Initialize unified property function.
        
        Args:
            esm_model: ESM-2 protein language model
            batch_converter: ESM batch converter
            alphabet: ESM alphabet
            activity_head: Neural network for predicting activity
            toxicity_head: Neural network for predicting toxicity
            stability_head: Neural network for predicting stability
            domain_encoder: Optional encoder for domain vectors (for toxicity)
            device: Device to run computations on
            max_length: Maximum sequence length for normalization
            representation_layer: Which ESM layer to use (33 for ESM2-650M)
        """
        self.esm_model = esm_model.to(device).eval()
        self.batch_converter = batch_converter
        self.alphabet = alphabet
        self.activity_head = activity_head.to(device).eval()
        self.toxicity_head = toxicity_head.to(device).eval()
        self.stability_head = stability_head.to(device).eval()
        self.domain_encoder = domain_encoder
        self.device = device
        self.max_length = max_length
        self.representation_layer = representation_layer
        
        # Freeze all models - they are fixed property predictors
        for param in self.esm_model.parameters():
            param.requires_grad = False
        for param in self.activity_head.parameters():
            param.requires_grad = False
        for param in self.toxicity_head.parameters():
            param.requires_grad = False
        for param in self.stability_head.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def __call__(self, sequences: List[str]) -> torch.Tensor:
        """
        Compute property vector g(x) for a batch of sequences.
        
        Args:
            sequences: List of protein sequences (strings)
        
        Returns:
            properties: Tensor of shape (batch_size, 4) with columns:
                        [p_act, p_tox, p_stab, p_len]
        """
        if len(sequences) == 0:
            return torch.empty((0, 4), device=self.device)
        
        # 1. Compute ESM embeddings e(x) - shared across all properties
        esm_embeddings = encode_sequences(
            sequences=sequences,
            esm_model=self.esm_model,
            batch_converter=self.batch_converter,
            alphabet=self.alphabet,
            device=self.device,
            representation_layer=self.representation_layer,
        )
        
        # 2. Compute activity scores p_act
        # Note: ActivityHead uses layer 6 encoding internally
        p_act = self.activity_head(sequences, device=self.device)
        if p_act.dim() == 0:
            p_act = p_act.unsqueeze(0)
        
        # 3. Compute stability scores p_stab
        # Note: StabilityHead uses EsmTherm model internally (sequences → stability)
        p_stab = self.stability_head(sequences, device=self.device)
        if p_stab.dim() == 0:
            p_stab = p_stab.unsqueeze(0)
        
        # 4. Compute toxicity scores p_tox
        # For toxicity, we need domain vectors. If domain_encoder is available, use it.
        # Otherwise, use zero vectors as a fallback.
        if self.domain_encoder is not None:
            domain_vectors = self._get_domain_vectors(sequences)
        else:
            # Fallback: zero domain vectors (256-dim for ToxDL2 compatibility)
            domain_vectors = torch.zeros(
                (len(sequences), 256),
                device=self.device,
                dtype=torch.float32
            )
        
        p_tox = self.toxicity_head(esm_embeddings, domain_vectors)
        if p_tox.dim() == 0:
            p_tox = p_tox.unsqueeze(0)
        
        # 5. Compute normalized length p_len
        lengths = torch.tensor([len(seq) for seq in sequences], device=self.device, dtype=torch.float32)
        p_len = lengths / self.max_length
        
        # 6. Stack into property vector [p_act, p_tox, p_stab, p_len]
        properties = torch.stack([p_act, p_tox, p_stab, p_len], dim=1)
        
        return properties
    
    def _get_domain_vectors(self, sequences: List[str]) -> torch.Tensor:
        """
        Get domain vectors for sequences.
        
        If domain_encoder is provided, encode sequences to domain vectors.
        Otherwise, return zero vectors.
        
        Args:
            sequences: List of protein sequences
        
        Returns:
            Tensor of shape (batch_size, domain_dim)
        """
        if self.domain_encoder is None:
            return torch.zeros((len(sequences), 256), device=self.device, dtype=torch.float32)
        
        # If domain encoder is provided, use it
        with torch.no_grad():
            domain_vectors = self.domain_encoder(sequences)
        
        return domain_vectors
    
    def compute_single_property(
        self,
        sequences: List[str],
        property_name: str
    ) -> torch.Tensor:
        """
        Compute a single property for debugging or analysis.
        
        Args:
            sequences: List of protein sequences
            property_name: One of ['activity', 'toxicity', 'stability', 'length']
        
        Returns:
            Tensor of shape (batch_size,) with the requested property
        """
        properties = self(sequences)
        
        property_map = {
            'activity': 0,
            'toxicity': 1,
            'stability': 2,
            'length': 3,
        }
        
        if property_name not in property_map:
            raise ValueError(
                f"Unknown property: {property_name}. "
                f"Must be one of {list(property_map.keys())}"
            )
        
        idx = property_map[property_name]
        return properties[:, idx]
    
    def get_property_names(self) -> List[str]:
        """Return names of properties in order."""
        return ['activity', 'toxicity', 'stability', 'length']
    
    def get_property_dict(self, sequences: List[str]) -> dict:
        """
        Compute properties and return as a dictionary.
        
        Args:
            sequences: List of protein sequences
        
        Returns:
            Dictionary mapping property names to tensors
        """
        properties = self(sequences)
        
        return {
            'activity': properties[:, 0],
            'toxicity': properties[:, 1],
            'stability': properties[:, 2],
            'length': properties[:, 3],
            'full_vector': properties,
        }


def create_unified_property_function(
    activity_checkpoint: str,
    toxicity_checkpoint: str,
    stability_checkpoint: str,
    esm_model_size: str = "650M",
    device: str = "cuda",
    max_length: int = 100,
) -> UnifiedPropertyFunction:
    """
    Convenience function to create a UnifiedPropertyFunction.
    
    Args:
        activity_checkpoint: Path to activity head checkpoint
        toxicity_checkpoint: Path to toxicity head checkpoint
        stability_checkpoint: Path to stability head checkpoint
        esm_model_size: ESM model size ("650M" or "8M")
        device: Device to run on
        max_length: Maximum sequence length for normalization
    
    Returns:
        Initialized UnifiedPropertyFunction
    """
    from personalization.property_models import (
        load_activity_head,
        load_toxicity_head,
        load_stability_head,
    )
    
    # Load ESM model
    print(f"Loading ESM-2 {esm_model_size} model...")
    if esm_model_size == "650M":
        esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        representation_layer = 33
    elif esm_model_size == "8M":
        esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        representation_layer = 6
    else:
        raise ValueError(f"Unsupported ESM model size: {esm_model_size}")
    
    batch_converter = alphabet.get_batch_converter()
    esm_model = esm_model.to(device)
    esm_model.eval()
    
    # Load property heads
    print("Loading property heads...")
    activity_head = load_activity_head(activity_checkpoint, device)
    toxicity_head = load_toxicity_head(toxicity_checkpoint, device)
    stability_head = load_stability_head(stability_checkpoint, device)
    
    # Create unified function
    print("Creating unified property function...")
    property_fn = UnifiedPropertyFunction(
        esm_model=esm_model,
        batch_converter=batch_converter,
        alphabet=alphabet,
        activity_head=activity_head,
        toxicity_head=toxicity_head,
        stability_head=stability_head,
        domain_encoder=None,  # TODO: Add domain encoder if needed
        device=device,
        max_length=max_length,
        representation_layer=representation_layer,
    )
    
    print("✓ Unified property function ready")
    return property_fn

