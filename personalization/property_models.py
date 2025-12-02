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


class AMPClassifier(nn.Module):
    """MLP architecture matching best_new_4.pth checkpoint."""
    def __init__(self, input_dim: int = 320, hidden_dim: int = 128, output_dim: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class ActivityHead(nn.Module):
    """
    Wrapper for existing AMP activity classifier (best_new_4.pth).
    
    Predicts p_act ∈ [0, 1]: probability of antimicrobial activity.
    
    NOTE: This classifier was trained on ESM2-8M (layer 6, 320-dim embeddings),
    NOT ESM2-650M. It uses a separate ESM model for encoding.
    """
    def __init__(self, checkpoint_path: str):
        super().__init__()
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Activity classifier not found at {checkpoint_path}")
        
        # Load ESM2-8M model (320-dim hidden size)
        import esm
        self.esm_model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        
        # Create the MLP architecture (must match checkpoint!)
        self.classifier = AMPClassifier(input_dim=320, hidden_dim=128, output_dim=1)
        
        # Load the state_dict
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self.classifier.load_state_dict(state_dict)
        self.classifier.eval()
        
        # Freeze parameters - this is a fixed property predictor
        for param in self.esm_model.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = False
    
    def encode_8M_layer6(self, sequences, device):
        """Encode sequences using ESM2-8M layer 6 (320-dim)."""
        from amp_design.reward import encode
        return encode(
            sequences,
            self.esm_model,
            self.batch_converter,
            self.alphabet,
            device=device,
            representation_layer=6,
        )
    
    def forward(self, sequences, device="cuda"):
        """
        Args:
            sequences: List of protein sequences (strings)
            device: Device to run on
        
        Returns:
            p_act: (batch_size,) tensor with activity probabilities
        """
        with torch.no_grad():
            # Move ESM model to device
            self.esm_model = self.esm_model.to(device)
            # Encode with ESM2-8M layer 6 (320-dim)
            embeddings = self.encode_8M_layer6(sequences, device)
            output = self.classifier(embeddings.to(device))
            if output.dim() > 1:
                output = output.squeeze(-1)
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
    Wrapper for EsmTherm stability prediction model.
    
    Predicts p_stab: continuous stability score (ΔΔG values).
    
    NOTE: This is a full end-to-end model (sequences → stability scores).
    It loads the pre-trained EsmTherm model which includes:
    - ESM-2 encoder
    - Classification head for stability prediction
    """
    def __init__(self, checkpoint_path: str = None, use_esmtherm: bool = True):
        super().__init__()
        self.use_esmtherm = use_esmtherm
        
        if use_esmtherm and checkpoint_path:
            # Load full EsmTherm model
            try:
                from transformers import EsmTokenizer, EsmForSequenceClassification
                
                # Check if checkpoint is our wrapped format or direct HF model
                checkpoint_path = Path(checkpoint_path)
                if checkpoint_path.suffix == '.pth':
                    # Our wrapped format
                    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                    if ckpt.get('model_type') == 'esmtherm_full':
                        # Load from the original source path
                        source_path = ckpt.get('source')
                        if source_path is None:
                            # Fallback: try to find EsmTherm checkpoint
                            possible_paths = [
                                checkpoint_path.parent.parent / 'EsmTherm-main/output_dir/checkpoint-best',
                                Path('EsmTherm-main/output_dir/checkpoint-best'),
                            ]
                            for p in possible_paths:
                                if p.exists():
                                    source_path = str(p)
                                    break
                        
                        if source_path and Path(source_path).exists():
                            self.model = EsmForSequenceClassification.from_pretrained(source_path)
                            # Load tokenizer from base ESM model (checkpoint doesn't include tokenizer files)
                            config_path = Path(source_path) / 'config.json'
                            if config_path.exists():
                                import json
                                with open(config_path) as f:
                                    config = json.load(f)
                                base_model = config.get('_name_or_path', 'facebook/esm2_t12_35M_UR50D')
                                self.tokenizer = EsmTokenizer.from_pretrained(base_model)
                            else:
                                # Fallback to default ESM-2 tokenizer
                                self.tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t12_35M_UR50D')
                        else:
                            print(f"Warning: EsmTherm source path not found: {source_path}")
                            self.use_esmtherm = False
                            self._init_placeholder()
                    else:
                        # Fallback to placeholder
                        print("Warning: Checkpoint not EsmTherm format, using placeholder")
                        self.use_esmtherm = False
                        self._init_placeholder()
                else:
                    # Direct HF checkpoint directory
                    self.model = EsmForSequenceClassification.from_pretrained(str(checkpoint_path))
                    self.tokenizer = EsmTokenizer.from_pretrained(str(checkpoint_path))
                
                if self.use_esmtherm:
                    self.model.eval()
                    # Freeze parameters
                    for param in self.model.parameters():
                        param.requires_grad = False
                    
            except Exception as e:
                print(f"Warning: Failed to load EsmTherm model: {e}")
                print("Using placeholder stability head instead")
                self.use_esmtherm = False
                self._init_placeholder()
        else:
            # Placeholder MLP
            self._init_placeholder()
    
    def _init_placeholder(self):
        """Initialize simple MLP placeholder."""
        # Simple 2-layer MLP for placeholder
        self.mlp = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.model = None
        self.tokenizer = None
    
    def forward(self, sequences, device="cuda"):
        """
        Args:
            sequences: List of protein sequences (strings) OR pre-computed embeddings
            device: Device to run on
        
        Returns:
            p_stab: (batch_size,) tensor with stability scores
        """
        if self.use_esmtherm and self.model is not None:
            # Use full EsmTherm model
            with torch.no_grad():
                # Move model to device
                self.model = self.model.to(device)
                
                # If sequences is a tensor, it's pre-computed embeddings (backward compat)
                if isinstance(sequences, torch.Tensor):
                    # Use placeholder MLP on embeddings
                    if hasattr(self, 'mlp'):
                        return self.mlp(sequences.to(device)).squeeze(-1)
                    else:
                        # No MLP, return zeros
                        return torch.zeros(sequences.shape[0], device=device)
                
                # Tokenize sequences
                inputs = self.tokenizer(
                    sequences,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Get predictions
                outputs = self.model(**inputs)
                logits = outputs.logits.squeeze(-1)
                
                return logits
        else:
            # Placeholder: use simple MLP or return zeros
            if isinstance(sequences, torch.Tensor):
                # Pre-computed embeddings
                if hasattr(self, 'mlp'):
                    return self.mlp(sequences.to(device)).squeeze(-1)
                else:
                    return torch.zeros(sequences.shape[0], device=device)
            else:
                # Sequences but no model - return zeros
                return torch.zeros(len(sequences), device=device)


def load_activity_head(checkpoint_path: str, device: str = "cuda") -> ActivityHead:
    """Load pre-trained activity head (loads its own ESM2-8M model internally)."""
    model = ActivityHead(checkpoint_path)
    model = model.to(device)
    model.eval()
    return model


def load_toxicity_head(checkpoint_path: str, device: str = "cuda") -> ToxicityHead:
    """Load trained toxicity head."""
    model = ToxicityHead()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Handle both direct state dict and wrapped checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    return model


def load_stability_head(checkpoint_path: str, device: str = "cuda") -> StabilityHead:
    """
    Load trained stability head.
    
    Handles both EsmTherm checkpoints and placeholder models.
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Check if it's an EsmTherm checkpoint
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if ckpt.get('model_type') == 'esmtherm_full':
            # Load EsmTherm model
            model = StabilityHead(checkpoint_path=str(checkpoint_path), use_esmtherm=True)
            model = model.to(device)
            model.eval()
            return model
    
    # Fallback: load placeholder model
    print("Warning: Loading placeholder stability head (not EsmTherm)")
    model = StabilityHead(use_esmtherm=False)
    
    # Try to load state dict if available
    if checkpoint_path.exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if 'model_state_dict' in checkpoint and hasattr(model, 'mlp'):
                model.mlp.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"Warning: Could not load checkpoint state: {e}")
    
    model = model.to(device)
    model.eval()
    return model

