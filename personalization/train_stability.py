"""
Training script for stability property head using EsmTherm approach.

This script provides two options:
1. Load pre-trained EsmTherm checkpoint (if available)
2. Train a new stability regressor on mega-scale stability data

The trained model predicts p_stab: continuous stability score (e.g., ΔΔG).
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

# Try importing EsmTherm components if available
try:
    from transformers import EsmTokenizer
    from transformers.utils import logging as transformers_logging
    transformers_logging.set_verbosity_error()
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Using fallback ESM loading.")

try:
    import esm
    ESM_AVAILABLE = True
except ImportError:
    ESM_AVAILABLE = False
    print("Error: fair-esm not available. Please install: pip install fair-esm")

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from personalization.property_models import StabilityHead


def load_esmtherm_checkpoint(checkpoint_path: Path, device: str = "cuda") -> Optional[nn.Module]:
    """
    Load pre-trained EsmTherm model if available.
    
    Args:
        checkpoint_path: Path to EsmTherm checkpoint directory
        device: Device to load model on
    
    Returns:
        Loaded model or None if not available
    """
    if not checkpoint_path.exists():
        print(f"EsmTherm checkpoint not found at {checkpoint_path}")
        return None
    
    if not TRANSFORMERS_AVAILABLE:
        print("transformers library not available. Cannot load EsmTherm checkpoint.")
        return None
    
    try:
        from transformers import EsmForSequenceClassification
        
        print(f"Loading EsmTherm checkpoint from {checkpoint_path}...")
        model = EsmForSequenceClassification.from_pretrained(checkpoint_path)
        model = model.to(device)
        model.eval()
        
        print("Successfully loaded EsmTherm checkpoint")
        return model
    
    except Exception as e:
        print(f"Failed to load EsmTherm checkpoint: {e}")
        return None


def encode_sequences_with_esm(
    sequences: list,
    esm_model,
    batch_converter,
    device: str
) -> torch.Tensor:
    """
    Encode sequences using ESM-2 model.
    
    Returns:
        Tensor of shape (len(sequences), esm_dim) with mean-pooled embeddings
    """
    data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    
    with torch.no_grad():
        results = esm_model(
            batch_tokens,
            repr_layers=[33],
            return_contacts=False
        )
    
    # Extract token representations
    token_representations = results["representations"][33]
    
    # Mean pooling (excluding BOS and EOS tokens)
    embeddings = []
    for i, seq in enumerate(sequences):
        # Tokens: [BOS, seq..., EOS, padding...]
        seq_tokens = token_representations[i, 1:len(seq)+1, :]
        embeddings.append(seq_tokens.mean(dim=0))
    
    return torch.stack(embeddings)


class SimpleStabilityDataset(Dataset):
    """
    Simple dataset for stability prediction.
    Assumes data is in CSV format with 'sequence' and 'stability' columns.
    """
    def __init__(
        self,
        csv_path: Path,
        esm_model,
        batch_converter,
        device: str,
        max_length: int = 1000,
    ):
        import pandas as pd
        
        self.esm_model = esm_model
        self.batch_converter = batch_converter
        self.device = device
        self.max_length = max_length
        
        # Load data
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Filter out sequences that are too long
        df = df[df['sequence'].str.len() <= max_length]
        
        self.sequences = df['sequence'].tolist()
        self.labels = df['stability'].values.astype(np.float32)
        
        print(f"Loaded {len(self)} samples")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            esm_embedding: (1280,) tensor
            label: scalar tensor (stability score)
        """
        # Get ESM embedding
        esm_emb = encode_sequences_with_esm(
            [self.sequences[idx]],
            self.esm_model,
            self.batch_converter,
            self.device
        )[0]
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return esm_emb.cpu(), label


def evaluate_stability_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, dict]:
    """Evaluate stability model and return loss and metrics."""
    model.eval()
    
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for esm_emb, labels in tqdm(dataloader, desc="Evaluating"):
            esm_emb = esm_emb.to(device)
            labels = labels.to(device)
            
            outputs = model(esm_emb)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * len(labels)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    spearman_corr, spearman_p = spearmanr(all_labels, all_preds)
    pearson_corr, pearson_p = pearsonr(all_labels, all_preds)
    
    metrics = {
        'spearman': spearman_corr,
        'pearson': pearson_corr,
        'mse': np.mean((all_preds - all_labels) ** 2),
        'mae': np.mean(np.abs(all_preds - all_labels)),
    }
    
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, metrics


def train_stability_model(
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    output_dir: Path,
    batch_size: int = 32,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cuda",
    patience: int = 10,
):
    """Train stability prediction model."""
    
    print("=" * 80)
    print("Training Stability Property Head")
    print("=" * 80)
    
    if not ESM_AVAILABLE:
        print("Error: fair-esm not installed. Cannot train model.")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load ESM-2 model
    print("\nLoading ESM-2 model...")
    protein_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    protein_model = protein_model.to(device)
    protein_model.eval()
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = SimpleStabilityDataset(
        csv_path=train_csv,
        esm_model=protein_model,
        batch_converter=batch_converter,
        device=device,
    )
    
    val_dataset = SimpleStabilityDataset(
        csv_path=val_csv,
        esm_model=protein_model,
        batch_converter=batch_converter,
        device=device,
    )
    
    test_dataset = SimpleStabilityDataset(
        csv_path=test_csv,
        esm_model=protein_model,
        batch_converter=batch_converter,
        device=device,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize model
    print("\nInitializing model...")
    model = StabilityHead(esm_dim=1280, hidden_dim=512, n_layers=2)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    best_val_spearman = 0.0
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for esm_emb, labels in pbar:
            esm_emb = esm_emb.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(esm_emb)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(labels)
            pbar.set_postfix({'loss': loss.item()})
        
        train_loss /= len(train_dataset)
        
        # Validation
        val_loss, val_metrics = evaluate_stability_model(model, val_loader, criterion, device)
        
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Metrics: Spearman={val_metrics['spearman']:.4f}, "
              f"Pearson={val_metrics['pearson']:.4f}, MAE={val_metrics['mae']:.4f}")
        
        # Early stopping based on Spearman correlation
        if val_metrics['spearman'] > best_val_spearman:
            best_val_spearman = val_metrics['spearman']
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
            }
            torch.save(checkpoint, output_dir / "stability_head.pth")
            print(f"  → Saved best model (Spearman: {best_val_spearman:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping after {epoch + 1} epochs")
                break
    
    # Test evaluation
    print("\n" + "=" * 80)
    print("Final Test Evaluation")
    print("=" * 80)
    
    # Load best model
    checkpoint = torch.load(output_dir / "stability_head.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_metrics = evaluate_stability_model(model, test_loader, criterion, device)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Metrics:")
    print(f"  Spearman Correlation: {test_metrics['spearman']:.4f}")
    print(f"  Pearson Correlation: {test_metrics['pearson']:.4f}")
    print(f"  MSE: {test_metrics['mse']:.4f}")
    print(f"  MAE: {test_metrics['mae']:.4f}")
    
    print(f"\nModel saved to: {output_dir / 'stability_head.pth'}")


def create_simple_stability_head(output_dir: Path, device: str = "cuda"):
    """
    Create a simple untrained stability head as a placeholder.
    This can be used when no training data is available.
    """
    print("Creating placeholder stability head (untrained)...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = StabilityHead(esm_dim=1280, hidden_dim=512, n_layers=2)
    
    checkpoint = {
        'epoch': 0,
        'model_state_dict': model.state_dict(),
        'note': 'Placeholder model - not trained on real data',
    }
    
    torch.save(checkpoint, output_dir / "stability_head.pth")
    print(f"Placeholder model saved to: {output_dir / 'stability_head.pth'}")
    print("WARNING: This is an untrained model. For real use, train on stability data.")


def main():
    parser = argparse.ArgumentParser(description="Train or load stability property head")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "load_esmtherm", "placeholder"],
        default="placeholder",
        help="Mode: train new model, load EsmTherm checkpoint, or create placeholder"
    )
    parser.add_argument(
        "--esmtherm-checkpoint",
        type=str,
        default="EsmTherm-main/output_dir/checkpoint-best",
        help="Path to EsmTherm checkpoint directory"
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        help="Path to training CSV (sequence, stability columns)"
    )
    parser.add_argument(
        "--val-csv",
        type=str,
        help="Path to validation CSV"
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        help="Path to test CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="personalization/checkpoints",
        help="Output directory for trained model"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    
    args = parser.parse_args()
    
    # Convert paths to absolute
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / args.output_dir
    
    if args.mode == "load_esmtherm":
        esmtherm_path = base_dir / args.esmtherm_checkpoint
        model = load_esmtherm_checkpoint(esmtherm_path, args.device)
        if model is None:
            print("Failed to load EsmTherm checkpoint. Creating placeholder instead.")
            create_simple_stability_head(output_dir, args.device)
    
    elif args.mode == "train":
        if not args.train_csv or not args.val_csv or not args.test_csv:
            print("Error: --train-csv, --val-csv, and --test-csv required for training mode")
            sys.exit(1)
        
        train_csv = Path(args.train_csv)
        val_csv = Path(args.val_csv)
        test_csv = Path(args.test_csv)
        
        if not train_csv.exists():
            print(f"Error: Training CSV not found: {train_csv}")
            sys.exit(1)
        
        train_stability_model(
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            output_dir=output_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            device=args.device,
            patience=args.patience,
        )
    
    else:  # placeholder mode
        create_simple_stability_head(output_dir, args.device)


if __name__ == "__main__":
    main()

