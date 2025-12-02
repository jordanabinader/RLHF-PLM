"""
Training script for toxicity property head using ToxDL2 dataset.

This script trains a simplified toxicity classifier using:
- ESM-2 embeddings (1280-dim)
- Domain vectors (256-dim)
- Binary toxicity labels

The trained model predicts p_tox ∈ [0, 1] for any protein sequence.
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef
from tqdm import tqdm
import esm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from personalization.property_models import ToxicityHead

# Constants
AMINOACID = 'ACDEFGHIKLMNPQRSTVWY'


def pretrain_protein(seq: str, protein_model, batch_converter, device: str) -> torch.Tensor:
    """
    Extract ESM-2 embeddings for a protein sequence.
    
    Based on ToxDL2-main/src/dataset.py pretrain_protein function.
    """
    # Handle long sequences by chunking
    if len(seq) > 1022:
        seq_feats = []
        for i in range(len(seq) // 1022):
            chunk = seq[i * 1022:(i + 1) * 1022]
            data = [("protein", chunk)]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            with torch.no_grad():
                results = protein_model(
                    batch_tokens.to(device),
                    repr_layers=[33],
                    return_contacts=False
                )
            token_representations = results["representations"][33]
            feat = token_representations.squeeze(0)[1:len(chunk) + 1]
            seq_feats.append(feat)
        
        # Handle remaining sequence
        if len(seq) % 1022 > 0:
            chunk = seq[(len(seq) // 1022) * 1022:]
            data = [("protein", chunk)]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            with torch.no_grad():
                results = protein_model(
                    batch_tokens.to(device),
                    repr_layers=[33],
                    return_contacts=False
                )
            token_representations = results["representations"][33]
            feat = token_representations.squeeze(0)[1:len(chunk) + 1]
            seq_feats.append(feat)
        
        seq_feat = torch.cat(seq_feats, dim=0)
    else:
        data = [("protein", seq)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        with torch.no_grad():
            results = protein_model(
                batch_tokens.to(device),
                repr_layers=[33],
                return_contacts=False
            )
        token_representations = results["representations"][33]
        seq_feat = token_representations.squeeze(0)[1:len(seq) + 1]
    
    # Mean pooling over sequence length
    seq_feat = seq_feat.mean(dim=0)
    return seq_feat.cpu()


class ToxicityDataset(Dataset):
    """
    Dataset for toxicity prediction.
    
    Loads sequences, labels, and domain vectors from ToxDL2 format.
    """
    def __init__(
        self,
        fasta_path: Path,
        domain_path: Path,
        esm_model,
        batch_converter,
        device: str = "cuda",
        cache_dir: Path = None
    ):
        self.esm_model = esm_model
        self.batch_converter = batch_converter
        self.device = device
        
        # Load data
        self.sequences = []
        self.labels = []
        self.domain_vectors = []
        self.names = []
        
        print(f"Loading data from {fasta_path} and {domain_path}...")
        self._load_data(domain_path)
        
        print(f"Loaded {len(self)} samples")
    
    def _load_data(self, domain_path: Path):
        """Load sequences, labels, and domain vectors from .domain file."""
        with open(domain_path, 'r') as f:
            lines = f.readlines()
        
        current_name = None
        current_seq = None
        current_label = None
        current_domain = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('>'):
                # Save previous entry if complete
                if current_name and current_seq and current_label is not None and current_domain is not None:
                    self.names.append(current_name)
                    self.sequences.append(current_seq)
                    self.labels.append(current_label)
                    self.domain_vectors.append(current_domain)
                
                # Start new entry
                current_name = line[1:]
                current_seq = None
                current_label = None
                current_domain = None
            
            elif sum([char in AMINOACID for char in line]) == len(line) and len(line) > 0:
                # This is the sequence
                current_seq = line
            
            elif len(line) == 1 and line in ['0', '1']:
                # This is the label
                current_label = int(line)
            
            elif len(line.split(',')) in [256, 269]:
                # This is the domain vector
                current_domain = [float(x) for x in line.split(',')]
        
        # Save last entry
        if current_name and current_seq and current_label is not None and current_domain is not None:
            self.names.append(current_name)
            self.sequences.append(current_seq)
            self.labels.append(current_label)
            self.domain_vectors.append(current_domain)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            esm_embedding: (1280,) tensor
            domain_vector: (256,) tensor
            label: scalar tensor (0 or 1)
        """
        # Get ESM embedding
        esm_emb = pretrain_protein(
            self.sequences[idx],
            self.esm_model,
            self.batch_converter,
            self.device
        )
        
        # Get domain vector (pad or truncate to 256)
        domain_vec = self.domain_vectors[idx]
        if len(domain_vec) < 256:
            domain_vec = domain_vec + [0.0] * (256 - len(domain_vec))
        elif len(domain_vec) > 256:
            domain_vec = domain_vec[:256]
        domain_vec = torch.tensor(domain_vec, dtype=torch.float32)
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return esm_emb, domain_vec, label


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, dict]:
    """Evaluate model and return loss and metrics."""
    model.eval()
    
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for esm_emb, domain_vec, labels in tqdm(dataloader, desc="Evaluating"):
            esm_emb = esm_emb.to(device)
            domain_vec = domain_vec.to(device)
            labels = labels.to(device)
            
            outputs = model(esm_emb, domain_vec)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * len(labels)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    pred_binary = (all_preds >= 0.5).astype(int)
    metrics = {
        'accuracy': accuracy_score(all_labels, pred_binary),
        'f1': f1_score(all_labels, pred_binary),
        'auroc': roc_auc_score(all_labels, all_preds),
        'auprc': average_precision_score(all_labels, all_preds),
        'mcc': matthews_corrcoef(all_labels, pred_binary),
    }
    
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, metrics


def train_toxicity_model(
    data_dir: Path,
    output_dir: Path,
    batch_size: int = 32,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cuda",
    patience: int = 10,
):
    """Train toxicity prediction model."""
    
    print("=" * 80)
    print("Training Toxicity Property Head")
    print("=" * 80)
    
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
    train_dataset = ToxicityDataset(
        fasta_path=data_dir / "protein_sequences" / "train.fasta",
        domain_path=data_dir / "domain_data" / "train.domain",
        esm_model=protein_model,
        batch_converter=batch_converter,
        device=device,
    )
    
    val_dataset = ToxicityDataset(
        fasta_path=data_dir / "protein_sequences" / "valid.fasta",
        domain_path=data_dir / "domain_data" / "valid.domain",
        esm_model=protein_model,
        batch_converter=batch_converter,
        device=device,
    )
    
    test_dataset = ToxicityDataset(
        fasta_path=data_dir / "protein_sequences" / "test.fasta",
        domain_path=data_dir / "domain_data" / "test.domain",
        esm_model=protein_model,
        batch_converter=batch_converter,
        device=device,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize model
    print("\nInitializing model...")
    model = ToxicityHead(esm_dim=1280, domain_dim=256, hidden_dim=512)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    best_val_auroc = 0.0
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for esm_emb, domain_vec, labels in pbar:
            esm_emb = esm_emb.to(device)
            domain_vec = domain_vec.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(esm_emb, domain_vec)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(labels)
            pbar.set_postfix({'loss': loss.item()})
        
        train_loss /= len(train_dataset)
        
        # Validation
        val_loss, val_metrics = evaluate_model(model, val_loader, criterion, device)
        
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Metrics: Acc={val_metrics['accuracy']:.4f}, F1={val_metrics['f1']:.4f}, "
              f"AUROC={val_metrics['auroc']:.4f}, AUPRC={val_metrics['auprc']:.4f}, MCC={val_metrics['mcc']:.4f}")
        
        # Early stopping
        if val_metrics['auroc'] > best_val_auroc:
            best_val_auroc = val_metrics['auroc']
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
            torch.save(checkpoint, output_dir / "toxicity_head.pth")
            print(f"  → Saved best model (AUROC: {best_val_auroc:.4f})")
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
    checkpoint = torch.load(output_dir / "toxicity_head.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_metrics = evaluate_model(model, test_loader, criterion, device)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Metrics:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {test_metrics['f1']:.4f}")
    print(f"  AUROC: {test_metrics['auroc']:.4f}")
    print(f"  AUPRC: {test_metrics['auprc']:.4f}")
    print(f"  MCC: {test_metrics['mcc']:.4f}")
    
    print(f"\nModel saved to: {output_dir / 'toxicity_head.pth'}")


def main():
    parser = argparse.ArgumentParser(description="Train toxicity property head")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="ToxDL2-main/data",
        help="Path to ToxDL2 data directory"
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
    data_dir = base_dir / args.data_dir
    output_dir = base_dir / args.output_dir
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    train_toxicity_model(
        data_dir=data_dir,
        output_dir=output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()

