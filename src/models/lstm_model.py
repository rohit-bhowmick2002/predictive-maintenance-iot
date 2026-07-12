"""
PyTorch LSTM model with attention mechanism for failure classification.

Architecture:
  Input → Bidirectional LSTM (2 layers) → Multi-head Attention → FC → Sigmoid

Key design decisions:
  - Bidirectional for context from both past and future within sequence
  - Multi-head attention to focus on critical time steps
  - Residual connections for gradient flow
  - Weighted loss function for class imbalance
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from ..utils.config import config
from ..utils.logger import logger


class AttentionBlock(nn.Module):
    """Multi-head self-attention block."""

    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + self.dropout(attn_out))
        return x


class LSTMFailureClassifier(nn.Module):
    """
    Bidirectional LSTM with attention for sequence-based failure classification.

    Args:
        input_size: Number of input features (sensor channels)
        hidden_size: LSTM hidden state dimension
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        bidirectional: Use bidirectional LSTM
        attention_heads: Number of attention heads
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        attention_heads: int = 4,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Projection layer
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Attention
        lstm_output_size = hidden_size * self.num_directions
        self.attention = AttentionBlock(lstm_output_size, attention_heads, dropout)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, sequence_length, input_size)

        Returns:
            Logits: (batch_size, 1)
        """
        # Project input
        x = self.input_projection(x)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (B, T, hidden*dirs)

        # Attention over time dimension
        attn_out = self.attention(lstm_out)

        # Global average pooling over time (with attention)
        pooled = attn_out.mean(dim=1)  # (B, hidden*dirs)

        # Classify
        logits = self.classifier(pooled)
        return logits


class LSTMTrainer:
    """
    Training wrapper for the LSTM model with:
      - Weighted BCE loss for class imbalance
      - Cosine annealing with warm restarts
      - Gradient clipping
      - Early stopping
      - MLflow logging
    """

    def __init__(
        self,
        model: LSTMFailureClassifier,
        device: str = "cpu",
        pos_weight: float = 12.0,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 15,
    ):
        self.model = model.to(device)
        self.device = device
        self.pos_weight = pos_weight
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience

        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device)
        )

        self.best_loss = float("inf")
        self.best_epoch = 0
        self.patience_counter = 0
        self.train_losses: list = []
        self.val_losses: list = []

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        verbose: bool = True,
    ) -> Dict:
        """Train the LSTM model."""
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device).float().unsqueeze(1)

                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_loss += loss.item() * len(X_batch)

            train_loss /= len(train_loader.dataset)
            self.train_losses.append(train_loss)

            # Validation
            val_loss = self._evaluate_loss(val_loader)
            self.val_losses.append(val_loss)

            self.scheduler.step()

            if verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    f"  Epoch {epoch+1:3d}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )

            # Early stopping
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(f"  Early stopping at epoch {epoch+1}")
                    break

        logger.info(f"✅ LSTM trained. Best val loss: {self.best_loss:.4f} at epoch {self.best_epoch+1}")

        return {
            "best_epoch": self.best_epoch + 1,
            "best_val_loss": self.best_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }

    def _evaluate_loss(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device).float().unsqueeze(1)
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                total_loss += loss.item() * len(X_batch)
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def predict(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """Predict probabilities."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        predictions = []
        for (batch,) in loader:
            logits = self.model(batch)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            predictions.append(probs)

        return np.concatenate(predictions)

    @torch.no_grad()
    def predict_classes(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary classes."""
        probs = self.predict(X)
        return (probs >= threshold).astype(int)

    def save(self, path: Optional[Path] = None) -> None:
        path = path or config.data.models_dir / "lstm_model.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "input_size": self.model.input_size,
            "hidden_size": self.model.hidden_size,
            "num_layers": self.model.num_layers,
            "bidirectional": self.model.bidirectional,
            "best_loss": self.best_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "LSTMTrainer":
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = LSTMFailureClassifier(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            num_layers=checkpoint["num_layers"],
            bidirectional=checkpoint["bidirectional"],
        )
        trainer = cls(model, device=device)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.best_loss = checkpoint["best_loss"]
        trainer.train_losses = checkpoint["train_losses"]
        trainer.val_losses = checkpoint["val_losses"]
        return trainer


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 256,
    val_split: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for LSTM training."""
    # Split train into train/val
    n_val = int(len(X_train) * val_split)
    X_tr, X_val = X_train[:-n_val], X_train[-n_val:]
    y_tr, y_val = y_train[:-n_val], y_train[-n_val:]

    train_dataset = TensorDataset(
        torch.FloatTensor(X_tr),
        torch.FloatTensor(y_tr),
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val),
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
