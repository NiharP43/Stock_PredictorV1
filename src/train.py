# src/train.py

"""
Entrypoint script to train various time-series models on processed stock data.
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path so `src` package can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl

from src.config import PROCESSED_DATA_DIR, DEFAULT_TICKER


class TimeSeriesDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, seq_len, horizon):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.seq_len = seq_len
        self.horizon = horizon
        data = df.reset_index(drop=True)
        self.features = data[feature_cols].values.astype(float)
        self.targets = data[target_col].values.astype(float)

    def __len__(self):
        return len(self.features) - self.seq_len - self.horizon + 1

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len + self.horizon - 1]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def get_dataloaders(df, feature_cols, target_col, seq_len, horizon, batch_size, val_split=0.2):
    dataset = TimeSeriesDataset(df, feature_cols, target_col, seq_len, horizon)
    total = len(dataset)
    n_val = int(total * val_split)
    n_train = total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["lstm", "transformer"],
                        help="Which model to train: lstm or transformer")
    parser.add_argument("--ticker", type=str, default=DEFAULT_TICKER)
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gpus", type=int, default=0)
    args = parser.parse_args()

    # Load processed data
    data_path = PROCESSED_DATA_DIR / f"{args.ticker}_features.csv"
    df = pd.read_csv(data_path, parse_dates=["Date"]).sort_values("Date")

    # Create prediction target (next-period return)
    df["target"] = df["return"].shift(-args.horizon)
    df = df.dropna().reset_index(drop=True)

    # Define features and target
    feature_cols = [c for c in df.columns if c not in ["Date", "target"]]
    target_col = "target"

    train_loader, val_loader = get_dataloaders(
        df, feature_cols, target_col,
        seq_len=args.seq_len,
        horizon=args.horizon,
        batch_size=args.batch_size,
    )

    # Dynamically import model
    if args.model == "lstm":
        from src.models.lstm_model import LSTMModel as ModelClass
        model = ModelClass(input_size=len(feature_cols), lr=args.lr)
    elif args.model == "transformer":
        from src.models.transformer_model import TransformerModel as ModelClass
        model = ModelClass(input_size=len(feature_cols), lr=args.lr)
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    # Trainer setup with updated GPU handling for PyTorch Lightning
    trainer_kwargs = {
        "max_epochs": args.epochs,
        "deterministic": True
    }
    if args.gpus and args.gpus > 0:
        trainer_kwargs.update({"accelerator": "gpu", "devices": args.gpus})

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
