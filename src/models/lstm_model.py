# src/models/lstm_model.py

import torch
import pytorch_lightning as pl
from torch import nn

class LSTMModel(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        out, _ = self.lstm(x)
        # take the last timestep
        last = out[:, -1, :]
        return self.fc(last)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat.squeeze(), y.float())
        self.log("train_mse", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat.squeeze(), y.float())
        self.log("val_mse", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

if __name__ == "__main__":
    # Smoke test
    model = LSTMModel(input_size=16)
    dummy_x = torch.randn(4, 10, 16)  # batch=4, seq_len=10, features=16
    out = model(dummy_x)
    assert out.shape == (4, 1)
    print("LSTMModel smoke test OK, output shape:", out.shape)
