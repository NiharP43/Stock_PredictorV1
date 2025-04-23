# src/models/transformer_model.py

import torch
import pytorch_lightning as pl
from torch import nn

class TransformerModel(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        x: [batch, seq_len, input_size]
        returns: [batch, 1]
        """
        x = self.input_proj(x)  # → [batch, seq_len, d_model]
        out = self.transformer(x)  # → [batch, seq_len, d_model]
        last = out[:, -1, :]       # use final timestep
        return self.fc_out(last)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = nn.functional.mse_loss(y_hat, y.float())
        self.log("train_mse", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = nn.functional.mse_loss(y_hat, y.float())
        self.log("val_mse", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

if __name__ == "__main__":
    model = TransformerModel(input_size=16)
    dummy_x = torch.randn(2, 12, 16)
    out = model(dummy_x)
    assert out.shape == (2, 1)
    print("TransformerModel smoke test OK, output shape:", out.shape)
