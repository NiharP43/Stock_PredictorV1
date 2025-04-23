# src/models/gnn_model.py

import torch
import pytorch_lightning as pl
from torch import nn
from torch_geometric.nn import GCNConv

class GNNModel(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 1,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        """
        x: [num_nodes, in_channels]
        edge_index: [2, num_edges]
        returns: [num_nodes, out_channels]
        """
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        return self.lin(x)

    def training_step(self, batch, batch_idx):
        x, edge_index, y = batch
        y_hat = self(x, edge_index).squeeze()
        loss = nn.functional.mse_loss(y_hat, y.float())
        self.log("train_mse", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, edge_index, y = batch
        y_hat = self(x, edge_index).squeeze()
        loss = nn.functional.mse_loss(y_hat, y.float())
        self.log("val_mse", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

if __name__ == "__main__":
    # Smoke test with a tiny graph
    num_nodes, in_ch = 5, 8
    x = torch.randn(num_nodes, in_ch)
    # fully connected graph
    row = torch.arange(num_nodes).repeat_interleave(num_nodes)
    col = torch.arange(num_nodes).repeat(num_nodes)
    edge_index = torch.stack([row, col], dim=0)
    model = GNNModel(in_channels=in_ch)
    out = model(x, edge_index)
    assert out.shape == (num_nodes, 1)
    print("GNNModel smoke test OK, output shape:", out.shape)
