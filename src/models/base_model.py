# src/models/base_model.py

from abc import ABC, abstractmethod
import torch
import pytorch_lightning as pl

class BaseModel(pl.LightningModule, ABC):
    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.lr = lr
        self.save_hyperparameters()

    @abstractmethod
    def forward(self, x):
        """Return model predictions."""
        pass

    @abstractmethod
    def training_step(self, batch, batch_idx):
        """Compute train loss."""
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        """Compute val loss/metrics."""
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

if __name__ == "__main__":
    # Smoke test: instantiate and pass dummy input through forward
    class DummyModel(BaseModel):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(10, 1)
        def forward(self, x): return self.lin(x)
        def training_step(self, batch, idx):
            x, y = batch
            loss = torch.nn.functional.mse_loss(self(x), y)
            return loss
        def validation_step(self, batch, idx): return self.training_step(batch, idx)

    model = DummyModel()
    dummy_x = torch.randn(8, 10)
    out = model(dummy_x)
    print("BaseModel smoke test OK, output shape:", out.shape)
