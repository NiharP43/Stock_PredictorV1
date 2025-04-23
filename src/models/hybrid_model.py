# src/models/hybrid_model.py

import torch
import pytorch_lightning as pl
from torch import nn
from transformers import AutoModel, AutoTokenizer

class HybridModel(pl.LightningModule):
    def __init__(
        self,
        price_input_size: int,
        sentiment_model_name: str = "prosusai/finbert",
        lstm_hidden: int = 32,
        sentiment_hidden: int = 32,
        combined_hidden: int = 64,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Price LSTM branch
        self.lstm = nn.LSTM(price_input_size, lstm_hidden, batch_first=True)

        # Sentiment encoder (FinBERT)
        self.tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
        self.sentiment_encoder = AutoModel.from_pretrained(sentiment_model_name)

        # Combine
        self.fc1 = nn.Linear(lstm_hidden + sentiment_hidden, combined_hidden)
        self.out = nn.Linear(combined_hidden, 1)

        # Project BERT CLS to sentiment_hidden
        self.sent_proj = nn.Linear(self.sentiment_encoder.config.hidden_size, sentiment_hidden)

    def forward(self, price_ts, sentiment_texts):
        """
        price_ts: [batch, seq_len, price_input_size]
        sentiment_texts: List[str] of length batch
        """
        # Price branch
        price_out, _ = self.lstm(price_ts)
        price_feat = price_out[:, -1, :]  # [batch, lstm_hidden]

        # Sentiment branch: encode text
        encoded = self.tokenizer(
            sentiment_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        ).to(price_ts.device)
        bert_out = self.sentiment_encoder(**encoded)
        cls_emb = bert_out.last_hidden_state[:, 0, :]  # [batch, hidden_size]
        sent_feat = self.sent_proj(cls_emb)           # [batch, sentiment_hidden]

        # Combine and predict
        combined = torch.relu(self.fc1(torch.cat([price_feat, sent_feat], dim=1)))
        return self.out(combined)

    def training_step(self, batch, batch_idx):
        price_ts, texts, y = batch
        y_hat = self(price_ts, texts).squeeze()
        loss = nn.functional.mse_loss(y_hat, y.float())
        self.log("train_mse", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        price_ts, texts, y = batch
        y_hat = self(price_ts, texts).squeeze()
        loss = nn.functional.mse_loss(y_hat, y.float())
        self.log("val_mse", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

if __name__ == "__main__":
    # Smoke test
    batch, seq_len, fts = 2, 10, 5
    price_ts = torch.randn(batch, seq_len, fts)
    texts = ["Strong earnings beat" for _ in range(batch)]
    model = HybridModel(price_input_size=fts)
    out = model(price_ts, texts)
    assert out.shape == (batch, 1)
    print("HybridModel smoke test OK, output shape:", out.shape)
