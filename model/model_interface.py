import pytorch_lightning as pl
import torch
import torch.nn as nn

from data.vocab import Vocab


class ModelInterface(pl.LightningModule):
    def __init__(self, model_name: str, src_vocab: Vocab, trg_vocab: Vocab):
        super().__init__()

        if model_name == "GRU":
            from .gru import GRU_Translator
            self.model = GRU_Translator(128, 256, src_vocab, trg_vocab, self.device)
        else:
            raise ValueError(f"Unrecognized model: {model_name}")

        self.loss = nn.CrossEntropyLoss(ignore_index=trg_vocab.pad_idx)

    def forward(self, batch, batch_idx):
        decoder_outputs = self.model(
            batch["src"], batch["src_size"], trg_token_ids=batch["trg"],
        )

        decoder_outputs = decoder_outputs[:, 1:].reshape(-1, decoder_outputs.size(2))
        gt_trg = batch["trg"].flatten()

        return self.loss(decoder_outputs, gt_trg)

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch, batch_idx)

        # self.log("train/loss", loss, on_epoch=True, on_step=True)
        self.log("train/loss", loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch, batch_idx)

        self.log("valid/loss", loss, on_epoch=True, on_step=True)

    def test_step(self, batch, batch_idx):
        loss = self.forward(batch, batch_idx)

        self.log("test/loss", loss, on_epoch=True, on_step=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.02)
