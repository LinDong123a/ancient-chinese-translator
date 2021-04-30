import random
from typing import List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn

from data.vocab import Vocab


class ModelInterface(pl.LightningModule):
    def __init__(self, model_name: str, src_vocab: Vocab, trg_vocab: Vocab):
        super().__init__()

        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

        if model_name == "GRU":
            from .gru import GRU_Translator
            self.model = GRU_Translator(128, 256, src_vocab, trg_vocab)
        else:
            raise ValueError(f"Unrecognized model: {model_name}")

        self.loss = nn.CrossEntropyLoss(ignore_index=trg_vocab.pad_idx)

    def forward(self, batch, batch_idx):
        decoder_outputs = self.model(
            batch["src"], batch["src_size"], trg_token_ids=batch["trg"],
        )

        decoder_outputs = decoder_outputs[:, 1:].reshape(-1, decoder_outputs.size(2))
        gt_trg = batch["trg"].flatten()

        return self.loss(decoder_outputs, gt_trg), decoder_outputs

    def training_step(self, batch, batch_idx):
        loss, _ = self.forward(batch, batch_idx)

        # self.log("train/loss", loss, on_epoch=True, on_step=True)
        self.log("train/loss", loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, predicts = self.forward(batch, batch_idx)

        self.log("valid/loss", loss, on_step=True)

        return loss

    def _validation_epoch_end(self, outputs: List[Tuple[torch.Tensor, torch.Tensor]]):
        output = random.choice(outputs)

        print(output[1].shape)
        for i in range(output[1].size(0)):
            input_token_ids, probs = output[0][i].tolist(), output[1][i]
            token_ids = probs.max(dim=-1).indices.tolist()

            print(input_token_ids)
            print(token_ids)

            try:
                print(
                    "".join([
                        self.src_vocab.itos(_id)
                        for _id in input_token_ids[
                            :input_token_ids.index(self.src_vocab.pad_idx)
                        ]
                    ]),
                )
            except ValueError:
                print(
                    "".join([
                        self.src_vocab.itos(_id) for _id in input_token_ids
                    ]),
                )

            try:
                print(
                    "".join([
                        self.trg_vocab.itos(_id)
                        for _id in token_ids[
                            :token_ids.index(self.trg_vocab.pad_idx)
                        ]
                    ]),
                )
            except ValueError:
                print(
                    "".join([self.trg_vocab.itos(_id) for _id in token_ids]),
                )

    def test_step(self, batch, batch_idx):
        loss, _ = self.forward(batch, batch_idx)

        self.log("test/loss", loss, on_epoch=True, on_step=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)
