import random
from argparse import ArgumentParser
from typing import List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn

from data.vocab import Vocab


class ModelInterface(pl.LightningModule):
    def __init__(
        self,
        src_vocab: Vocab,
        trg_vocab: Vocab,
        lr: float,
        num_epoch: int,
        steps_per_epoch: int,
        model_config: dict,
    ):
        super().__init__()

        self.num_epoch = num_epoch
        self.steps_per_epoch = steps_per_epoch
        self.lr = lr

        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

        self.model = self.model_cls(
            src_vocab=src_vocab, trg_vocab=trg_vocab, **model_config,
        )

        self.loss = nn.CrossEntropyLoss(ignore_index=trg_vocab.pad_idx)

    @classmethod
    def add_trainer_args(cls, parent_parser: ArgumentParser):
        parent_parser.add_argument("--model", type=str, default="GRU", help="模型类型")
        known_args, _ = parent_parser.parse_known_args()

        cls.model_name = known_args
        if known_args.model == "GRU":
            from .gru import GRU_Translator
            cls.model_cls = GRU_Translator
            GRU_Translator.add_model_args(parent_parser)
        else:
            raise ValueError(f"Unrecognized model: {known_args.model}")

        parser = parent_parser.add_argument_group("trainer")

        parser.add_argument("--lr", type=float, default=0.01, help="模型学习率")

        cls.parser = parser

        return parent_parser

    def forward(self, batch, batch_idx):
        decoder_outputs = self.model(
            batch["src"], batch["src_size"], trg_token_ids=batch["trg"],
        )

        decoder_outputs = decoder_outputs[:, 1:].reshape(-1, decoder_outputs.size(2))
        gt_trg = batch["trg"].flatten()

        return self.loss(decoder_outputs, gt_trg), decoder_outputs

    def inference(self, src: torch.Tensor, src_size: torch.Tensor) -> List[str]:
        decoder_output = self.model(
            src.to(self.device), src_size.to(self.device), max_sequence_len=128,
        )

        sent_list = []
        for i in range(decoder_output.size(0)):
            token_ids = decoder_output[i].max(dim=-1).indices
            token_list = []
            for tid in token_ids:
                if tid == self.trg_vocab.eos_idx:
                    break

                token_list.append(self.trg_vocab.itos(tid.item()))

            sent_list.append("".join(token_list))

        return sent_list

    def training_step(self, batch, batch_idx):
        loss, _ = self.forward(batch, batch_idx)

        # self.log("train/loss", loss, on_epoch=True, on_step=True)
        self.log("train/loss", loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self.forward(batch, batch_idx)

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
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            epochs=self.num_epoch,
            steps_per_epoch=self.steps_per_epoch,
        )

        return [optimizer], [{
            "scheduler": scheduler,
            "interval": "step",
        }]
