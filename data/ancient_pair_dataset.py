import argparse
import logging
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from data.vocab import Vocab

logger = logging.getLogger(__name__)


class AncientPairDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, data_dir: str):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size

        if not self.data_dir.exists():
            raise ValueError("Directory or file doesn't exist")
        if not self.data_dir.is_dir():
            raise ValueError("`data_dir` must be a path to directory")

    @classmethod
    def add_data_args(cls, parent_parser: argparse.ArgumentParser):
        parser = parent_parser.add_argument_group("data")

        parser.add_argument("--data_dir", type=str, default="./data", help="数据存储路径")
        parser.add_argument("--batch_size", type=int, default=128, help="一个batch的大小")

        cls.parser = parser

        return parent_parser

    def prepare_data(self):
        """数据已提前准备完成"""

    def setup(self, stage: Optional[str] = None):
        self.src_vocab = Vocab()
        self.src_vocab.load(str(self.data_dir / "src_vocab.json"))
        self.src_vocab_size = len(self.src_vocab)

        self.trg_vocab = Vocab()
        self.trg_vocab.load(str(self.data_dir / "trg_vocab.json"))
        self.trg_vocab_size = len(self.trg_vocab)

        self.train_dataset = AncientPairDataset(
            str(self.data_dir / "train.tsv"), 128, self.src_vocab, self.trg_vocab,
        )
        self.valid_dataset = AncientPairDataset(
            str(self.data_dir / "valid.tsv"), 128, self.src_vocab, self.trg_vocab,
        )
        self.test_dataset = AncientPairDataset(
            str(self.data_dir / "test.tsv"), 128, self.src_vocab, self.trg_vocab,
        )

        logger.info(
            f"数据集信息:\n\t"
            f"训练集: {len(self.train_dataset)}, "
            f"验证集: {len(self.valid_dataset)}, "
            f"测试集: {len(self.test_dataset)}",
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class AncientPairDataset(Dataset):
    def __init__(self, fpath: str, max_seq_len: int, src_vocab: Vocab, trg_vocab: Vocab):
        self.fpath = fpath
        self.max_seq_len = max_seq_len

        self.src_tokens_list = []
        self.trg_tokens_list = []
        with open(self.fpath, "r", encoding="utf-8") as rfile:
            for line in rfile:
                parts = line.strip().split("\t")
                src_text, trg_text = parts[0], parts[1]

                self.src_tokens_list.append(src_text.split(" "))
                self.trg_tokens_list.append(trg_text.split(" "))

        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

    def __len__(self):
        return len(self.src_tokens_list)

    def __getitem__(self, idx):
        src_token_ids = [self.src_vocab.stoi(t) for t in self.src_tokens_list[idx]]
        trg_token_ids = [self.trg_vocab.stoi(t) for t in self.trg_tokens_list[idx]]

        src_token_ids = src_token_ids[:self.max_seq_len]

        # 添加eos token
        trg_token_ids = trg_token_ids[:self.max_seq_len - 1]
        trg_token_ids.append(self.trg_vocab.eos_idx)

        src_token_len = len(src_token_ids)
        trg_token_len = len(trg_token_ids)

        src_token_ids = src_token_ids + [
            self.src_vocab.pad_idx,
        ] * max(0, self.max_seq_len - len(src_token_ids))
        trg_token_ids = trg_token_ids + [
            self.trg_vocab.pad_idx,
        ] * max(0, self.max_seq_len - len(trg_token_ids))

        return {
            "src": torch.LongTensor(src_token_ids),
            "src_size": src_token_len,
            "trg": torch.LongTensor(trg_token_ids),
            "trg_size": trg_token_len,
        }
