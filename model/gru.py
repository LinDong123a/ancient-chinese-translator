import random
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn

# import torch.nn.functional as F


class GRU_Translator(pl.LightningModule):
    def __init__(
        self,
        embedding_dim: int,
        enc_hidden_dim: int,
        dec_hidden_dim: int,
        dropout: float,
        src_vocab_size: int,
        trg_vocab_size: int,
        trg_sos_idx: int,
    ) -> None:
        super().__init__()

        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.trg_sos_idx = trg_sos_idx

        self.encoder = GRU_Encoder(
            src_vocab_size, embedding_dim, enc_hidden_dim, dec_hidden_dim, dropout,
        )
        self.attn = Attention(enc_hidden_dim, dec_hidden_dim)
        self.decoder = GRU_Decoder(
            trg_vocab_size,
            embedding_dim,
            enc_hidden_dim,
            dec_hidden_dim,
            dropout,
            self.attn,
        )

        self.save_hyperparameters(
            "embedding_dim", "enc_hidden_dim", "dec_hidden_dim", "dropout",
            "src_vocab_size", "trg_vocab_size", "trg_sos_idx",
        )

    @classmethod
    def add_model_args(cls, parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("gru")
        parser.add_argument("--embedding_dim", type=int, default=128)
        parser.add_argument("--enc_hidden_dim", type=int, default=256)
        parser.add_argument("--dec_hidden_dim", type=int, default=256)
        parser.add_argument("--dropout", type=float, default=0.1)

        cls.parser = parser

        return parent_parser

    def forward(
        self,
        src_token_ids: torch.Tensor,
        src_sizes: torch.Tensor,
        max_sequence_len: int = None,
        trg_token_ids: torch.Tensor = None,
        teacher_forcing: float = 1,
    ) -> torch.Tensor:
        """
        Args:
            src_token_ids (Tensor): 原始文本，[batch size, max sequnece len]
            src_sizes (Tensor): 原始文本中每个样本的实际长度
            max_sequence_len (int): 解码时的最长长度
            trg_token_ids (Tensor, optional): 目标文本，[batch size, max sequence len],
                当需要teacher forcing时才需要添加，默认为None
            teacher_forcing (float): teacher forcing的概率

        Returns:
            torch.Tensor, [batch size, max sequence len, label size]
        """
        if max_sequence_len is None and trg_token_ids is None:
            raise ValueError("one of max_sequence_len or trg_token_ids must be specifed")

        batch_size = src_token_ids.size(0)

        encoder_outs, d_hn = self.encoder(src_token_ids, src_sizes)

        src_max_sequence_len = encoder_outs.size(1)
        encode_len_arange = torch.arange(
            0, src_max_sequence_len, device=self.device,
        ).unsqueeze(0).repeat(batch_size, 1)
        encode_mask = (
            encode_len_arange >= src_sizes.unsqueeze(1).repeat(1, src_max_sequence_len)
        )

        if trg_token_ids is not None:
            max_sequence_len = trg_token_ids.size(1)

        decoder_outputs = torch.zeros(
            batch_size, max_sequence_len, self.trg_vocab_size, device=self.device,
        )
        decoder_input = torch.LongTensor([self.trg_sos_idx]).repeat(batch_size)
        for step in range(max_sequence_len):
            decoder_out, d_hn = self.decoder(
                decoder_input.unsqueeze(1), d_hn, encoder_outs, encode_mask,
            )

            # [batch size, 1, embedding dim]
            decoder_outputs[:, step] = decoder_out

            if random.random() < teacher_forcing and trg_token_ids is not None:
                decoder_input = trg_token_ids[:, step]
            else:
                decoder_input = decoder_out.max(dim=-1).indices

        return decoder_outputs


class GRU_Encoder(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        dec_hidden_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(
            embedding_dim, hidden_dim, bidirectional=True, batch_first=True,
        )
        self.hn_fc = nn.Linear(2 * hidden_dim, dec_hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src_token_ids: torch.Tensor,
        src_sizes: torch.Tensor,
    ):
        """编码输入的文本

        Args:
            src_token_ids (torch.Tensor): 原始文本，[batch size, max sequnece len]
            src_sizes (torch.Tensor): 原始文本中每个样本的实际长度

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                0: 每个token的编码结果，[batch size, max sequence len, 2 * hidden]
                1: 输入句子的embedding表达, [1, batch size, dec_hidden_dim]
        """
        # [batch size, max sequence len, embedding dim]
        src_token_embed = self.dropout(self.embedding(src_token_ids))

        packed_src_token_ids = nn.utils.rnn.pack_padded_sequence(
            src_token_embed, src_sizes.cpu(), batch_first=True, enforce_sorted=False,
        )

        packed_out, hn = self.rnn(packed_src_token_ids)
        # [batch size, max seq len, 2 * hidden dim]
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        # [batch size, 2 * hidden dim]
        return (
            out,
            torch.tanh(self.hn_fc(torch.cat([hn[-2], hn[-1]], dim=1).unsqueeze(0))),
        )


class Attention(pl.LightningModule):
    def __init__(self, enc_hidden_dim: int, dec_hidden_dim: int):
        super().__init__()

        self.attn = nn.Linear(2 * enc_hidden_dim + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)

    def forward(
        self,
        dec_hn: torch.Tensor,
        encoder_outputs: torch.Tensor,
        encode_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            encoder_outputs (torch.Tensor): 输入层的编码信息，
                [batch size, max sequence len, 2 * hidden dim]
            encode_mask (torch.Tensor): 输入的长度mask，对于padding元素不进行
                attention, [batch size, max sequence len]

        Returns:
            torch.Tensor: [batch size, max sequnce len]，attention的分数
        """
        # [batch size, max sequence len , dec_hidden_dim]
        energy = torch.tanh(
            self.attn(
                torch.cat(
                    [
                        encoder_outputs,
                        dec_hn.squeeze(0).unsqueeze(1).repeat(1, encoder_outputs.size(1), 1),
                    ],
                    dim=-1,
                ),
            ),
        )

        # [batch size, max seq len]
        attention = self.v(energy).squeeze(dim=2)

        return torch.softmax(attention.masked_fill(encode_mask, -1e4), dim=-1)


class GRU_Decoder(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        enc_hidden_dim: int,
        hidden_dim: int,
        dropout: float,
        attention: Attention,
    ):
        super().__init__()
        self.attn = attention

        self.dropout = nn.Dropout(dropout)

        self.trg_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(
            2 * enc_hidden_dim + embedding_dim, hidden_dim, batch_first=True,
        )
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        token_id: torch.Tensor,
        hn: torch.Tensor,
        encoder_outputs: torch.Tensor,
        encode_mask: torch.Tensor,
    ):
        """完成一个时间步的解码

        Args:
            token_id (torch.Tensor): 输入的token id, [batch size, 1]
            hn (torch.Tensor): GRU的中间状态，[1, bach_size, 2 * hidden dim]
            encoder_outputs (torch.Tensor): 输入层的编码信息，
                [batch size, max sequence len, 2 * hidden dim]
            encode_mask (torch.Tensor): 输入的长度mask，对于padding元素不进行
                attention, [batch size, max sequence len]

        Returns:
            torch.Tensor: [batch size, 1, vocab size]
            torch.Tensor: [batch size, dec hidden dim]
        """
        # [batch size, 1, embedding dim]
        deocder_input = self.dropout(self.trg_embedding(token_id))

        # [batch size, max seq len]
        attn_score = self.attn(hn, encoder_outputs, encode_mask)

        # [batch size, 2 * hidden dim]
        atten_embed = torch.matmul(attn_score.unsqueeze(1), encoder_outputs).squeeze(1)

        output, d_hn = self.rnn(
            torch.cat([deocder_input, atten_embed.unsqueeze(1)], dim=-1),
            hn,
        )

        return self.out(torch.tanh(output.squeeze(dim=1))), d_hn
