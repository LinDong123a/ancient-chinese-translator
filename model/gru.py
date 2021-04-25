import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.vocab import Vocab


class GRU_Translator(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        src_vocab: Vocab,
        trg_vocab: Vocab,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.device = device

        self.src_vocab = src_vocab
        self.src_vocab_size = len(src_vocab)

        self.trg_vocab = trg_vocab
        self.trg_vocab_size = len(trg_vocab)

        self.src_embedding = nn.Embedding(self.src_vocab_size, embedding_dim)
        self.encoder = nn.GRU(
            embedding_dim, hidden_dim, bidirectional=True, batch_first=True,
        )

        self.trg_embedding = nn.Embedding(self.trg_vocab_size, embedding_dim)
        self.decoder = nn.GRU(
            2 * hidden_dim + embedding_dim, 2 * hidden_dim, batch_first=True,
        )
        self.attn_a = nn.Linear(4 * hidden_dim, 1)
        self.out = nn.Linear(2 * hidden_dim, self.trg_vocab_size)

    def encode(
        self,
        src_token_ids: torch.Tensor,
        src_sizes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码输入的文本

        Args:
            src_token_ids (torch.Tensor): 原始文本，[batch size, max sequnece len]
            src_sizes (torch.Tensor): 原始文本中每个样本的实际长度

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                0: 每个token的编码结果，[batch size, max sequence len, 2 * hidden]
                1: 输入句子的embedding表达, [1, batch size, 2 * hidden size]
        """
        # [batch size, max sequence len, embedding dim]
        src_token_embed = self.src_embedding(src_token_ids)

        packed_src_token_ids = nn.utils.rnn.pack_padded_sequence(
            src_token_embed, src_sizes, batch_first=True, enforce_sorted=False,
        )

        packed_out, hn = self.encoder(packed_src_token_ids)
        # [batch size, max seq len, 2 * hidden dim]
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        # [batch size, 2 * hidden dim]
        return out, torch.cat([hn[0], hn[1]], dim=1).unsqueeze(0)

    def decode(
        self,
        token_id: torch.Tensor,
        hn: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> torch.Tensor:
        """完成一个时间步的解码

        Args:
            token_id (torch.Tensor): 输入的token id, [batch size, 1]
            hn (torch.Tensor): GRU的中间状态，[1, bach_size, 2 * hidden dim]
            encoder_outputs (torch.Tensor): 输入层的编码信息，
                [batch size, max sequence len, 2 * hidden dim]

        Returns:
            torch.Tensor: [batch size, 1, vocab size]
        """
        # [batch size, 1, embedding dim]
        deocder_input = self.trg_embedding(token_id)

        # [batch size, max seq len, 1]
        atten_score = self.attn_a(
            torch.cat(
                [
                    encoder_outputs,
                    hn.squeeze(0).unsqueeze(1).repeat(1, encoder_outputs.size(1), 1),
                ],
                dim=-1,
            ),
        )

        # [batch size, 2 * hidden dim]
        atten_embed = torch.matmul(atten_score.transpose(1, 2), encoder_outputs).squeeze(1)

        output, d_hn = self.decoder(
            torch.cat([deocder_input, atten_embed.unsqueeze(1)], dim=-1),
            hn,
        )

        return self.out(F.relu(output.squeeze(dim=1))), d_hn

    def forward(
        self,
        src_token_ids: torch.Tensor,
        src_sizes: torch.Tensor,
        max_sequence_len: int = None,
        trg_token_ids: torch.Tensor = None,
        teacher_forcing: float = 0,
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

        encoder_outs, hn = self.encode(src_token_ids, src_sizes)

        if trg_token_ids is not None:
            max_sequence_len = trg_token_ids.size(1)

        batch_size = src_token_ids.size(0)

        d_hn = hn
        decoder_outputs = torch.zeros(
            batch_size, max_sequence_len+1, self.trg_vocab_size, device=self.device,
        )
        decoder_outputs[:, 0, self.trg_vocab.unk_idx] = 1  # 初始化原始输入为SOS
        for step in range(1, max_sequence_len+1):
            if random.random() < teacher_forcing and trg_token_ids:
                last_decode_output = torch.LongTensor(
                    [trg_token_ids[step - 1]],
                ).repeat(batch_size)
            else:
                last_decode_output = torch.LongTensor(
                    decoder_outputs[:, step-1].max(dim=-1).indices,
                )

            decoder_out, d_hn = self.decode(
                last_decode_output.unsqueeze(1), d_hn, encoder_outs,
            )

            # [batch size, 1, embedding dim]
            decoder_outputs[:, step] = decoder_out

        return decoder_outputs
