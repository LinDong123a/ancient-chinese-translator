from argparse import ArgumentParser
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(pl.LightningModule):
    def __init__(
        self,
        src_vocab_size: int,
        trg_vocab_size: int,
        trg_sos_idx: int,
        d_model: int,
        hidden_dim: int,
        n_head: int,
        n_enc_layers: int,
        n_dec_layers: int,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        self.trg_sos_idx = trg_sos_idx
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size

        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, hidden_dim, n_head, n_enc_layers, dropout,
        )

        self.decoder = TransformerDecoder(
            trg_vocab_size, d_model, n_head, hidden_dim, n_dec_layers, dropout,
        )

        self.proj_to_vocab = nn.Linear(d_model, trg_vocab_size)

    @classmethod
    def add_model_args(cls, parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("gru")
        parser.add_argument("--d_model", type=int, default=256)
        parser.add_argument("--hidden_dim", type=int, default=256)
        parser.add_argument("--n_head", type=int, default=4)
        parser.add_argument("--n_enc_layers", type=int, default=6)
        parser.add_argument("--n_dec_layers", type=int, default=6)
        parser.add_argument("--dropout", type=float, default=0.1)

        cls.parser = parser

        return parent_parser

    def build_pad_mask(
        self,
        sizes: torch.Tensor,
        max_seq_len: int,
    ) -> torch.BoolTensor:
        """创建encoder的mask矩阵

        Args:
            sizes (torch.Tensor): [batch size]
            max_seq_len (int): 最大长度

        Returns:
            torch.BoolTensor: [batch size, max seq len]
        """
        batch_size = sizes.size(0)

        arange = torch.arange(
            0, max_seq_len, device=self.device,
        ).unsqueeze(0).repeat(batch_size, 1)
        size_array = sizes.unsqueeze(1).repeat(1, max_seq_len)

        return arange < size_array

    def get_subseq_mask(self, len_seq: torch.Tensor) -> torch.BoolTensor:
        """生成下三角阵

        Args:
            len_seq (torch.Tensor): 下三角阵的长度

        Returns:
            torch.BoolTensor: [len_seq, len_seq]
        """
        return (
            1 - torch.triu(
                torch.ones(len_seq, len_seq, device=self.device), diagonal=1,
            )
        ).bool()

    def build_src_and_trg_mask(
        self,
        len_src: int,
        src_sizes: torch.Tensor,
        len_trg: int,
        trg_sizes: torch.Tensor,
    ) -> Tuple[torch.BoolTensor, torch.BoolTensor, torch.BoolTensor]:
        """创建encoder和decoder过程中需要用到的mask矩阵

        Args:
            len_src (int): 输入的长度
            src_sizes (torch.Tensor): [batch size]
            len_trg (int): 输出的长度
            trg_sizes (torch.Tensor): [batch size]

        Returns:
            Tuple[torch.BoolTensor, , torch.BoolTensor]:
            torch.BoolTensor: [batch_size, len_src, len_src], 输入的attn矩阵
            torch.BoolTensor: [batch size, len_trg, len_src], 输出对输出的mask矩阵
            torch.BoolTensor: [batch size, len_trg, len_trg], 输出的attn矩阵
        """
        # [batch size, max enc len]
        enc_mask = self.build_pad_mask(src_sizes, len_src)
        # [batch size, max dec len]
        dec_mask = self.build_pad_mask(trg_sizes, len_trg)

        # [batch size, max enc len, max enc len]
        enc_attn_mask = enc_mask.unsqueeze(2) * enc_mask.unsqueeze(1)
        # [batch size, max dec len, max enc len]
        enc_dec_attn_mask = dec_mask.unsqueeze(2) * enc_mask.unsqueeze(1)
        # [batch size, max dec len, max dec len]
        dec_attn_mask = (
            dec_mask.unsqueeze(2) * dec_mask.unsqueeze(1)
            & self.get_subseq_mask(len_trg).unsqueeze(0)
        )

        return enc_attn_mask, enc_dec_attn_mask, dec_attn_mask

    def forward(
        self,
        src_token_ids: torch.Tensor,
        src_sizes: torch.Tensor,
        trg_token_ids: torch.Tensor,
        trg_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            src_token_ids (torch.Tensor): [batch size, max enc len]
            src_sizes (torch.Tensor): [batch size]
            trg_token_ids (torch.Tensor): [batch size, max dec len]
            trg_sizes (torch.Tensor): [batch size]

        Returns:
            torch.Tensor: [batch size, max sequence len, trg vocab size]
        """
        # [batch size, max enc len]
        enc_mask = self.build_pad_mask(src_sizes, src_token_ids.size(1))
        # [batch size, max dec len]
        dec_mask = self.build_pad_mask(trg_sizes, trg_token_ids.size(1))

        # [batch size, max enc len, max enc len]
        enc_attn_mask = enc_mask.unsqueeze(2) * enc_mask.unsqueeze(1)
        # [batch size, max dec len, max enc len]
        enc_dec_attn_mask = dec_mask.unsqueeze(2) * enc_mask.unsqueeze(1)
        # [batch size, max dec len, max dec len]
        dec_attn_mask = (
            dec_mask.unsqueeze(2) * dec_mask.unsqueeze(1)
            & self.get_subseq_mask(trg_token_ids.size(1)).unsqueeze(0)
        )

        enc_outputs = self.encoder(src_token_ids, mask=enc_attn_mask)
        dec_outputs = self.decoder(
            trg_token_ids,
            enc_outputs,
            dec_mask=dec_attn_mask,
            enc_dec_mask=enc_dec_attn_mask,
        )

        return self.proj_to_vocab(dec_outputs)

    def inference(
        self,
        src_token_ids: torch.Tensor,
        src_sizes: torch.Tensor,
        trg_eos_idx: int,
        max_seq_len: int = 128,
        beam_size: int = 3,
    ) -> torch.Tensor:
        assert src_token_ids.size(0) == 1, "目前仅支持batch size为1的beam search"

        trg_token_ids = torch.tensor([[self.trg_sos_idx]], device=self.device).long()

        enc_mask = self.build_pad_mask(src_sizes, src_token_ids.size(1))
        enc_attn_mask = enc_mask.unsqueeze(2) * enc_mask.unsqueeze(1)
        enc_dec_mask = enc_mask.unsqueeze(1).repeat(beam_size, max_seq_len, 1)

        gen_seqs = torch.full(
            (beam_size, max_seq_len), self.trg_sos_idx, device=self.device,
        )
        beam_log_probs = torch.zeros(beam_size, device=self.device)

        len_map = torch.arange(1, max_seq_len + 1).unsqueeze(0)

        with torch.no_grad():
            enc_outputs = self.encoder(src_token_ids, mask=enc_attn_mask)
            dec_output = self.decoder(trg_token_ids, enc_outputs)

            trg_vocab_probs = F.log_softmax(self.proj_to_vocab(dec_output), dim=-1)
            trg_vocab_probs = trg_vocab_probs[0, -1]
            beam_log_probs, gen_seqs[:, 1] = trg_vocab_probs.topk(beam_size)

            # [beam size, max src seq len,  d_model]
            # 通过这种方式使得计算一个step的beam时，能够并行计算完成
            enc_outputs = enc_outputs.repeat(beam_size, 1, 1)

            ans_idx = 0
            for step in range(2, max_seq_len):
                dec_output = self.decoder(
                    gen_seqs[:, :step],
                    enc_outputs,
                    dec_mask=self.get_subseq_mask(step).unsqueeze(0),
                    enc_dec_mask=enc_dec_mask[:, :step],
                )

                trg_vocab_probs = F.log_softmax(
                    self.proj_to_vocab(dec_output[:, -1]), dim=-1,
                )
                topk_value, topk_idx = trg_vocab_probs.topk(beam_size, dim=-1)

                topk_value = (
                    beam_log_probs.view(beam_size, -1) + topk_value
                ).view(-1)

                beam_log_probs, beam_topk_idx = topk_value.topk(beam_size)

                beam_prev_idx, beam_cur_idx = (
                    beam_topk_idx // beam_size, beam_topk_idx % beam_size,
                )
                best_beam_idx = topk_idx[beam_prev_idx, beam_cur_idx]

                gen_seqs[:, :step] = gen_seqs[beam_prev_idx, :step]
                gen_seqs[:, step] = best_beam_idx

                # 以下处理逻辑借鉴自：https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Translator.py # noqa
                # Check if all path finished
                # -- locate the eos in the generated sequences
                eos_locs = gen_seqs == trg_eos_idx
                # -- replace the eos with its position for the length penalty use
                seq_lens, _ = len_map.masked_fill(~eos_locs, max_seq_len).min(1)
                # -- check if all beams contain eos
                if (eos_locs.sum(1) > 0).sum(0).item() == beam_size:
                    _, ans_idx = beam_log_probs.div(seq_lens.float() ** 0.7).max(0)
                    ans_idx = ans_idx.item()
                    break

        # return gen_seqs[beam_log_probs.max(dim=-1).indices.item(), 1:]
        return gen_seqs[ans_idx][1:seq_lens[ans_idx]]


class PositionalEncoding(pl.LightningModule):

    def __init__(self, d_model, max_position=200):
        super().__init__()

        # Not a parameter
        self.register_buffer(
            "pos_table", self._get_sinusoid_encoding_table(max_position, d_model),
        )

    def _get_sinusoid_encoding_table(self, n_position, d_model):
        """Sinusoid position encoding table"""
        def get_position_angle_vec(position):
            return [
                position / 10000 ** (2 * (hid_j // 2) / d_model)
                for hid_j in range(d_model)
            ]

        # [max position, d_model]
        sinusoid_table = torch.FloatTensor([
            get_position_angle_vec(pos_i) for pos_i in range(n_position)
        ]).to(self.device)
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        # 将batch size维度置为1以利用broadcast
        return sinusoid_table.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [batch size, max seq len, d_model]

        Returns:
            torch.Tensor: [batch size, max seq len, d_model]
        """
        return x + self.pos_table[:, :x.size(1)]


class TransformerEncoder(pl.LightningModule):
    def __init__(
        self,
        src_vocab_size: int,
        d_model: int,
        hidden_dim: int,
        n_head: int,
        n_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        self.enc_layer_list = nn.ModuleList([
            TransformerEncoderLayer(d_model, hidden_dim, n_head, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            enc_token_ids (torch.Tensor): [batch size, max seq len]
            mask (torch.Tensor): [batch size, max seq len, max seq len]

        Returns:
            torch.Tensor: [batch size, max seq len, d_model]
        """
        enc_embed = self.dropout(self.embedding(enc_token_ids))
        enc_input = self.pos_encoding(enc_embed)

        enc_output = enc_input
        for enc_layer in self.enc_layer_list:
            enc_output = enc_layer(enc_output, mask=mask)

        return enc_output


class TransformerEncoderLayer(pl.LightningModule):
    def __init__(self, d_model: int, hidden_dim: int, n_head: int, dropout: float = 0.1):
        super().__init__()

        assert d_model == hidden_dim
        self.head_dim = hidden_dim // n_head

        self.self_attn = MultiHeadAttention(
            d_model, self.head_dim, self.head_dim, n_head, dropout=dropout,
        )
        self.ffn = PositionalFeedForward(d_model, 4 * d_model, dropout=dropout)

    def forward(self, enc_input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            enc_input (torch.Tensor): [batch size, max seq len, d_model]
            mask (torch.Tensor): [batch size, max seq len, max seq len]

        Returns:
            torch.Tensor: [batch size, max seq len, d_model]
        """
        self_attn_res = self.self_attn(enc_input, enc_input, enc_input, mask=mask)
        ffn_res = self.ffn(self_attn_res)

        return ffn_res


class TransformerDecoder(pl.LightningModule):
    def __init__(
        self,
        trg_vocab_size: int,
        d_model: int,
        n_head: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(trg_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        self.dec_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_head, hidden_dim, dropout)
            for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        dec_inputs: torch.Tensor,
        enc_output: torch.Tensor,
        dec_mask: torch.Tensor = None,
        enc_dec_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            dec_inputs (torch.Tensor): [batch size, max dec seq len]
            enc_output (torch.Tensor): [batch size, max enc seq len, d_model]
            dec_mask (torch.Tensor): [batch size, max dec seq len, max dec seq len]
            enc_dec_mask (torch.Tensor): [batch size, max dec seq len, max enc seq len]

        Returns:
            torch.Tensor: [batch size, max dec seq len, trg_vocab_size]
        """
        dec_embed = self.dropout(self.embedding(dec_inputs))
        dec_inputs = self.pos_encoding(dec_embed)

        dec_output = dec_inputs
        for module in self.dec_layers:
            dec_output = module(
                dec_output, enc_output, dec_mask=dec_mask, enc_dec_mask=enc_dec_mask,
            )

        return dec_output


class TransformerDecoderLayer(pl.LightningModule):
    def __init__(self, d_model: int, n_head: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        self.head_dim = hidden_dim // n_head

        self.self_attn = MultiHeadAttention(
            d_model, self.head_dim, self.head_dim, n_head, dropout,
        )
        self.enc_dec_attn = MultiHeadAttention(
            d_model, self.head_dim, self.head_dim, n_head, dropout,
        )
        self.ffn = PositionalFeedForward(d_model, 4 * d_model, dropout)

    def forward(
        self,
        dec_input: torch.Tensor,
        enc_output: torch.Tensor,
        dec_mask: torch.Tensor,
        enc_dec_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            dec_input (torch.Tensor): [batch size, max dec seq len, d_model]
            enc_output (torch.Tensor): [batch size, max enc seq len, d_model]
            dec_mask (torch.Tensor): [batch size, max dec seq len, max dec seq len]
            enc_dec_mask (torch.Tensor): [batch size, max dec seq len, max enc seq len]

        Returns:
            [type]: [description]
        """
        dec_output = self.self_attn(
            dec_input, dec_input, dec_input, mask=dec_mask,
        )
        dec_output = self.enc_dec_attn(
            dec_output, enc_output, enc_output, mask=enc_dec_mask,
        )

        dec_output = self.ffn(dec_output)

        return dec_output


class MultiHeadAttention(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        d_k: int,
        d_v: int,
        n_head: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, d_k * n_head, bias=False)
        self.w_ks = nn.Linear(d_model, d_k * n_head, bias=False)
        self.w_vs = nn.Linear(d_model, d_v * n_head, bias=False)

        self.attn = ScaledDotAttention(d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.BoolTensor = None,
    ) -> torch.Tensor:
        """
        Args:
            q (torch.Tensor): [batch size, len_q, d_model]
            k (torch.Tensor): [batch size, len_k, d_model]
            v (torch.Tensor): [batch size, len_v, d_model]
            mask (torch.BoolTensor, optional): [batch size, len_q, len_k]. Defaults to None.

        Returns:
            torch.Tensor: [batch size, len_q, d_v * n_head]
        """
        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        qs = self.w_qs(q).view(batch_size, len_q, self.n_head, self.d_k)
        ks = self.w_ks(k).view(batch_size, len_k, self.n_head, self.d_k)
        vs = self.w_vs(v).view(batch_size, len_v, self.n_head, self.d_v)

        # output = [batch size, n_head, len_q, d_v]
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)

        output, _ = self.attn(
            qs.transpose(1, 2),
            ks.transpose(1, 2),
            vs.transpose(1, 2),
            mask=mask,
        )

        # [batch size, len_q, n_head * d_v]
        output = output.transpose(1, 2).reshape(batch_size, len_q, -1)
        output = self.layer_norm(output + residual)

        return output


class ScaledDotAttention(pl.LightningModule):
    def __init__(self, d_model: int):
        super().__init__()

        self.scale = d_model ** 0.5

    def forward(
        self,
        qs: torch.Tensor,
        ks: torch.Tensor,
        vs: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            qs (torch.Tensor): [..., len_q, d_k]
            ks (torch.Tensor): [..., len_k, d_k]
            vs (torch.Tensor): [..., len_v, d_v]
            mask (torch.Tensor, optional): [..., len_q, len_k]

        Returns:
            torch.Tensor: [..., len_q, d_v], attention后的结果
            torch.Tensor: [..., len_q, len_k]，attention值
        """
        assert qs.size(-1) == ks.size(-1), "query and key must have same embedding dim"
        assert ks.size(-2) == vs.size(-2), "key and value must have same length"

        # [..., len_q, len_k]
        attn = torch.matmul(qs, ks.transpose(-2, -1)) / self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e4)

        attn = torch.softmax(attn, dim=-1)

        # [..., len_q, d_v]
        output = torch.matmul(attn, vs)

        return output, attn


class PositionalFeedForward(pl.LightningModule):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [..., input dim]

        Returns:
            torch.Tensor: [..., input dim]
        """
        residual = x

        x = self.linear2(F.relu(self.linear1(x)))

        return self.layer_norm(x + residual)
