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
        d_model: int,
        hidden_dim: int,
        n_head: int,
        n_enc_layers: int,
        n_dec_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, hidden_dim, n_head, n_enc_layers, dropout,
        )

        self.decoder = TransformerDecoder(
            trg_vocab_size, d_model, n_head, hidden_dim, n_dec_layers, dropout,
        )

        self.proj_to_vocab = nn.Linear(d_model, trg_vocab_size)

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
            torch.BoolTensor: [batch size, max seq len, max seq len]
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
        return torch.triu(
            torch.ones(len_seq, len_seq, device=self.device), diagonal=0,
        ).bool()

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
        dec_mask: torch.Tensor,
        enc_dec_mask: torch.Tensor,
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
            d_model, self.head_dim, self.head_dim, dropout,
        )
        self.enc_dec_attn = MultiHeadAttention(
            d_model, self.head_dim, self.head_dim, dropout,
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
        output, _ = self.attn(
            qs.transpose(1, 2),
            ks.transpose(1, 2),
            vs.transpose(1, 2),
            mask=mask.unsqueeze(1).repeat(1, self.n_head, 1, 1),
        )

        # [batch size, len_q, n_head * d_v]
        output = output.transpose(1, 2).view(batch_size, len_q, -1)
        output = self.layer_norm(output + residual)

        return output


class ScaledDotAttention(pl.LightningModule):
    def __init__(self, d_model: int):
        super().__init__()

        self.scale = torch.sqrt(d_model)

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
            attn = attn.masked_fill(mask, -1e9)

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
