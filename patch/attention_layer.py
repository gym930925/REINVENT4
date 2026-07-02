import math

import torch
from torch import nn as tnn


class AttentionLayer(tnn.Module):
    def __init__(self, num_dimensions):
        super(AttentionLayer, self).__init__()

        self.num_dimensions = num_dimensions
        self._scale = math.sqrt(self.num_dimensions)

        self._attention_linear = tnn.Sequential(
            tnn.Linear(self.num_dimensions * 2, self.num_dimensions), tnn.Tanh()
        )

    def forward(
        self, padded_seqs, encoder_padded_seqs, decoder_mask
    ):  # pylint: disable=arguments-differ
        """
        Performs the forward pass.
        :param padded_seqs: A tensor with the output sequences (batch, seq_d, dim).
        :param encoder_padded_seqs: A tensor with the encoded input scaffold sequences (batch, seq_e, dim).
        :param decoder_mask: A tensor that represents the encoded input mask.
        :return : Two tensors: one with the modified logits and another with the attention weights.
        """
        # scaled dot-product
        # (batch, seq_d, 1, dim)*(batch, 1, seq_e, dim) => (batch, seq_d, seq_e*)
        # attention_scores = (
        #     (padded_seqs.unsqueeze(dim=2) * encoder_padded_seqs.unsqueeze(dim=1))
        #     .sum(dim=3)
        #     .div(self._scale)
        # )
        attention_scores = padded_seqs.bmm(
            encoder_padded_seqs.transpose(1, 2)
        ).div(self._scale)
        encoder_mask = (
            encoder_padded_seqs.abs().sum(dim=-1) != 0
        ).unsqueeze(1)
        attention_scores = attention_scores.masked_fill(~encoder_mask, float("-inf"))
        attention_weights = attention_scores.softmax(dim=2)
        # (batch, seq_d, seq_e*)@(batch, seq_e, dim) => (batch, seq_d, dim)
        attention_context = attention_weights.bmm(encoder_padded_seqs)
        return (
            self._attention_linear(torch.cat([padded_seqs, attention_context], dim=2))
            * decoder_mask,
            attention_weights,
        )

    def forward_single_step(
        self, padded_seqs, encoder_padded_seqs, encoder_mask, decoder_mask
    ):
        """NPU 优化：采样专用单步 attention，复用预计算的 encoder_mask。

        :param padded_seqs: (batch, 1, dim) decoder 当前步输出
        :param encoder_padded_seqs: (batch, seq_e, dim) encoder 输出
        :param encoder_mask: (batch, 1, seq_e) 预计算的 encoder 有效位置 mask
        :param decoder_mask: (batch, 1, 1) decoder 有效位置 mask
        :return: (batch, 1, dim), (batch, 1, seq_e)
        """
        # 用 bmm 替代 unsqueeze+sum 广播，减少 NPU kernel 数量
        # (batch, 1, dim) @ (batch, dim, seq_e) -> (batch, 1, seq_e)
        attention_scores = padded_seqs.bmm(
            encoder_padded_seqs.transpose(1, 2)
        ).div(self._scale)

        attention_scores = attention_scores.masked_fill(~encoder_mask, float("-inf"))
        attention_weights = attention_scores.softmax(dim=2)
        # (batch, 1, seq_e) @ (batch, seq_e, dim) -> (batch, 1, dim)
        attention_context = attention_weights.bmm(encoder_padded_seqs)
        return (
            self._attention_linear(torch.cat([padded_seqs, attention_context], dim=2))
            * decoder_mask,
            attention_weights,
        )
