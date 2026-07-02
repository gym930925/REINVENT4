from typing import Tuple

import torch
from torch import nn as tnn
from torch.nn.utils import rnn as tnnur

from reinvent.models.linkinvent.networks.attention_layer import AttentionLayer
from reinvent.models.model_parameter_enum import ModelParametersEnum


class Decoder(tnn.Module):
    """
    Simple RNN decoder.
    """

    def __init__(self, num_layers: int, num_dimensions: int, vocabulary_size: int, dropout: float):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.num_dimensions = num_dimensions
        self.vocabulary_size = vocabulary_size
        self.dropout = dropout

        self._embedding = tnn.Sequential(
            tnn.Embedding(self.vocabulary_size, self.num_dimensions),
            tnn.Dropout(dropout),
        )
        self._rnn = tnn.LSTM(
            self.num_dimensions,
            self.num_dimensions,
            self.num_layers,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=False,
        )

        self._attention = AttentionLayer(self.num_dimensions)

        self._linear = tnn.Linear(self.num_dimensions, self.vocabulary_size)  # just to redimension

    def forward(
        self,
        padded_seqs: torch.Tensor,
        seq_lengths: torch.Tensor,
        encoder_padded_seqs: torch.Tensor,
        hidden_states: Tuple[torch.Tensor],
    ) -> (
        torch.Tensor,
        Tuple[torch.Tensor],
        torch.Tensor,
    ):  # pylint: disable=arguments-differ
        """
        Performs the forward pass.
        :param padded_seqs: A tensor with the output sequences (batch, seq_d, dim).
        :param seq_lengths: A list with the length of each output sequence.
        :param encoder_padded_seqs: A tensor with the encoded input sequences (batch, seq_e, dim).
        :param hidden_states: The hidden states from the encoder.
        :return : Three tensors: The output logits, the hidden states of the decoder and the attention weights.
        """
        # FIXME: this is to guard against non compatible `gpu` input for pack_padded_sequence() method in pytorch 1.7
        # P0: .size() -> .shape 避免 _local_scalar_dense 同步
        batch_size = padded_seqs.shape[0]
        max_seq_size = padded_seqs.shape[1]
        # Embedding
        padded_encoded_seqs = self._embedding(padded_seqs)
        # === NPU-safe: 使用 padded input + mask，避免 PackedSequence ===
        
        # 关键修复：根据 seq_lengths 截断序列
        # seq_lengths 已经是 linker_seq_lengths - 1
        max_output_len = seq_lengths.max().item()
        if max_output_len < max_seq_size:
            # 截断到实际需要的长度（去掉最后一个 token）
            padded_encoded_seqs = padded_encoded_seqs[:, :max_output_len, :]
            max_seq_size = max_output_len
        
        # 创建 mask 处理 padding 位置
        seq_range = torch.arange(max_seq_size, device=padded_seqs.device)
        seq_lengths_dev = seq_lengths.to(padded_seqs.device)
        mask = (seq_range.unsqueeze(0) < seq_lengths_dev.unsqueeze(1))  # (batch, seq)
        mask = mask.unsqueeze(-1).type_as(padded_encoded_seqs)  # (batch, seq, 1)
        # 确保隐藏状态在正确设备
        h_n, c_n = hidden_states
        if h_n.device != padded_encoded_seqs.device:
            h_n = h_n.to(padded_encoded_seqs.device)
            c_n = c_n.to(padded_encoded_seqs.device)
            hidden_states = (h_n, c_n)
        # 直接使用 padded input（不使用 pack_padded_sequence）
        padded_encoded_seqs, hidden_states = self._rnn(padded_encoded_seqs, hidden_states)
        # 应用 mask（将 padding 位置置零）
        padded_encoded_seqs = padded_encoded_seqs * mask
        # Attention 和后续处理
        attn_mask = (padded_encoded_seqs[:, :, 0] != 0).unsqueeze(dim=-1).type(torch.float)
        attn_padded_encoded_seqs, attention_weights = self._attention(
            padded_encoded_seqs, encoder_padded_seqs, attn_mask
        )
        
        logits = self._linear(attn_padded_encoded_seqs) * mask  # (batch, seq, voc_size)

        return logits, hidden_states, attention_weights

    def forward_single_step(
        self,
        input_token: torch.Tensor,
        encoder_padded_seqs: torch.Tensor,
        encoder_mask: torch.Tensor,
        hidden_states: Tuple[torch.Tensor],
    ) -> (torch.Tensor, Tuple[torch.Tensor], torch.Tensor):
        """NPU 优化：采样专用单步解码，复用预计算的 encoder_mask。

        采样时 decoder 输入恒为 (batch, 1) —— 这是 RNN 自回归生成的本质，
        与输入分子长度无关。跳过 seq_lengths.max().item() 和 mask 构建中的
        D2H 同步（_local_scalar_dense）。likelihood/训练路径仍走 forward()。

        :param input_token: (batch, 1) 当前步输入 token
        :param encoder_padded_seqs: (batch, seq_e, dim) encoder 输出
        :param encoder_mask: (batch, 1, seq_e) 预计算的 encoder 有效位置 mask
        :param hidden_states: 上一步隐状态
        :return: logits (batch, 1, voc), hidden_states, attention_weights
        """
        embedded = self._embedding(input_token)                 # (batch, 1, dim)
        embedded, hidden_states = self._rnn(embedded, hidden_states)
        # input_token 是否为 0(padding) 决定该步是否有效
        mask = (input_token != 0).unsqueeze(dim=-1).float()     # (batch, 1, 1)
        attn_out, attention_weights = self._attention.forward_single_step(
            embedded, encoder_padded_seqs, encoder_mask, mask
        )
        logits = self._linear(attn_out) * mask                  # (batch, 1, voc)
        return logits, hidden_states, attention_weights

    def get_params(self) -> dict:
        """Obtains the params for the network.

        :returns: A dict with the params.
        """

        parameter_enum = ModelParametersEnum

        return {
            parameter_enum.NUMBER_OF_LAYERS: self.num_layers,
            parameter_enum.NUMBER_OF_DIMENSIONS: self.num_dimensions,
            parameter_enum.VOCABULARY_SIZE: self.vocabulary_size,
            parameter_enum.DROPOUT: self.dropout,
        }
