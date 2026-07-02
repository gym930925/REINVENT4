import torch
from torch import nn as tnn
from torch.nn.utils import rnn as tnnur

from reinvent.models.model_parameter_enum import ModelParametersEnum


class Encoder(tnn.Module):
    """
    Simple bidirectional RNN encoder implementation.
    """

    def __init__(self, num_layers: int, num_dimensions: int, vocabulary_size: int, dropout: float):
        super(Encoder, self).__init__()

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
            bidirectional=True,
        )

    def forward(self, padded_seqs: torch.Tensor, seq_lengths: torch.Tensor) -> (
        torch.Tensor,
        (torch.Tensor, torch.Tensor),
    ):  # pylint: disable=arguments-differ
        """
        Performs the forward pass.
        :param padded_seqs: A tensor with the sequences (batch, seq).
        :param seq_lengths: The lengths of the sequences (for packed sequences).
        :return : A tensor with all the output values for each step and the two hidden states.
        """
        # P0: .size() -> .shape 避免部分后端的 _local_scalar_dense 同步
        batch_size = padded_seqs.shape[0]
        max_seq_size = padded_seqs.shape[1]

        # P0: 直接在目标设备创建隐藏状态，避免 .to() 跨设备拷贝
        hidden_state = self._initialize_hidden_state(batch_size, padded_seqs.device)
        
        padded_seqs = self._embedding(padded_seqs)
        hs_h, hs_c = (hidden_state, hidden_state.clone().detach())
        
        # === 修改：直接使用 padded input，不使用 pack_packed_sequence ===
        # NPU bidirectional LSTM 不支持 PackedSequence
        
        # 创建 mask 处理 padding 位置
        seq_range = torch.arange(max_seq_size, device=padded_seqs.device)
        mask = (seq_range.unsqueeze(0) < seq_lengths.unsqueeze(1).to(padded_seqs.device))
        mask = mask.unsqueeze(-1).type_as(padded_seqs)
        
        # 直接调用 LSTM（使用 padded input）
        output, (hs_h, hs_c) = self._rnn(padded_seqs, (hs_h, hs_c))
        
        # 应用 mask（将 padding 位置置零）
        output = output * mask
        
        # === 手动处理 bidirectional 输出 ===
        # sum up bidirectional layers and collapse
        hs_h = hs_h.view(self.num_layers, 2, batch_size, self.num_dimensions).sum(dim=1)
        hs_c = hs_c.view(self.num_layers, 2, batch_size, self.num_dimensions).sum(dim=1)
        
        # 手动合并 bidirectional output
        output = output.view(batch_size, max_seq_size, 2, self.num_dimensions).sum(dim=2)
        
        return output, (hs_h, hs_c)


    def _initialize_hidden_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # P0: 显式 device 参数，避免默认 CPU 创建后跨设备拷贝
        return torch.zeros(self.num_layers * 2, batch_size, self.num_dimensions, device=device)

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
