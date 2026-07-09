"""Modules and code used in the core part of PE (Protein Engineering) transformer model.

This module implements a transformer-based neural network architecture specifically designed
for protein sequence design tasks. The transformer model processes input features representing
protein structural information and outputs predictions for optimal amino acid sequences.
"""
from torch import nn

class Transformer(nn.Module):
    """Transformer model for protein sequence design.
    
    This class implements a transformer encoder architecture that takes in protein structural
    features and outputs sequence predictions. The model consists of:
    - An input projection layer
    - A stack of transformer encoder layers
    - An output projection layer
    
    Args:
        d_input (int): Dimensionality of input features
        d_output (int): Dimensionality of output predictions
        d_model (int): Hidden dimension size of transformer (default: 256)
        nhead (int): Number of attention heads in multi-head attention (default: 4)
        nlayer (int): Number of transformer encoder layers (default: 3)
        **kw: Additional keyword arguments passed to layer initialization
    """

    def __init__(self, d_input, d_output, d_model=256, nhead=4, nlayer=3, **kw):
        super().__init__()
        
        # Create a single transformer encoder layer with specified dimensions
        # This layer includes self-attention and feed-forward components
        layer = nn.TransformerEncoderLayer(
            d_model,  # Hidden dimension size
            nhead,    # Number of attention heads
            batch_first=True,  # Input format: (batch, sequence, features)
            **kw
        )
        
        # Input projection layer: maps input features to transformer dimension
        self.input = nn.Linear(d_input, d_model, **kw)
        
        # Main transformer encoder: stack of multiple identical layers
        # enable_nested_tensor=False: 关闭 nested tensor 优化路径。
        # 原因：NPU 不支持 aten::_nested_tensor_from_mask_left_aligned 算子，
        # 开启时会 fallback 到 CPU 执行，影响性能。关闭后语义不变，
        # 但需在 forward 中手动将 padding 位置输出置零以保持 mean 行为一致。
        self.main = nn.TransformerEncoder(
            layer, num_layers=nlayer, enable_nested_tensor=False)
        
        # Output projection layer: maps transformer outputs to final predictions
        self.output = nn.Linear(d_model, d_output, **kw)

    def forward(self, feat, mask):
        """Forward pass through the transformer model.
        
        Args:
            feat: Input features tensor of shape (batch_size, sequence_length, feature_dim)
            mask: Boolean mask tensor of shape (batch_size, sequence_length) indicating
                  which positions should be attended to (True = attend, False = ignore)
        
        Returns:
            Output predictions tensor of shape (batch_size, output_dim)
        """
        # Project input features to transformer dimension
        feat = self.input(feat)
        
        # Process features through transformer encoder
        # Note: ~mask inverts the mask since PyTorch expects True = ignore
        logits = self.main(feat, src_key_padding_mask=~mask)

        # 关闭 nested tensor 后，padding 位置输出不再被自动置零。
        # 这里手动置零以匹配原 enable_nested_tensor=True 的 mean 行为：
        #   sum(valid_outputs) / L  （除以总长度 L，而非有效长度 valid_count）
        # mask: (B, L) True=有效 -> 扩展为 (B, L, 1) 广播到 feature 维度
        logits = logits * mask.unsqueeze(-1).to(logits.dtype)

        # Average over sequence dimension to get global representation
        logits = logits.mean(dim=1)
        
        # Project to final output dimension
        return self.output(logits)
