import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from .SubLayers import PositionwiseFeedForward


class FourierTransform(nn.Module):
    """傅里叶变换层，替代多头自注意力机制"""
    
    def __init__(self, norm=None):
        super(FourierTransform, self).__init__()
        self.norm = norm

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x, mask=None):
        """
        应用二维傅里叶变换到输入张量
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, hidden_dim]
            mask: 掩码，在傅里叶变换中不使用，但保留参数以保持接口一致
            
        Returns:
            output: 傅里叶变换的实部，形状为 [batch_size, seq_len, hidden_dim]
            None: 为了与原MultiHeadAttention保持接口一致，返回None代替注意力权重
        """
        # 保存原始形状
        batch_size, seq_len, hidden_dim = x.shape
        
        # 应用二维FFT
        # 注意：torch.fft.fftn需要PyTorch 1.7+
        x_fft = torch.fft.fftn(x.float(), dim=(-2, -1), norm=self.norm)        
        # 只保留实部
        output = x_fft.real
        
        return output, None


class FourierBlock(nn.Module):
    """Fourier Block - 使用傅里叶变换替代自注意力机制的FFT Block"""
    
    def __init__(self, d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=0.1):
        """
        初始化FourierBlock
        
        Args:
            d_model: 模型维度
            n_head, d_k, d_v: 为了与FFTBlock保持接口一致，但在FourierBlock中不使用
            d_inner: 前馈网络内部维度
            kernel_size: 前馈网络卷积核大小
            dropout: dropout比率
        """
        super(FourierBlock, self).__init__()
        # 使用傅里叶变换替代多头自注意力
        self.fourier = FourierTransform(norm=None)
        # 保留原有的前馈网络
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, kernel_size, dropout=dropout
        )
    
    def forward(self, enc_input, mask=None, slf_attn_mask=None):
        """
        前向传播
        
        Args:
            enc_input: 输入张量
            mask: 填充掩码
            slf_attn_mask: 自注意力掩码(在傅里叶变换中不使用，但保留参数以保持接口一致)
            
        Returns:
            enc_output: 处理后的输出
            fourier_attn: None (为了与原FFTBlock保持接口一致)
        """
        # 应用傅里叶变换
        enc_output, fourier_attn = self.fourier(enc_input, mask=slf_attn_mask)
        
        # 应用掩码
        if mask is not None:
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)
        
        # 应用前馈网络
        enc_output = self.pos_ffn(enc_output)
        
        # 再次应用掩码
        if mask is not None:
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)
        
        return enc_output, fourier_attn 