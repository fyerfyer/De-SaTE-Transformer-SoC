import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pywt
from typing import Optional

class DenoisingAutoEncoder(nn.Module):
    """
    去噪自编码器模块，用于不同类型的噪声处理
    """
    def __init__(self, input_size: int, hidden_size: int = None, noise_type: str = 'gaussian', noise_level: float = 0.01):
        super(DenoisingAutoEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size or input_size // 2
        self.noise_type = noise_type
        self.noise_level = noise_level
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size, input_size),
            nn.Tanh()  # 输出归一化
        )
    
    def add_noise(self, x):
        """添加不同类型的噪声"""
        if not self.training or self.noise_level == 0:
            return x
        
        if self.noise_type == 'gaussian':
            noise = self.noise_level * torch.randn_like(x)
            return x + noise
        elif self.noise_type == 'poisson':
            # 泊松噪声
            rate = torch.abs(x) / (self.noise_level + 1e-6)
            poisson_noise = torch.poisson(rate) * (self.noise_level + 1e-6)
            return x + poisson_noise
        elif self.noise_type == 'speckle':
            # 斑点噪声
            noise = self.noise_level * torch.randn_like(x)
            return x * (1 + noise)
        elif self.noise_type == 'uniform':
            # 均匀噪声
            noise = self.noise_level * (torch.rand_like(x) - 0.5)
            return x + noise
        else:
            return x
    
    def forward(self, x):
        # 添加噪声
        x_noisy = self.add_noise(x)
        
        # 编码
        encoded = self.encoder(x_noisy)
        
        # 解码
        decoded = self.decoder(encoded)
        
        return encoded, decoded

class WaveletDenoiser(nn.Module):
    """
    小波去噪模块
    """
    def __init__(self, wavelet='db4', mode='soft', threshold=0.01):
        super(WaveletDenoiser, self).__init__()
        self.wavelet = wavelet
        self.mode = mode
        self.threshold = threshold
    
    def forward(self, x):
        """
        对输入进行小波去噪
        x: shape (batch_size, seq_len, features)
        """
        batch_size, seq_len, features = x.shape
        x_denoised = torch.zeros_like(x)
        
        # 对每个特征维度分别进行小波去噪
        for b in range(batch_size):
            for f in range(features):
                signal = x[b, :, f].detach().cpu().numpy()
                
                # 小波变换
                coeffs = pywt.wavedec(signal, self.wavelet)
                
                # 阈值处理
                if self.mode == 'soft':
                    coeffs_thresh = [pywt.threshold(c, self.threshold, 'soft') for c in coeffs]
                elif self.mode == 'hard':
                    coeffs_thresh = [pywt.threshold(c, self.threshold, 'hard') for c in coeffs]
                else:  # garrote
                    coeffs_thresh = []
                    for c in coeffs:
                        thresh_c = c.copy()
                        mask = np.abs(c) > self.threshold
                        thresh_c[mask] = c[mask] - np.sign(c[mask]) * self.threshold**2 / np.abs(c[mask])
                        thresh_c[~mask] = 0
                        coeffs_thresh.append(thresh_c)
                
                # 小波逆变换
                signal_denoised = pywt.waverec(coeffs_thresh, self.wavelet)
                
                # 确保长度一致
                if len(signal_denoised) != seq_len:
                    signal_denoised = signal_denoised[:seq_len]
                
                x_denoised[b, :, f] = torch.tensor(signal_denoised, dtype=x.dtype, device=x.device)
        
        return x_denoised

class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力机制
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        # 计算Q, K, V
        Q = self.q_linear(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        output = self.out(context)
        return output

class PositionalEncoding(nn.Module):
    """
    位置编码
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class DeSaTETransformerBranch(nn.Module):
    """
    De-SaTE Transformer的单个分支
    每个分支对应一种去噪方法
    """
    def __init__(self, input_size, d_model=128, n_heads=8, n_layers=3, 
                 denoiser_type='autoencoder', noise_type='gaussian', noise_level=0.01):
        super(DeSaTETransformerBranch, self).__init__()
        
        self.denoiser_type = denoiser_type
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 去噪模块
        if denoiser_type == 'autoencoder':
            self.denoiser = DenoisingAutoEncoder(
                input_size=input_size, 
                hidden_size=d_model//2,
                noise_type=noise_type, 
                noise_level=noise_level
            )
            self.feature_projection = nn.Linear(d_model//2, d_model)
        elif denoiser_type == 'wavelet':
            self.denoiser = WaveletDenoiser()
            self.feature_projection = nn.Linear(input_size, d_model)
        else:
            # 无去噪
            self.denoiser = None
            self.feature_projection = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer编码器层
        self.attention_layers = nn.ModuleList([
            MultiHeadSelfAttention(d_model, n_heads) for _ in range(n_layers)
        ])
        
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        
        self.feed_forward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(0.1)
            ) for _ in range(n_layers)
        ])
        
        # 输出层
        self.output_projection = nn.Linear(d_model, 1)
        
        # dropout
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        """
        x: shape (batch_size, seq_len, input_size)
        """
        batch_size, seq_len, input_size = x.shape
        
        # 去噪处理
        reconstruction_loss = 0
        if self.denoiser is not None:
            if self.denoiser_type == 'autoencoder':
                # 重塑输入以适应自编码器
                x_flat = x.reshape(-1, input_size)
                encoded, decoded = self.denoiser(x_flat)
                
                # 计算重构损失
                reconstruction_loss = F.mse_loss(decoded, x_flat)
                
                # 使用编码特征
                features = encoded.reshape(batch_size, seq_len, -1)
                features = self.feature_projection(features)
            elif self.denoiser_type == 'wavelet':
                # 小波去噪
                x_denoised = self.denoiser(x)
                features = self.feature_projection(x_denoised)
                reconstruction_loss = F.mse_loss(x_denoised, x)
        else:
            features = self.feature_projection(x)
        
        # 位置编码
        features = features.permute(1, 0, 2)  # (seq_len, batch_size, d_model)
        features = self.pos_encoding(features)
        features = features.permute(1, 0, 2)  # (batch_size, seq_len, d_model)
        
        # Transformer编码器层
        for attention, norm1, ff, norm2 in zip(
            self.attention_layers, self.norm_layers, 
            self.feed_forward_layers, self.norm_layers
        ):
            # 自注意力 + 残差连接
            attn_out = attention(features)
            features = norm1(features + self.dropout(attn_out))
            
            # 前馈网络 + 残差连接  
            ff_out = ff(features)
            features = norm2(features + self.dropout(ff_out))
        
        # 输出预测（使用最后一个时间步）
        output = self.output_projection(features[:, -1, :])  # (batch_size, 1)
        
        return output, reconstruction_loss

class DeSaTETransformer(nn.Module):
    """
    De-SaTE Transformer主模型
    包含多个分支，每个分支对应不同的去噪方法
    """
    def __init__(self, input_size, d_model=128, n_heads=8, n_layers=3, 
                 alpha=1e-4, branch_configs=None):
        super(DeSaTETransformer, self).__init__()
        
        self.alpha = alpha  # 重构损失权重
        
        # 默认分支配置
        if branch_configs is None:
            branch_configs = [
                {'denoiser_type': 'autoencoder', 'noise_type': 'gaussian', 'noise_level': 0.01},
                {'denoiser_type': 'autoencoder', 'noise_type': 'poisson', 'noise_level': 0.01},
                {'denoiser_type': 'autoencoder', 'noise_type': 'speckle', 'noise_level': 0.01},
                {'denoiser_type': 'wavelet', 'noise_type': None, 'noise_level': 0},
            ]
        
        # 创建多个分支
        self.branches = nn.ModuleList([
            DeSaTETransformerBranch(
                input_size=input_size,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                **config
            ) for config in branch_configs
        ])
        
        self.num_branches = len(self.branches)
    
    def forward(self, x):
        """
        x: shape (batch_size, seq_len, input_size)
        """
        branch_outputs = []
        total_reconstruction_loss = 0
        
        # 每个分支的前向传播
        for branch in self.branches:
            output, recon_loss = branch(x)
            branch_outputs.append(output)
            total_reconstruction_loss += recon_loss
        
        # 将分支输出堆叠
        branch_outputs = torch.stack(branch_outputs, dim=2)  # (batch_size, 1, num_branches)
        
        return branch_outputs, total_reconstruction_loss
    
    def get_best_prediction(self, x, true_values=None):
        """
        获取最佳分支的预测结果
        如果提供了真实值，根据误差选择最佳分支
        否则返回所有分支的平均值
        """
        branch_outputs, recon_loss = self.forward(x)
        
        if true_values is not None and self.training:
            # 计算每个分支的误差
            branch_errors = []
            for i in range(self.num_branches):
                pred = branch_outputs[:, :, i]
                error = F.mse_loss(pred, true_values)
                branch_errors.append(error)
            
            # 选择误差最小的分支
            best_branch_idx = torch.argmin(torch.stack(branch_errors))
            best_prediction = branch_outputs[:, :, best_branch_idx]
            
            return best_prediction, recon_loss, best_branch_idx
        else:
            # 返回所有分支的平均值
            avg_prediction = torch.mean(branch_outputs, dim=2)
            return avg_prediction, recon_loss, None
    
    def compute_loss(self, predictions, targets, reconstruction_loss):
        """
        计算总损失：预测损失 + 重构损失
        """
        prediction_loss = F.mse_loss(predictions, targets)
        total_loss = prediction_loss + self.alpha * reconstruction_loss
        return total_loss, prediction_loss, reconstruction_loss
    
# Previous Note:   
# --- 2. Build Model ---
# The input feature size to the transformer is 1 (capacity), but we project it.
# The notebook code is a bit confusing here. It seems feature_size is d_model.
# Let's add an input layer to project our single feature to `feature_size`.
# No, let's re-read the notebook code. `build_sequences` creates sequences of shape (len, window_size).
# The model seems to take `(window_size, batch_size, 1)` and the linear layer at the end maps `feature_size` to 1.
# This means the transformer's d_model should be 1. This can't be right.
# Let's look at the original paper's code. Ah, the `Data-agnostic Transformer` uses a linear layer to embed the input.
# Let's stick to the notebook implementation which is simpler. It seems to directly feed the sequence in.
# A feature_size of 1 is not valid for multi-head attention.
    
# Re-reading the notebook: It appears the `train_x` is of shape (num_sequences, window_size).
# It gets reshaped to (num_sequences, window_size, 1).
# The model must have an input embedding layer. Let's add it.

# After another look at the user-provided notebook, it seems `feature_size` is just the `d_model`
# and the input is implicitly handled. But a transformer encoder needs `d_model` to be the last dim of the input.
# The notebook code is flawed. I'll correct it by adding an input projection layer.