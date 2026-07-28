import torch
import torch.nn as nn
import numpy as np

from registry import register_model

@register_model("AttentionLSTM")
class AttentionLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, 
                 output_size=1, dropout=0.3):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # 上下文映射
        self.context_fc = nn.Linear(hidden_size, output_size)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
        
        # 用于存储注意力权重（可视化用）
        self.attention_weights = None

    def forward(self, x):
        # LSTM处理
        lstm_out, _ = self.lstm(x)
        batch_size, seq_length, hidden_size = lstm_out.shape
        
        # 注意力权重计算
        attn_scores = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_scores, dim=1)
        self.attention_weights = attn_weights.detach()
        
        # 加权平均得到上下文
        context = torch.sum(attn_weights * lstm_out, dim=1, keepdim=True)
        
        # 上下文映射
        context_output = self.context_fc(context)
        
        # 每个时间步的预测
        # lstm_out_reshaped = lstm_out.reshape(-1, hidden_size)
        # fc_out = self.fc(lstm_out_reshaped)
        # output = fc_out.reshape(batch_size, seq_length, -1)
        output = self.fc(lstm_out)

        # 上下文特征融合
        context_expanded = context_output.repeat(1, seq_length, 1)
        final_output = output + 0.1 * context_expanded
        
        return final_output
    

@register_model("CausalAttentionLSTM")
class CausalAttentionLSTM(nn.Module):
    """因果注意力LSTM - 只关注过去"""
    def __init__(self, input_size=1, hidden_size=128, num_layers=2,
                 output_size=1, dropout=0.3):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # 因果注意力层
        self.attention = nn.Linear(hidden_size, 1)
        
        # 全连接层
        self.fc = nn.Linear(hidden_size * 2, output_size)  # 2倍因为要拼接
        
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        
        # Step 1: LSTM处理
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)
        
        outputs = []
        
        # Step 2: 对每个时间步进行因果预测
        for t in range(lstm_out.shape[1]):
            # 只使用当前和之前的时间步
            current_and_past = lstm_out[:, :t+1, :]  # (batch, t+1, hidden)
            
            # 计算注意力权重（只关注过去）
            attn_scores = self.attention(current_and_past)  # (batch, t+1, 1)
            attn_weights = torch.softmax(attn_scores, dim=1)  # 归一化
            
            # 加权平均（上下文向量）
            context = torch.sum(attn_weights * current_and_past, dim=1, keepdim=True)  # (batch, 1, hidden)
            
            # 当前时间步的LSTM输出
            current = lstm_out[:, t:t+1, :]  # (batch, 1, hidden)
            
            # 拼接当前信息和上下文
            combined = torch.cat([current, context], dim=2)  # (batch, 1, hidden*2)
            
            # 预测
            pred = self.fc(combined)  # (batch, 1, output)
            outputs.append(pred)
        
        # 堆叠所有时间步
        return torch.cat(outputs, dim=1)


# ============================================================
# Random Mask Attention LSTM — 对照实验
# 使用固定的随机掩码（既非双向也非因果），
# 用于排除"注意力惩罚源于过拟合/参数量"的替代假设。
# 若随机掩码性能介于双向和因果之间，则证明因果性本身是关键。
# ============================================================
@register_model("RandomMaskAttentionLSTM")
class RandomMaskAttentionLSTM(nn.Module):
    """随机掩码注意力LSTM — 注意力惩罚归因对照实验"""
    def __init__(self, input_size=1, hidden_size=128, num_layers=2,
                 output_size=1, dropout=0.3, mask_ratio=0.5, mask_seed=2024):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.mask_ratio = mask_ratio
        self.mask_seed = mask_seed

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # 因果注意力层 (与CausalAttentionLSTM结构相同)
        self.attention = nn.Linear(hidden_size, 1)

        # 全连接层
        self.fc = nn.Linear(hidden_size * 2, output_size)

        # 预生成固定随机掩码 (在forward中按需裁剪)
        self._mask_rng = np.random.RandomState(mask_seed)

    def _get_random_mask(self, seq_len, device):
        """生成固定的随机注意力掩码 (每次调用相同seed → 相同掩码)"""
        rng = np.random.RandomState(self.mask_seed)
        # 生成 seq_len x seq_len 的随机掩码
        mask = rng.random((seq_len, seq_len)) < self.mask_ratio
        # 确保对角线可见（当前时间步总能看到自己）
        np.fill_diagonal(mask, True)
        # 转为 float mask: True=可见, False=被遮蔽
        mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device)
        return mask_tensor  # (seq_len, seq_len)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)
        batch_size, seq_len, hidden = lstm_out.shape

        # 获取随机掩码
        mask = self._get_random_mask(seq_len, x.device)  # (seq_len, seq_len)

        outputs = []
        for t in range(seq_len):
            # 对时间步t，获取可见位置
            visible_mask = mask[t]  # (seq_len,) — 1.0=可见, 0.0=遮蔽

            # 计算注意力分数
            attn_scores = self.attention(lstm_out).squeeze(-1)  # (batch, seq_len)

            # 应用掩码: 被遮蔽位置设为-inf
            masked_scores = attn_scores.masked_fill(
                visible_mask.unsqueeze(0) == 0, float('-inf')
            )
            attn_weights = torch.softmax(masked_scores, dim=1)  # (batch, seq_len)

            # 加权平均
            context = torch.bmm(
                attn_weights.unsqueeze(1), lstm_out
            )  # (batch, 1, hidden)

            # 当前时间步
            current = lstm_out[:, t:t+1, :]  # (batch, 1, hidden)

            # 拼接
            combined = torch.cat([current, context], dim=2)  # (batch, 1, hidden*2)
            pred = self.fc(combined)
            outputs.append(pred)

        return torch.cat(outputs, dim=1)


# ============================================================
# PCCA-LSTM (Physically Consistent Causal Attention LSTM)
# 架构与 CausalAttentionLSTM 完全相同，
# 区别仅在于训练时使用 PhysicallyConsistentLoss（见 physics_loss.py）
# ============================================================
@register_model("PCCA-LSTM")
class PCCALSTM(CausalAttentionLSTM):
    """
    Physically Consistent Causal Attention LSTM

    与 CausalAttentionLSTM 架构完全相同，仅训练时使用物理一致性损失函数：
    L = MSE + lambda_smooth * SmoothLoss + lambda_peak * PeakTimeLoss

    注意：SmoothLoss 和 PeakTimeLoss 是时序正则化约束（Temporal Regularization），
    而非物理方程约束（如圣维南方程残差）。它们基于数据可观测的、物理上合理的
    时序特征（平滑性、峰值时滞）来构造正则项，属于"物理先验软约束"范式。

    SmoothLoss:  惩罚预测序列的二阶差分（抑制非物理高频波动）
    PeakTimeLoss: 惩罚预测峰值时刻与真实峰值时刻的偏差（可微分soft-argmax）
    """
    pass