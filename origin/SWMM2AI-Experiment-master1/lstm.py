import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from registry import register_model

@register_model("SimpleLSTM")
class SimpleLSTM(nn.Module):
    """LSTM模型（单LSTM层+全连接）"""
    
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
        
        # 全连接层（应用到每个时间步）
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入降雨序列，形状 (batch_size, seq_length, input_size)
            
        返回:
            预测的水位序列，形状 (batch_size, seq_length, output_size)
        """
        # LSTM处理
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_length, hidden_size)
        
        # # 对每个时间步应用全连接层
        # batch_size, seq_length, hidden_size = lstm_out.shape
        
        # # 重塑以便批量处理所有时间步
        # lstm_out_reshaped = lstm_out.reshape(-1, hidden_size)  # (batch_size * seq_length, hidden_size)
        # fc_out = self.fc(lstm_out_reshaped)  # (batch_size * seq_length, output_size)
        
        # # 恢复原始形状
        # output = fc_out.reshape(batch_size, seq_length, -1)  # (batch_size, seq_length, output_size)
        output = self.fc(lstm_out)
        return output
    
@register_model("MultiScaleLSTM")
class MultiScaleLSTM(nn.Module):
    """多尺度特征融合的LSTM"""
    
    def __init__(self, input_size=1, hidden_size=128, num_layers=2,
                 output_size=1, dropout=0.3):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # 主要LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 不同窗口大小的卷积来提取多尺度特征
        self.conv_short = nn.Conv1d(hidden_size, hidden_size // 2, kernel_size=3, padding=1)
        self.conv_mid = nn.Conv1d(hidden_size, hidden_size // 2, kernel_size=5, padding=2)
        self.conv_long = nn.Conv1d(hidden_size, hidden_size // 2, kernel_size=7, padding=3)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size // 2 * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        # LSTM处理
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden)
        
        # 转换为卷积需要的格式 (batch, channels, seq)
        lstm_out_t = lstm_out.transpose(1, 2)  # (batch, hidden, seq)
        
        # 不同尺度的特征提取
        short_feat = F.relu(self.conv_short(lstm_out_t))  # (batch, hidden//2, seq)
        mid_feat = F.relu(self.conv_mid(lstm_out_t))  # (batch, hidden//2, seq)
        long_feat = F.relu(self.conv_long(lstm_out_t))  # (batch, hidden//2, seq)
        
        # 拼接特征
        combined = torch.cat([short_feat, mid_feat, long_feat], dim=1)  # (batch, hidden//2*3, seq)
        
        # 转回原始格式
        combined = combined.transpose(1, 2)  # (batch, seq, hidden//2*3)
        
        # 为每个时间步生成预测
        output = self.fusion(combined)  # (batch, seq, output_size)
        
        return output