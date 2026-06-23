import torch
import torch.nn as nn
import numpy as np
from registry import register_model

@register_model("SimpleGRU")
class SimpleGRU(nn.Module):
    """GRU模型（单GRU层+全连接）"""
    
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, 
                 output_size=1, dropout=0.3):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU层
        self.gru = nn.GRU(
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
        # GRU处理
        gru_out, _ = self.gru(x)  # (batch_size, seq_length, hidden_size)
        
        # # 对每个时间步应用全连接层
        # batch_size, seq_length, hidden_size = gru_out.shape
        
        # # 重塑以便批量处理所有时间步
        # gru_out_reshaped = gru_out.reshape(-1, hidden_size)  # (batch_size * seq_length, hidden_size)
        # fc_out = self.fc(gru_out_reshaped)  # (batch_size * seq_length, output_size)
        
        # # 恢复原始形状
        # output = fc_out.reshape(batch_size, seq_length, -1)  # (batch_size, seq_length, output_size)
        output = self.fc(gru_out)
        return output
    