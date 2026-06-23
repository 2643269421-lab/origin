import torch
import torch.nn as nn
import numpy as np

from registry import register_model

@register_model("ResidualLSTM")
class ResidualLSTM(nn.Module):
    """带残差连接的LSTM"""
    
    def __init__(self, input_size=1, hidden_size=128, num_layers=2,
                 output_size=1, dropout=0.3):
        super().__init__()
        
        self.hidden_size = hidden_size

         # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 残差连接适配层（可选）
        self.residual_adapter = None
        if input_size != hidden_size:
            self.residual_adapter = nn.Linear(input_size, hidden_size)
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        # LSTM处理
        lstm_out, _ = self.lstm(x)
        
        # 残差连接
        if self.residual_adapter is not None:
            # 调整输入维度后相加
            residual = self.residual_adapter(x)
            lstm_out = lstm_out + residual
        else:
            # 直接相加
            lstm_out = lstm_out + x
        
        # 输出
        output = self.fc(lstm_out)
        return output