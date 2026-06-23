import torch
import torch.nn as nn

from registry import register_model

@register_model("SimpleMLP")
class SimpleMLP(nn.Module):
    """LSTM模型（单LSTM层+全连接）"""
    
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, 
                 output_size=1, dropout=0.3):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        
        # 全连接层（应用到每个时间步）
        self.layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2, output_size)
        )
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入降雨序列，形状 (batch_size, seq_length, input_size)
            
        返回:
            预测的水位序列，形状 (batch_size, seq_length, output_size)
        """
       
        # 全连接
        output = self.layer(x)
        return output