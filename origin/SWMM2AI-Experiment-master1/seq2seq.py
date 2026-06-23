import torch
import torch.nn as nn
import numpy as np
from registry import register_model

@register_model("Seq2SeqLSTM")
class Seq2SeqLSTM(nn.Module):
    """Seq2Seq模型用于降雨-水位预测"""
    
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, 
                 output_size=1, dropout=0.3):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        # 编码器
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # 解码器
        self.decoder = nn.LSTM(
            input_size=output_size,  # 输入是上一时刻的隐藏状态
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

    
    def forward(self, src, tgt=None):
        """
        src: 输入降雨序列 (batch, src_len, 1)
        tgt_len: 要预测的水位序列长度
        teacher_forcing_ratio: 训练时用真实值的概率
        """

        batch_size = src.size(0)
        
        # 编码
        _, (hidden, cell) = self.encoder(src)
        
        # 解码器初始输入
        decoder_input = torch.zeros(batch_size, 1, self.output_size).to(src.device)
        
        # 确定解码长度
        if tgt is not None:
            # 训练模式：使用目标序列长度
            tgt_len = tgt.size(1)
        else:
            # 推理模式：使用指定的最大长度或源序列长度
            tgt_len = src.size(1)
        
        # 存储所有预测
        outputs = []
        
        for t in range(tgt_len):
            # 解码
            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            
            # 生成预测
            pred = self.fc(decoder_output)
            outputs.append(pred)
            
            # 准备下一个输入
            if tgt is not None and self.training:
                # 训练时：使用teacher forcing（这里简化，可以加入概率控制）
                decoder_input = tgt[:, t:t+1, :]
            else:
                # 推理时：使用自己的预测
                decoder_input = pred

        
        # 堆叠所有时间步
        return torch.cat(outputs, dim=1)  # (batch, tgt_len, 1)
    
