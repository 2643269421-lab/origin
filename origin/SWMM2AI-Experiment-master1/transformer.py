import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import math

class PositionalEncoding(nn.Module):
    """Transformer位置编码"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]
    
class Transformer(nn.Module):
    """专门用于降雨-水位预测的因果Transformer"""
    
    def __init__(self, input_size=1, d_model=128, nhead=8, 
                 num_layers=4, output_size=1, dropout=0.1, max_len=288):
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        
        # 输入投影
        self.input_proj = nn.Linear(input_size, d_model)
        
        # 位置编码（使用可学习的位置编码更简单）
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        # 因果Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x, is_causal=True):
        """
        x: (batch_size, seq_len, input_size)
        is_causal: 是否使用因果掩码
        """
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # 输入投影
        x_proj = self.input_proj(x)  # (batch_size, seq_len, d_model)
        
        # 位置编码
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        positions = positions.expand(batch_size, seq_len)
        pos_emb = self.pos_embedding(positions)
        x_proj = x_proj + pos_emb
        
        # 因果掩码
        attn_mask = None
        if is_causal:
            # 创建因果掩码：只能看到当前及之前的时间步
            attn_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device) * float('-inf'),
                diagonal=1
            )
        
        # Transformer处理
        transformer_out = self.transformer(x_proj, mask=attn_mask)
        
        # 输出
        predictions = self.output_layer(transformer_out)
        
        return predictions
    

class TransformerPredictor:
    """水位预测器类"""
    
    def __init__(self, model_path='transformer_model.pth', output_dir='output', device=None):
        """
        初始化预测器
        
        参数:
            model_path: 模型文件路径
            device: 计算设备（自动检测）
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # 加载模型
        self.model, self.scalers, self.model_params = self._load_model(model_path)
        
        print(f"模型加载成功，使用设备: {self.device}")
        print(f"序列长度: {self.model_params['seq_length']}")
        print(f"输入特征: {self.model_params['input_size']}")
    
    def _load_model(self, model_path):
        """加载保存的模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载检查点
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # 重建模型
        # from lstm import AttentionLSTM
        
        model = Transformer(
            input_size=checkpoint['input_size'],
            output_size=1,  # 水位输出总是1维
            dropout=0.0  # 预测时不需要dropout
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()  # 设置为评估模式
        
        # 获取模型参数
        model_params = {
            'seq_length': checkpoint['seq_length'],
            'input_size': checkpoint['input_size']
        }
        
        # 获取标准化器
        scalers = {
            'rain': checkpoint['rain_scaler'],
            'water': checkpoint['water_scaler']
        }
        
        return model, scalers, model_params
    
    def predict(self, rainfall_sequence):
        """
        预测水位序列
        
        参数:
            rainfall_sequence: 降雨序列，形状应为 (seq_length,) 或 (seq_length, 1)
            
        返回:
            预测的水位序列，形状 (seq_length,)
        """
        seq_length = self.model_params['seq_length']
        
        # 检查输入长度
        if len(rainfall_sequence) != seq_length:
            raise ValueError(
                f"输入序列长度应为 {seq_length}，但实际为 {len(rainfall_sequence)}。"
                f"请使用长度为 {seq_length} 的序列或进行插值。"
            )
        
        # 转换为numpy数组并确保形状正确
        rainfall_array = np.array(rainfall_sequence, dtype=np.float32)
        
        if rainfall_array.ndim == 1:
            rainfall_array = rainfall_array.reshape(-1, 1)
        
        # 标准化降雨数据
        rainfall_scaled = self.scalers['rain'].transform(rainfall_array)
        
        # 转换为张量并添加批次维度
        rainfall_tensor = torch.FloatTensor(rainfall_scaled).unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            water_scaled_tensor = self.model(rainfall_tensor)
        
        # 转换为numpy并移除批次维度
        water_scaled = water_scaled_tensor.squeeze(0).cpu().numpy()
        
        # 反标准化
        water_predicted = self.scalers['water'].inverse_transform(water_scaled)
        
        return water_predicted.flatten()
    
    def predict_batch(self, rainfall_sequences):
        """
        批量预测
        
        参数:
            rainfall_sequences: 降雨序列列表或数组，形状 (batch_size, seq_length)
            
        返回:
            预测的水位序列数组，形状 (batch_size, seq_length)
        """
        batch_size = len(rainfall_sequences)
        seq_length = self.model_params['seq_length']
        
        # 检查输入形状
        if isinstance(rainfall_sequences, list):
            rainfall_sequences = np.array(rainfall_sequences)
        
        if rainfall_sequences.shape[1] != seq_length:
            raise ValueError(
                f"每个序列长度应为 {seq_length}，但实际为 {rainfall_sequences.shape[1]}"
            )
        
        # 重塑为 (batch_size, seq_length, 1)
        if rainfall_sequences.ndim == 2:
            rainfall_sequences = rainfall_sequences.reshape(batch_size, seq_length, 1)
        
        # 标准化
        original_shape = rainfall_sequences.shape
        rainfall_2d = rainfall_sequences.reshape(-1, 1)
        rainfall_scaled_2d = self.scalers['rain'].transform(rainfall_2d)
        rainfall_scaled = rainfall_scaled_2d.reshape(original_shape)
        
        # 转换为张量
        rainfall_tensor = torch.FloatTensor(rainfall_scaled).to(self.device)
        
        # 预测
        with torch.no_grad():
            water_scaled_tensor = self.model(rainfall_tensor)
        
        # 转换为numpy
        water_scaled = water_scaled_tensor.cpu().numpy()
        
        # 反标准化
        water_predicted = self.scalers['water'].inverse_transform(
            water_scaled.reshape(-1, 1)
        ).reshape(batch_size, seq_length)
        
        return water_predicted
    
    def visualize_prediction(self, rainfall_sequence, water_predicted=None, 
                            title=None, time_step_min=5):
        """
        可视化预测结果
        
        参数:
            rainfall_sequence: 降雨序列
            water_predicted: 预测的水位序列，如果为None则自动预测
            title: 图表标题
            time_step_min: 时间步长（分钟）
        """
        if water_predicted is None:
            water_predicted = self.predict(rainfall_sequence)
        
        seq_length = len(rainfall_sequence)
        time_hours = np.arange(seq_length) * time_step_min / 60
        
        # 创建图表
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # 降雨图
        axes[0].bar(time_hours, rainfall_sequence, width=time_step_min/60/1.5, 
                   alpha=0.7, color='blue', edgecolor='darkblue')
        axes[0].set_xlabel('时间 (小时)')
        axes[0].set_ylabel('降雨强度 (mm/h)', color='blue')
        axes[0].tick_params(axis='y', labelcolor='blue')
        axes[0].set_title('输入降雨序列')
        axes[0].grid(True, alpha=0.3)
        
        # 水位图
        axes[1].plot(time_hours, water_predicted, 'r-', linewidth=2, label='预测水位')
        axes[1].fill_between(time_hours, 0, water_predicted, alpha=0.3, color='red')
        axes[1].set_xlabel('时间 (小时)')
        axes[1].set_ylabel('水位 (m)', color='red')
        axes[1].tick_params(axis='y', labelcolor='red')
        
        if title:
            axes[1].set_title(title)
        else:
            axes[1].set_title('LSTM预测的水位序列')
        
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 添加统计信息
        stats_text = (f"最大降雨: {rainfall_sequence.max():.1f} mm/h\n"
                     f"总降雨量: {rainfall_sequence.sum() * time_step_min/60:.1f} mm\n"
                     f"最大水位: {water_predicted.max():.3f} m\n"
                     f"平均水位: {water_predicted.mean():.3f} m")
        
        axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes,
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def visualize_prediction_with_swmm(self, rainfall_sequence, water_predicted=None, 
                                   swmm_water_sequence=None, title=None, 
                                   time_step_min=5, swmm_simulator=None):
        """
        可视化预测结果，并与SWMM模拟结果对比
        
        参数:
            rainfall_sequence: 降雨序列
            water_predicted: 预测的水位序列，如果为None则自动预测
            swmm_water_sequence: SWMM模拟的水位序列，如果为None且提供了swmm_simulator则自动模拟
            title: 图表标题
            time_step_min: 时间步长（分钟）
            swmm_simulator: SWMM模拟器实例，用于生成SWMM模拟结果
            
        返回:
            matplotlib图表对象
        """
        if water_predicted is None:
            water_predicted = self.predict(rainfall_sequence)
        
        # 如果提供了SWMM模拟器但没有提供SWMM水位序列，则进行SWMM模拟
        if swmm_water_sequence is None and swmm_simulator is not None:
            print("正在运行SWMM模拟进行对比...")
            try:
                swmm_results = swmm_simulator.run_swmm_simulation(
                    rainfall_mm_h=rainfall_sequence
                )
                swmm_water_sequence = swmm_results['values']
                
                # 确保SWMM结果长度与预测结果一致
                if len(swmm_water_sequence) != len(water_predicted):
                    print(f"警告: SWMM结果长度({len(swmm_water_sequence)})与预测长度({len(water_predicted)})不一致，进行插值")
                    from scipy import interpolate
                    x_old = np.linspace(0, 1, len(swmm_water_sequence))
                    x_new = np.linspace(0, 1, len(water_predicted))
                    f = interpolate.interp1d(x_old, swmm_water_sequence, kind='linear', fill_value='extrapolate')
                    swmm_water_sequence = f(x_new)
                    
            except Exception as e:
                print(f"SWMM模拟失败: {e}")
                swmm_water_sequence = None
        
        seq_length = len(rainfall_sequence)
        time_hours = np.arange(seq_length) * time_step_min / 60
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # ==================== 子图1: 降雨序列 ====================
        ax1 = axes[0, 0]
        ax1.bar(time_hours, rainfall_sequence, width=time_step_min/60/1.5, 
              alpha=0.7, color='blue', edgecolor='darkblue')
        ax1.set_xlabel('时间 (小时)')
        ax1.set_ylabel('降雨强度 (mm/h)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_title('输入降雨序列')
        ax1.grid(True, alpha=0.3)
        
        # 添加降雨统计信息
        rain_stats_text = (f"最大强度: {rainfall_sequence.max():.1f} mm/h\n"
                          f"平均强度: {rainfall_sequence.mean():.1f} mm/h\n"
                          f"总降雨量: {rainfall_sequence.sum() * time_step_min/60:.1f} mm\n"
                          f"降雨历时: {np.sum(rainfall_sequence > 0) * time_step_min/60:.1f} h")
        
        ax1.text(0.02, 0.98, rain_stats_text, transform=ax1.transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # ==================== 子图2: 水位对比 ====================
        ax2 = axes[0, 1]
        
        # 绘制预测水位
        ax2.plot(time_hours, water_predicted, 'r-', linewidth=3, label='LSTM预测水位', alpha=0.8)
        ax2.fill_between(time_hours, 0, water_predicted, alpha=0.2, color='red')
        
        # 如果提供了SWMM结果，绘制SWMM模拟水位
        if swmm_water_sequence is not None:
            ax2.plot(time_hours, swmm_water_sequence, 'b--', linewidth=2.5, 
                    label='SWMM模拟水位', alpha=0.7)
            ax2.fill_between(time_hours, water_predicted, swmm_water_sequence, 
                            alpha=0.1, color='gray', label='误差区域')
        
        ax2.set_xlabel('时间 (小时)')
        ax2.set_ylabel('水位 (m)')
        ax2.set_title('水位对比: LSTM预测 vs SWMM模拟')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # 添加水位统计信息
        water_stats_text = (f"LSTM最大水位: {water_predicted.max():.3f} m\n"
                          f"LSTM平均水位: {water_predicted.mean():.3f} m")
        
        if swmm_water_sequence is not None:
            water_stats_text += (f"\n\nSWMM最大水位: {swmm_water_sequence.max():.3f} m\n"
                              f"SWMM平均水位: {swmm_water_sequence.mean():.3f} m")
        
        ax2.text(0.02, 0.98, water_stats_text, transform=ax2.transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        
        # ==================== 子图3: 降雨与水位叠加 ====================
        ax3 = axes[1, 0]
        
        # 绘制降雨（次坐标轴）
        ax3_rain = ax3.twinx()
        bars = ax3_rain.bar(time_hours, rainfall_sequence, width=time_step_min/60/1.5,
                          alpha=0.3, color='blue', label='降雨强度')
        ax3_rain.set_ylabel('降雨强度 (mm/h)', color='blue')
        ax3_rain.tick_params(axis='y', labelcolor='blue')
        
        # 绘制水位
        ax3.plot(time_hours, water_predicted, 'r-', linewidth=2, label='LSTM预测水位')
        if swmm_water_sequence is not None:
            ax3.plot(time_hours, swmm_water_sequence, 'b--', linewidth=2, label='SWMM模拟水位')
        
        ax3.set_xlabel('时间 (小时)')
        ax3.set_ylabel('水位 (m)', color='red')
        ax3.tick_params(axis='y', labelcolor='red')
        ax3.set_title('降雨与水位响应关系')
        ax3.grid(True, alpha=0.3)
        
        # 合并图例
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_rain.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # ==================== 子图4: 误差分析 ====================
        ax4 = axes[1, 1]
        
        if swmm_water_sequence is not None:
            # 计算绝对误差
            absolute_error = np.abs(water_predicted - swmm_water_sequence)
            relative_error = np.abs(water_predicted - swmm_water_sequence) / (swmm_water_sequence + 1e-10) * 100
            
            # 绘制绝对误差
            ax4.plot(time_hours, absolute_error, 'g-', linewidth=2, label='绝对误差 (m)')
            ax4.fill_between(time_hours, 0, absolute_error, alpha=0.3, color='green')
            
            # 绘制相对误差（次坐标轴）
            ax4_rel = ax4.twinx()
            ax4_rel.plot(time_hours, relative_error, 'orange', linewidth=1.5, 
                        linestyle=':', label='相对误差 (%)', alpha=0.7)
            ax4_rel.set_ylabel('相对误差 (%)', color='orange')
            ax4_rel.tick_params(axis='y', labelcolor='orange')
            ax4_rel.set_ylim(0, max(100, relative_error.max() * 1.1))
            
            ax4.set_xlabel('时间 (小时)')
            ax4.set_ylabel('绝对误差 (m)', color='green')
            ax4.tick_params(axis='y', labelcolor='green')
            ax4.set_title('预测误差分析')
            ax4.grid(True, alpha=0.3)
            
            # 合并误差图例
            lines_err1, labels_err1 = ax4.get_legend_handles_labels()
            lines_err2, labels_err2 = ax4_rel.get_legend_handles_labels()
            ax4.legend(lines_err1 + lines_err2, labels_err1 + labels_err2, loc='upper left')
            
            # 计算并显示误差统计
            mse = np.mean((water_predicted - swmm_water_sequence) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(absolute_error)
            mape = np.mean(relative_error[swmm_water_sequence > 0])  # 避免除以0
            max_abs_error = np.max(absolute_error)
            max_rel_error = np.max(relative_error[swmm_water_sequence > 0])
            
            # R²计算
            ss_res = np.sum((swmm_water_sequence - water_predicted) ** 2)
            ss_tot = np.sum((swmm_water_sequence - np.mean(swmm_water_sequence)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-10))
            
            error_stats_text = (f"均方误差 (MSE): {mse:.4f}\n"
                              f"均方根误差 (RMSE): {rmse:.4f}\n"
                              f"平均绝对误差 (MAE): {mae:.4f}\n"
                              f"平均绝对百分比误差 (MAPE): {mape:.2f}%\n"
                              f"最大绝对误差: {max_abs_error:.4f}\n"
                              f"最大相对误差: {max_rel_error:.2f}%\n"
                              f"R²分数: {r2:.4f}")
            
            ax4.text(0.02, 0.98, error_stats_text, transform=ax4.transAxes,
                    verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            
            # 保存误差统计到文件
            self._save_error_statistics(rainfall_sequence, water_predicted, 
                                      swmm_water_sequence, mse, rmse, mae, mape, r2)
        else:
            ax4.text(0.5, 0.5, '无SWMM模拟结果进行对比', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('误差分析 (需要SWMM模拟结果)')
            ax4.grid(True, alpha=0.3)
        
        # ==================== 添加总体标题 ====================
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        else:
            fig.suptitle('LSTM水位预测与SWMM模拟对比分析', fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        figure_filename = os.path.join(self.output_dir, 'figure.png')
        plt.savefig(figure_filename)
        return fig

    def evaluate_on_dataset(self, dataset):
        """
        在数据集上评估模型
        
        参数:
            dataset: SWMMDataset实例
            
        返回:
            评估指标字典
        """
        from torch.utils.data import DataLoader
        
        # 创建数据加载器
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        device = self.device
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for rainfall, water_level in dataloader:
                rainfall = rainfall.to(device)
                water_level = water_level.to(device)
                
                predictions = self.model(rainfall)
                
                # 转换为原始值
                batch_size, seq_length, _ = predictions.shape
                
                # 重塑为2D进行反标准化
                pred_2d = predictions.reshape(-1, 1).cpu().numpy()
                target_2d = water_level.reshape(-1, 1).cpu().numpy()
                
                pred_original = self.scalers['water'].inverse_transform(pred_2d)
                target_original = self.scalers['water'].inverse_transform(target_2d)
                
                # 恢复原始形状
                pred_original = pred_original.reshape(batch_size, seq_length)
                target_original = target_original.reshape(batch_size, seq_length)
                
                all_predictions.append(pred_original)
                all_targets.append(target_original)
        
        # 合并所有批次
        predictions_all = np.concatenate(all_predictions, axis=0)
        targets_all = np.concatenate(all_targets, axis=0)
        
        # 计算评估指标
        metrics = self._calculate_metrics(predictions_all, targets_all)
        
        return metrics, predictions_all, targets_all
    
    def _calculate_metrics(self, predictions, targets):
        """计算评估指标"""
        # 确保形状一致
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        # 计算各种指标
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        mape = np.mean(np.abs((predictions - targets) / (targets + 1e-10))) * 100
        
        # R²分数
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2,
            'Max_Error': np.max(np.abs(predictions - targets)),
            'Std_Error': np.std(predictions - targets)
        }
        
        return metrics
    
    def _save_error_statistics(self, rainfall, lstm_water, swmm_water, 
                          mse, rmse, mae, mape, r2):
        """保存误差统计到文件"""
        import pandas as pd
        import json
        from datetime import datetime
        
        # 创建时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 保存详细数据到CSV
        time_minutes = np.arange(len(rainfall)) * 5
        time_str = [f"{int(t//60):02d}:{int(t%60):02d}" for t in time_minutes]
        
        df_detailed = pd.DataFrame({
            '时间': time_str,
            '降雨强度_mm_h': rainfall,
            'LSTM预测水位_m': lstm_water,
            'SWMM模拟水位_m': swmm_water,
            '绝对误差_m': np.abs(lstm_water - swmm_water),
            '相对误差_%': np.abs(lstm_water - swmm_water) / (swmm_water + 1e-10) * 100
        })
        
        detailed_filename = os.path.join(self.output_dir, 'results.csv')
        df_detailed.to_csv(detailed_filename, index=False, encoding='utf-8-sig')
        
        # 2. 保存统计摘要到JSON
        stats_summary = {
            'timestamp': timestamp,
            'data_length': len(rainfall),
            'rainfall_stats': {
                'max_intensity_mm_h': float(rainfall.max()),
                'mean_intensity_mm_h': float(rainfall.mean()),
                'total_rainfall_mm': float(rainfall.sum() * 5/60),
                'duration_hours': float(np.sum(rainfall > 0) * 5/60)
            },
            'water_stats': {
                'lstm_max_m': float(lstm_water.max()),
                'lstm_mean_m': float(lstm_water.mean()),
                'swmm_max_m': float(swmm_water.max()),
                'swmm_mean_m': float(swmm_water.mean())
            },
            'error_metrics': {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'mape_percent': float(mape),
                'r2_score': float(r2),
                'max_absolute_error_m': float(np.max(np.abs(lstm_water - swmm_water))),
                'correlation_coefficient': float(np.corrcoef(lstm_water, swmm_water)[0, 1])
            },
            'performance_assessment': self._assess_performance(rmse, mape, r2)
        }
        
        stats_filename = os.path.join(self.output_dir, 'statistics.json')
        with open(stats_filename, 'w', encoding='utf-8') as f:
            json.dump(stats_summary, f, ensure_ascii=False, indent=2)
        
        print(f"详细对比结果已保存到: {detailed_filename}")
        print(f"误差统计已保存到: {stats_filename}")

    def _assess_performance(self, rmse, mape, r2):
        """评估模型性能"""
        performance = {}
        
        # 基于RMSE评估
        if rmse < 0.05:
            performance['rmse_rating'] = '优秀'
            performance['rmse_description'] = '预测非常准确'
        elif rmse < 0.1:
            performance['rmse_rating'] = '良好'
            performance['rmse_description'] = '预测准确'
        elif rmse < 0.2:
            performance['rmse_rating'] = '一般'
            performance['rmse_description'] = '预测基本准确'
        else:
            performance['rmse_rating'] = '需要改进'
            performance['rmse_description'] = '预测误差较大'
        
        # 基于MAPE评估
        if mape < 5:
            performance['mape_rating'] = '优秀'
            performance['mape_description'] = '百分比误差很小'
        elif mape < 10:
            performance['mape_rating'] = '良好'
            performance['mape_description'] = '百分比误差较小'
        elif mape < 20:
            performance['mape_rating'] = '一般'
            performance['mape_description'] = '百分比误差可接受'
        else:
            performance['mape_rating'] = '需要改进'
            performance['mape_description'] = '百分比误差较大'
        
        # 基于R²评估
        if r2 > 0.9:
            performance['r2_rating'] = '优秀'
            performance['r2_description'] = '模型解释力很强'
        elif r2 > 0.7:
            performance['r2_rating'] = '良好'
            performance['r2_description'] = '模型解释力较好'
        elif r2 > 0.5:
            performance['r2_rating'] = '一般'
            performance['r2_description'] = '模型解释力一般'
        else:
            performance['r2_rating'] = '需要改进'
            performance['r2_description'] = '模型解释力不足'
        
        # 总体评估
        ratings = [performance['rmse_rating'], performance['mape_rating'], performance['r2_rating']]
        if all(r == '优秀' for r in ratings):
            performance['overall_rating'] = '优秀'
            performance['recommendation'] = '模型性能优秀，可以替代SWMM用于实时预测'
        elif any(r == '需要改进' for r in ratings):
            performance['overall_rating'] = '需要改进'
            performance['recommendation'] = '模型某些方面需要进一步优化'
        else:
            performance['overall_rating'] = '良好'
            performance['recommendation'] = '模型性能良好，可用于辅助决策'
        
        return performance