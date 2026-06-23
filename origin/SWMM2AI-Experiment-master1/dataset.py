# ==================== 与LSTM数据集集成 ====================
import numpy as np
from swmm.rainfall.generator import RainfallGenerator
from swmm.simulator import SWMMSimulator
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class SWMMDataset(Dataset):
    """SWMM降雨-水位数据集"""

    def __init__(self, n_events: int = 100, seq_length: int = 288,
                 time_step_min: int = 5,
                 swmm_simulator: SWMMSimulator = None,
                 max_return_period: int = None,
                 return_period: int = None):
        """
        初始化数据集

        参数:
            n_events: 降雨事件数量
            seq_length: 输入序列长度
            predict_length: 预测序列长度
            time_step_min: 时间步长
            swmm_simulator: SWMM模拟器实例
            max_return_period: 最大重现期（仅生成 ≤ 此值的降雨事件，用于训练集约束）
            return_period: 固定重现期（用于生成特定重现期的测试集）
        """
        self.n_events = n_events
        self.seq_length = seq_length
        self.time_step_min = time_step_min
        self.max_return_period = max_return_period
        self.return_period = return_period

        # 创建暴雨生成器
        self.rainfall_generator = RainfallGenerator(time_step_min=time_step_min)

        # 创建SWMM模拟器（如果未提供）
        if swmm_simulator is None:
            self.simulator = SWMMSimulator(template_inp_path='template.inp', output_element='SN_001', output_type='node', output_variable='depth')
        else:
            self.simulator = swmm_simulator

        # 生成数据
        self._generate_dataset()
        self._get_training_data()

    def _generate_dataset(self):
        """生成数据集"""

        # 根据参数显示不同信息
        if self.return_period is not None:
            print(f"生成 {self.n_events} 个 {self.return_period}年一遇 降雨事件...")
        elif self.max_return_period is not None:
            print(f"生成 {self.n_events} 个降雨事件 (重现期 ≤ {self.max_return_period}年)...")
        else:
            print(f"生成 {self.n_events} 个降雨事件...")

        # 生成降雨事件
        self.rainfall_events = self.rainfall_generator.generate_multiple_events(
            n_events=self.n_events,
            seq_length=self.seq_length,
            rain_type='chicago',
            min_duration=1,
            max_duration=6,
            max_return_period=self.max_return_period,
            return_period=self.return_period
        )
        
        # 批量模拟
        self.water_level_results = self._batch_simulate_rainfall_events(
            self.rainfall_events, self.time_step_min
        )
        
        # 提取水位数据，并保持与降雨事件对齐
        self.water_level_events = []
        valid_rainfall_events = []
        for i, result in enumerate(self.water_level_results):
            if result is None or result.get('values') is None:
                print(f"警告: 事件 {i} 的SWMM模拟结果为空，跳过")
                continue
            self.water_level_events.append(result['values'])
            valid_rainfall_events.append(self.rainfall_events[i])

        self.rainfall_events = valid_rainfall_events
        
        self.rainfall_events = np.array(self.rainfall_events)
        self.water_level_events = np.array(self.water_level_events)
        
        print(f"数据集创建完成: 降雨数据形状 {self.rainfall_events.shape}, 水位数据形状 {self.water_level_events.shape}")
    
    # ==================== 批量模拟函数 ====================

    def _batch_simulate_rainfall_events(self, rainfall_events: List[np.ndarray], 
                                      time_step_min: int = 5) -> List[Dict]:
        """
        批量模拟多个降雨事件
        
        参数:
            rainfall_events: 降雨事件列表
            simulator: SWMM模拟器实例
            time_step_min: 时间步长
            
        返回:
            模拟结果列表
        """
        all_results = []
        
        print(f"开始批量模拟 {len(rainfall_events)} 个降雨事件...")
        
        for i, rainfall in enumerate(rainfall_events):
            if (i + 1) % 10 == 0:
                print(f"  已模拟 {i+1}/{len(rainfall_events)} 个事件")
            
            # 运行模拟
            try:
                results = self.simulator.run_swmm_simulation(
                    rainfall_mm_h=rainfall
                )
                
                all_results.append(results)
                
            except Exception as e:
                print(f"事件 {i+1} 模拟失败: {e}")
                continue
        
        print(f"批量模拟完成，成功模拟 {len(all_results)} 个事件")
        
        return all_results

    def _get_training_data(self):
        """
        获取训练数据
        
        参数:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            
        返回:
            (X_train, y_train, X_val, y_val, X_test, y_test, scalers)
        """
        from sklearn.preprocessing import MinMaxScaler
        
        # 标准化
        self.rain_scaler = MinMaxScaler()
        self.water_scaler = MinMaxScaler()
        
        rainfall_2d = self.rainfall_events.reshape(-1, 1)
        water_2d = self.water_level_events.reshape(-1, 1)
        
        self.rainfall_scaled = self.rain_scaler.fit_transform(rainfall_2d).reshape(
            self.rainfall_events.shape
        )
        self.water_level_scaled = self.water_scaler.fit_transform(water_2d).reshape(
            self.water_level_events.shape
        )
        
        # 每个事件就是一个样本：输入降雨序列，输出对应水位序列
        self.X = self.rainfall_scaled
        self.y = self.water_level_scaled
        
        print(f"数据集大小: {len(self.X)} 个样本")
        print(f"输入形状: {self.X.shape}")
        print(f"输出形状: {self.y.shape}")
        
        return self.X, self.y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # 每个样本：输入降雨序列，输出水位序列
        # 形状: (seq_length, 1)
        rainfall_seq = self.X[idx]
        water_seq = self.y[idx]
        
        # 转换为张量
        rainfall_tensor = torch.FloatTensor(rainfall_seq).unsqueeze(-1)  # (seq_length, 1)
        water_tensor = torch.FloatTensor(water_seq).unsqueeze(-1)  # (seq_length, 1)
        
        return rainfall_tensor, water_tensor