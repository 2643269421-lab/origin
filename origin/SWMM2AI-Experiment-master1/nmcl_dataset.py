#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NMCL 数据集与训练管线
=====================
支持精确 Q_in（从 SWMM 上游管渠流量求和）的 MassConservationLoss 训练。

与原始 SWMMDataset 的关键区别:
  - __getitem__ 返回 (rainfall, depth, q_in_exact) 三元素
  - 预计算全部数据（避免训练时重复运行 SWMM）
  - 支持 Q_in 按样本独立缩放
"""

import os
import json
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional, Dict
from datetime import datetime

from exact_qin import ExactQInExtractor, generate_nmcl_dataset
from node_physics import get_or_fit_node_physics, NodePhysicsParams


class NMCLDataset(Dataset):
    """
    含精确 Q_in 的训练数据集。

    __getitem__ 返回:
      (rainfall_tensor, depth_tensor, q_in_exact_tensor)

    其中 q_in_exact 是每时间步的精确节点入流 (m³/s)，由 SWMM 上游管渠流量求和得到。
    """

    def __init__(self,
                 inp_path: str = 'template.inp',
                 node_id: str = 'SN_001',
                 n_events: int = 200,
                 seq_length: int = 288,
                 time_step_min: int = 5,
                 max_return_period: int = 10,
                 cache_dir: str = None,
                 force_regenerate: bool = False):
        """
        Args:
            inp_path: template.inp 路径
            node_id:  目标节点
            n_events: 事件数
            seq_length: 序列长度
            time_step_min: 时间步长
            max_return_period: 最大重现期
            cache_dir: 缓存目录
            force_regenerate: 强制重新生成（跳过缓存）
        """
        self.node_id = node_id
        self.seq_length = seq_length
        self.time_step_min = time_step_min
        self.inp_path = inp_path

        if cache_dir is None:
            cache_dir = os.path.join('output', 'nmcl_cache')
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # 数据缓存路径
        cache_file = os.path.join(
            cache_dir,
            f'nmcl_data_{node_id}_n{n_events}_rp{max_return_period}.npz'
        )

        # 尝试从缓存加载
        if not force_regenerate and os.path.exists(cache_file):
            print(f"从缓存加载 NMCL 数据: {cache_file}")
            data = np.load(cache_file, allow_pickle=True)
            self.rainfall_raw = data['rainfall']
            self.depth_raw = data['depth']
            self.q_in_raw = data['q_in']
        else:
            print(f"生成 NMCL 数据: node={node_id}, n={n_events}, RP≤{max_return_period}")
            self.rainfall_raw, self.depth_raw, self.q_in_raw = generate_nmcl_dataset(
                inp_path=inp_path,
                node_id=node_id,
                n_events=n_events,
                seq_length=seq_length,
                time_step_min=time_step_min,
                max_return_period=max_return_period,
            )
            # 缓存
            np.savez_compressed(
                cache_file,
                rainfall=self.rainfall_raw,
                depth=self.depth_raw,
                q_in=self.q_in_raw,
            )
            print(f"数据已缓存: {cache_file}")

        n_valid = len(self.rainfall_raw)
        print(f"有效样本: {n_valid}/{n_events}")
        print(f"  降雨范围: {self.rainfall_raw.min():.1f} ~ {self.rainfall_raw.max():.1f} mm/h")
        print(f"  水深范围: {self.depth_raw.min():.4f} ~ {self.depth_raw.max():.4f} m")
        print(f"  Q_in 范围: {self.q_in_raw.min():.6f} ~ {self.q_in_raw.max():.6f} m³/s")

        # 标准化
        self._fit_scalers()

    def _fit_scalers(self):
        """拟合标准化器（仅基于训练数据）"""
        rain_2d = self.rainfall_raw.reshape(-1, 1)
        depth_2d = self.depth_raw.reshape(-1, 1)

        self.rain_scaler = MinMaxScaler()
        self.depth_scaler = MinMaxScaler()

        self.rainfall = self.rain_scaler.fit_transform(rain_2d).reshape(
            self.rainfall_raw.shape
        )
        self.depth = self.depth_scaler.fit_transform(depth_2d).reshape(
            self.depth_raw.shape
        )

        # Q_in 不做标准化（保留物理单位 m³/s）
        # 但记录全局统计量用于 loss 中的动态缩放
        self.q_in_mean = float(np.mean(self.q_in_raw))
        self.q_in_std = float(np.std(self.q_in_raw))

    def __len__(self):
        return len(self.rainfall)

    def __getitem__(self, idx):
        rain_t = torch.FloatTensor(self.rainfall[idx]).unsqueeze(-1)  # (seq, 1)
        depth_t = torch.FloatTensor(self.depth[idx]).unsqueeze(-1)   # (seq, 1)
        q_in_t = torch.FloatTensor(self.q_in_raw[idx])               # (seq,)

        return rain_t, depth_t, q_in_t


def train_nmcl_model(
    node_id: str = 'SN_001',
    n_events: int = 200,
    epochs: int = 200,
    lr: float = 0.001,
    batch_size: int = 32,
    lambda_smooth: float = 0.01,
    lambda_peak: float = 0.05,
    lambda_mass: float = 0.01,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    output_dir: str = None,
    device: str = None,
) -> Dict:
    """
    完整的 NMCL 训练流程。

    返回训练结果字典，包含模型路径、训练历史和评估指标。
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = f'output/nmcl_train_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(
        output_dir, f'pcca_lstm_nmcl_{node_id}_lmass{lambda_mass}.pth'
    )

    print(f"{'='*60}")
    print(f"  NMCL 训练: node={node_id}, λ_mass={lambda_mass}")
    print(f"  Model: PCCA-LSTM, epochs={epochs}, n_events={n_events}")
    print(f"{'='*60}")

    # 1) 获取物理参数
    print("\n[1/4] 提取节点物理参数...")
    physics = get_or_fit_node_physics(
        'template.inp', node_id, cache_dir=output_dir,
    )
    print(f"  H₀={physics.H0:.3f}m, A_surf={physics.A_surface:.3f}m²")
    print(f"  Rating curve: α={physics.alpha:.4f}, β={physics.beta:.4f}")

    # 2) 生成数据集（含精确 Q_in）
    print("\n[2/4] 生成 NMCL 训练数据...")
    t0 = time.time()
    dataset = NMCLDataset(
        inp_path='template.inp',
        node_id=node_id,
        n_events=n_events,
        max_return_period=10,
        cache_dir=output_dir,
    )
    print(f"  耗时: {time.time() - t0:.1f}s")

    # 3) 划分数据集
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    print(f"  训练/验证/测试: {train_size}/{val_size}/{test_size}")

    # 4) 初始化模型
    print("\n[3/4] 初始化 PCCA-LSTM 模型...")
    from attention import PCCALSTM
    model = PCCALSTM(
        input_size=1, hidden_size=hidden_size,
        num_layers=num_layers, output_size=1, dropout=dropout,
    ).to(device)

    from physics_loss import PhysicallyConsistentLoss
    criterion = PhysicallyConsistentLoss(
        lambda_smooth=lambda_smooth,
        lambda_peak=lambda_peak,
        lambda_mass=lambda_mass,
        node_physics_params=physics,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=15, factor=0.5
    )

    # 5) 训练循环
    print("\n[4/4] 开始训练...")
    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # --- 训练 ---
        model.train()
        train_loss = 0.0
        train_comp = {'mse': 0.0, 'smooth': 0.0, 'peak_time': 0.0, 'mass_cons': 0.0}

        for rain, depth, q_in in train_loader:
            rain = rain.to(device)
            depth = depth.to(device)
            q_in = q_in.to(device)

            optimizer.zero_grad()
            pred = model(rain)

            loss, comps = criterion(
                pred, depth,
                rainfall=rain,
                rain_scaler=dataset.rain_scaler,
                water_scaler=dataset.depth_scaler,
                q_in_exact=q_in,  # ← 精确 Q_in!
                return_components=True,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            for k in train_comp:
                if k in comps:
                    train_comp[k] += comps[k]

        # --- 验证 ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for rain, depth, q_in in val_loader:
                rain = rain.to(device)
                depth = depth.to(device)
                q_in = q_in.to(device)
                pred = model(rain)
                vloss, _ = criterion(
                    pred, depth,
                    rainfall=rain,
                    rain_scaler=dataset.rain_scaler,
                    water_scaler=dataset.depth_scaler,
                    q_in_exact=q_in,
                    return_components=True,
                )
                val_loss += vloss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_type': 'PCCA-LSTM',
                'model_params': {
                    'input_size': 1, 'hidden_size': hidden_size,
                    'num_layers': num_layers, 'output_size': 1,
                },
                'rain_scaler': dataset.rain_scaler,
                'water_scaler': dataset.depth_scaler,
                'node_id': node_id,
                'n_events': n_events,
                'time_step_min': 5,
                'seq_length': dataset.seq_length,
                'loss_type': 'physically_consistent',
                'lambda_smooth': lambda_smooth,
                'lambda_peak': lambda_peak,
                'lambda_mass': lambda_mass,
                'physics_params': {
                    'H0': physics.H0, 'A_surface': physics.A_surface,
                    'alpha': physics.alpha, 'beta': physics.beta,
                },
            }, model_path)

        if (epoch + 1) % 20 == 0:
            n = len(train_loader)
            tm = train_comp['mse'] / n
            ts = train_comp['smooth'] / n
            tp = train_comp['peak_time'] / n
            info = f'MSE:{tm:.4f} Smooth:{ts:.4f} Peak:{tp:.4f}'
            if 'mass_cons' in train_comp:
                tmass = train_comp['mass_cons'] / n
                info += f' Mass:{tmass:.4f}'
            print(f'Epoch [{epoch+1:3d}/{epochs}] Train: {train_loss:.6f} ({info}) '
                  f'Val: {val_loss:.6f}')

    # 6) 最终保存
    final_path = os.path.join(output_dir, f'pcca_lstm_nmcl_{node_id}_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': 'PCCA-LSTM',
        'model_params': {
            'input_size': 1, 'hidden_size': hidden_size,
            'num_layers': num_layers, 'output_size': 1,
        },
        'rain_scaler': dataset.rain_scaler,
        'water_scaler': dataset.depth_scaler,
        'node_id': node_id,
        'n_events': n_events,
        'time_step_min': 5,
        'seq_length': dataset.seq_length,
        'loss_type': 'physically_consistent',
        'lambda_smooth': lambda_smooth,
        'lambda_peak': lambda_peak,
        'lambda_mass': lambda_mass,
        'physics_params': {
            'H0': physics.H0, 'A_surface': physics.A_surface,
            'alpha': physics.alpha, 'beta': physics.beta,
        },
    }, final_path)

    # 7) 保存训练历史
    results = {
        'node_id': node_id,
        'lambda_mass': lambda_mass,
        'lambda_smooth': lambda_smooth,
        'lambda_peak': lambda_peak,
        'epochs': epochs,
        'n_events': n_events,
        'best_val_loss': float(best_val_loss),
        'final_train_loss': float(train_losses[-1]),
        'final_val_loss': float(val_losses[-1]),
        'model_path': model_path,
        'physics': {
            'H0': physics.H0,
            'A_surface': physics.A_surface,
            'catchment_ha': physics.catchment_area_ha,
            'runoff_coeff': physics.runoff_coeff,
            'alpha': physics.alpha,
            'beta': physics.beta,
        },
        'q_in_stats': {
            'mean': dataset.q_in_mean,
            'std': dataset.q_in_std,
        },
    }

    with open(os.path.join(output_dir, 'training_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n训练完成!")
    print(f"  最佳验证损失: {best_val_loss:.6f}")
    print(f"  最佳模型: {model_path}")
    print(f"  最终模型: {final_path}")
    results['final_model_path'] = final_path
    return results
