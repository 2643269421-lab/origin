#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Physically Consistent Loss Functions (时序正则化约束)
========================================================
SmoothLoss    — 二阶平滑正则项，抑制非物理高频波动
PeakTimeLoss  — 可微分峰值索引对齐损失 (soft argmax)
PhysicallyConsistentLoss — 联合损失: MSE + λ1*SmoothLoss + λ2*PeakTimeLoss

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SmoothLoss(nn.Module):
    """
    时序平滑正则项（Temporal Smoothness Regularization）
    惩罚预测序列的二阶差分 (离散曲率)

    物理直觉：真实水位过程线是连续且平滑的（由管网调蓄作用保证），
    不应出现锯齿状高频振荡。通过最小化二阶差分，强制预测序列
    在时序上具有合理的平滑性。

    注意：这是在时序维度上的通用平滑正则化，不依赖任何物理方程。
    其数学形式为离散Laplacian算子作用于时间维度。

    L_smooth = mean( |h[t+1] - 2*h[t] + h[t-1]|^2 )
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred):
        """
        Args:
            pred: (batch, seq_len, 1) or (batch, seq_len)
        Returns:
            scalar loss
        """
        if pred.dim() == 3:
            pred = pred.squeeze(-1)  # (batch, seq_len)

        # 二阶中心差分: h[t+1] - 2*h[t] + h[t-1]
        second_diff = pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]
        return torch.mean(second_diff ** 2)


class PeakTimeLoss(nn.Module):
    """
    峰值索引对齐正则项（Peak Index Alignment Regularization）
    通过可微分 soft-argmax 比较预测与目标峰值位置（索引而非绝对时间）

    物理直觉：真实的降雨-水位峰值时滞关系由管网汇流时间决定，
    模型预测的峰现索引应与SWMM基准一致。
    注意：5分钟时间步长下，0.3 min 等亚时间步精度来自 soft-argmax 的连续插值，
    实际物理可观测精度为 ±1 个时间步（±5 min）。

    使用带温度系数的 softmax 近似 argmax，保证完全可微分。
    温度越低，近似越接近硬 argmax（但梯度越稀疏）。

    L_peak = MSE( soft_argmax(pred), soft_argmax(target) ) / seq_len
    """

    def __init__(self, temperature: float = 1.0):
        """
        Args:
            temperature: softmax 温度系数 (越小越接近硬argmax)
        """
        super().__init__()
        self.temperature = temperature

    def _soft_argmax(self, x):
        """
        Args:
            x: (batch, seq_len) or (batch, seq_len, 1)
        Returns:
            (batch,) — 软峰值位置 (0 ~ seq_len-1 连续值)
        """
        if x.dim() == 3:
            x = x.squeeze(-1)

        batch_size, seq_len = x.shape
        # 对每个样本沿时间轴做 softmax
        weights = F.softmax(x / self.temperature, dim=1)  # (batch, seq_len)
        positions = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        # 加权求和 → 连续峰值位置
        soft_peak = (weights * positions.unsqueeze(0)).sum(dim=1)  # (batch,)
        return soft_peak

    def forward(self, pred, target):
        """
        Args:
            pred:   (batch, seq_len, 1) or (batch, seq_len)
            target: same shape as pred
        Returns:
            scalar loss (normalized by seq_len so magnitude ≈ 0~1)
        """
        if pred.dim() == 3:
            pred = pred.squeeze(-1)
        if target.dim() == 3:
            target = target.squeeze(-1)

        seq_len = pred.shape[1]
        peak_pred = self._soft_argmax(pred)
        peak_target = self._soft_argmax(target)

        # 归一化到 [0, 1] 区间（除以序列长度）
        return F.mse_loss(peak_pred / seq_len, peak_target / seq_len)


class MassConservationLoss(nn.Module):
    """
    节点质量守恒损失 (Node Mass Conservation Loss, NMCL)
    =====================================================
    基于水量连续方程 (dS/dt = Q_in - Q_out) 构造的物理约束损失项。

    物理原理:
      对排水管网中的任一节点，水流运动遵循质量守恒:
        dS/dt = Q_in(t) - Q_out(t)
      其中:
        - dS/dt ≈ A_surface × dH/dt  (蓄水量变化率，A_surface 为有效水面面积)
        - Q_in(t)  = C_runoff × A_catchment × P(t)  (降雨-径流转化入流)
        - Q_out(t) = α × max(0, H - H₀)^β  (rating curve 出流)

    离散残差 (有限差分, 时间步 Δt):
      R_t = Q_in(t) - α × max(0, H_t - H₀)^β - A_surface × (H_{t+1} - H_t) / Δt

    损失:
      L_mass = mean(R_t²)

    与 SmoothLoss / PeakTimeLoss 的本质区别:
      - 编码了流体力学中的质量守恒定律
      - 依赖管网物理参数 (A_surface, H₀, rating curve)
      - 是真正的物理约束 (physics-constrained)，而非时序正则化

    重要说明:
      Q_in 通过简化的降雨-径流系数法估算（非 SWMM 直接输出），
      存在一定近似误差。但在训练数据由 SWMM 生成的前提下，
      该近似已足够约束模型学习物理上一致的水位轨迹。

    引用分类: Physics-Constrained / Physics-Informed Soft Constraint
    """

    def __init__(self, node_physics_params, dt_min: float = 5.0,
                 Q_ref: float = None):
        """
        Args:
            node_physics_params: NodePhysicsParams 对象，包含:
                - H0: 节点底高程 (m) — 用于参考，不直接参与 Q_out 计算
                - A_surface: 有效水面面积 (m²)
                - catchment_area_m2: 上游汇水面积 (m²) — 用于 Q_ref 估算
                - runoff_coeff: 综合径流系数 — 简化估算回退方案
                - alpha, beta: rating curve 参数 Q_out = α·max(0, H)^β
            dt_min: 时间步长 (分钟)
            Q_ref: 参考流量 (m³/s)，用于量纲归一化。
                   若为 None，使用动态归一化（逐样本的自适应尺度）。
        """
        super().__init__()
        self.H0 = node_physics_params.H0        # 底高程 (参考)
        self.A_surface = node_physics_params.A_surface
        self.A_catchment = node_physics_params.catchment_area_m2
        self.C_runoff = node_physics_params.runoff_coeff
        self.alpha = node_physics_params.alpha
        self.beta = node_physics_params.beta

        # 时间步长 (秒)
        self.dt = dt_min * 60.0

        # 固定参考流量 (若为 None 则使用动态归一化)
        if Q_ref is not None:
            self.Q_ref = Q_ref
            self.use_dynamic_scale = False
        else:
            # 估算参考流量: 50mm/h × catchment_area
            self.Q_ref = (50.0 / 1000.0 / 3600.0) * self.A_catchment
            self.Q_ref = max(self.Q_ref, 1e-6)
            self.use_dynamic_scale = True

    def forward(self, pred, rainfall_mmh, rain_scaler=None, water_scaler=None,
                q_in_exact=None):
        """
        Args:
            pred:           (batch, seq_len, 1) — 模型预测水位 (已标准化)
            rainfall_mmh:   (batch, seq_len, 1) — 降雨强度 (已标准化)
                             当提供 q_in_exact 时，仅用于量纲参考
            rain_scaler:    MinMaxScaler — 降雨反标准化器 (可选)
            water_scaler:   MinMaxScaler — 水位反标准化器 (可选)
            q_in_exact:     (batch, seq_len) — 精确 Q_in (m³/s), 从 SWMM link flow 求和得到
                             若为 None，回退到简化 C·A·P 估算

        Returns:
            scalar loss (量纲归一化后的均方根残差)
        """
        batch_size, seq_len = pred.shape[0], pred.shape[1]

        # 反标准化到物理单位
        if water_scaler is not None:
            H_pred = _inverse_transform(pred, water_scaler)  # (batch, seq_len, 1) → m
        else:
            H_pred = pred  # 假设已在物理单位

        # 确保 2D: (batch, seq_len)
        if H_pred.dim() == 3:
            H_pred = H_pred.squeeze(-1)

        # ---- 1) Q_in: 精确 SWMM 流量 或 简化估算 ----
        if q_in_exact is not None:
            # 精确 Q_in — 直接使用 SWMM 输出的上游管渠流量求和
            Q_in = q_in_exact  # (batch, seq_len), unit: m³/s
            if Q_in.dim() == 3:
                Q_in = Q_in.squeeze(-1)
        else:
            # 简化估算（回退方案）
            if rain_scaler is not None:
                P = _inverse_transform(rainfall_mmh, rain_scaler)  # mm/h
            else:
                P = rainfall_mmh
            if P.dim() == 3:
                P = P.squeeze(-1)
            Q_in = self.C_runoff * self.A_catchment * P / (1000.0 * 3600.0)

        # ---- 2) 计算 Q_out: rating curve ----
        # H_pred 是相对节点底高程的水深 (m)，rating curve 直接以水深为变量
        # Q_out = α × max(0, H)^β
        H_above = torch.clamp(H_pred, min=0.0)  # (batch, seq_len)
        Q_out = self.alpha * (H_above ** self.beta)        # (batch, seq_len)

        # ---- 3) 计算 dH/dt: 前向差分 ----
        dH_dt = (H_pred[:, 1:] - H_pred[:, :-1]) / self.dt  # (batch, seq_len-1)

        # ---- 4) 计算残差 R_t ----
        Q_in_aligned = Q_in[:, :-1]     # (batch, seq_len-1)
        Q_out_aligned = Q_out[:, :-1]   # (batch, seq_len-1)

        storage_change = self.A_surface * dH_dt  # (batch, seq_len-1)

        R_t = Q_in_aligned - Q_out_aligned - storage_change  # (batch, seq_len-1)

        # ---- 4b) 干旱节点检测 ----
        # 若整个 batch 的 Q_in 均值接近零（如上游节点在低重现期训练中无入流），
        # NMCL 退化为惩罚"有水出无水入"的非物理约束，此时跳过
        if torch.mean(torch.abs(Q_in_aligned)) < 1e-10:
            return torch.tensor(0.0, device=Q_in_aligned.device, dtype=Q_in_aligned.dtype)

        # ---- 5) 量纲归一化 ----
        if self.use_dynamic_scale:
            sample_scale = torch.mean(
                torch.abs(Q_in_aligned) + torch.abs(Q_out_aligned), dim=1, keepdim=True
            )
            scale = torch.clamp(sample_scale, min=self.Q_ref)
            R_t_normalized = R_t / (scale + 1e-8)
        else:
            R_t_normalized = R_t / self.Q_ref

        # ---- 6) 均方根残差 ----
        loss_mass = torch.sqrt(torch.mean(R_t_normalized ** 2) + 1e-12)

        return loss_mass


def _inverse_transform(x_scaled, scaler):
    """
    对 PyTorch 张量进行反标准化。

    sklearn MinMaxScaler 的反变换:
      x_original = x_scaled × (data_max - data_min) + data_min

    Args:
        x_scaled: (batch, seq_len, 1) or (batch, seq_len) — PyTorch tensor
        scaler:   sklearn MinMaxScaler (fitted)

    Returns:
        PyTorch tensor (same shape)
    """
    import numpy as np
    data_min = scaler.data_min_
    data_max = scaler.data_max_
    # data_min/data_max 是 numpy array, shape 可能为 (1,) 或 (n_features,)
    data_min = float(np.asarray(data_min).flatten()[0])
    data_max = float(np.asarray(data_max).flatten()[0])

    # 转换为 PyTorch tensor (放在与输入相同的设备上)
    device = x_scaled.device
    data_min_t = torch.tensor(data_min, device=device, dtype=x_scaled.dtype)
    data_max_t = torch.tensor(data_max, device=device, dtype=x_scaled.dtype)

    return x_scaled * (data_max_t - data_min_t) + data_min_t


class PhysicallyConsistentLoss(nn.Module):
    """
    物理一致性联合损失函数（时序正则化 + 质量守恒约束 + 数据保真）

    L_total = MSE + λ₁·L_smooth + λ₂·L_peak + λ₃·L_mass

    其中:
      - MSE:        保证逐点预测精度（数据驱动）
      - SmoothLoss: 保证时序平滑性（时序正则化：管网调蓄导致平滑水位过程线）
      - PeakTimeLoss: 保证峰值索引正确（时序正则化：降雨-水位峰值时滞由汇流时间决定）
      - MassConservationLoss: 节点质量守恒约束（物理约束：dS/dt = Q_in - Q_out）

    推荐超参数:
      lambda_smooth = 0.01
      lambda_peak   = 0.05
      lambda_mass   = 0.01  (从 0.001~0.1 搜索)

    损失项分类:
      ┌──────────────────┬────────────────────┬──────────────────────┐
      │ 损失项           │ 类型               │ 物理来源             │
      ├──────────────────┼────────────────────┼──────────────────────┤
      │ SmoothLoss       │ 时序正则化         │ 通用信号平滑先验     │
      │ PeakTimeLoss     │ 时序正则化         │ 数据驱动峰值对齐     │
      │ MassConservation │ 物理约束 (hard)    │ 流体力学连续方程     │
      └──────────────────┴────────────────────┴──────────────────────┘
    """

    def __init__(self, lambda_smooth: float = 0.01, lambda_peak: float = 0.05,
                 lambda_mass: float = 0.01,
                 peak_temperature: float = 1.0,
                 node_physics_params=None):
        """
        Args:
            lambda_smooth: SmoothLoss 权重
            lambda_peak:   PeakTimeLoss 权重
            lambda_mass:   MassConservationLoss 权重 (设为0则退化为原版)
            peak_temperature: PeakTimeLoss 的 softmax 温度
            node_physics_params: NodePhysicsParams 对象 (启用 MassConservationLoss 时必填)
        """
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_peak = lambda_peak
        self.lambda_mass = lambda_mass

        self.mse = nn.MSELoss()
        self.smooth = SmoothLoss()
        self.peak_time = PeakTimeLoss(temperature=peak_temperature)

        if lambda_mass > 0 and node_physics_params is not None:
            self.mass_conservation = MassConservationLoss(node_physics_params)
        else:
            self.mass_conservation = None

    def forward(self, pred, target, rainfall=None,
                rain_scaler=None, water_scaler=None,
                q_in_exact=None,
                return_components=False):
        """
        Args:
            pred:     (batch, seq_len, 1) — 模型预测
            target:   (batch, seq_len, 1) — SWMM 标签
            rainfall: (batch, seq_len, 1) — 降雨输入 (MassConservationLoss 需要)
            rain_scaler:  MinMaxScaler — 降雨反标准化器
            water_scaler: MinMaxScaler — 水位反标准化器
            q_in_exact:   (batch, seq_len) — 精确 Q_in (m³/s), 来自 SWMM link flow 求和
            return_components: 是否返回各分量 (用于日志)
        Returns:
            total_loss or (total_loss, dict_of_components)
        """
        loss_mse = self.mse(pred, target)
        loss_smooth = self.smooth(pred)
        loss_peak = self.peak_time(pred, target)

        total = loss_mse + self.lambda_smooth * loss_smooth + self.lambda_peak * loss_peak

        components = {
            'mse': loss_mse.item(),
            'smooth': loss_smooth.item(),
            'peak_time': loss_peak.item(),
        }

        if self.mass_conservation is not None and rainfall is not None:
            loss_mass = self.mass_conservation(
                pred, rainfall,
                rain_scaler=rain_scaler,
                water_scaler=water_scaler,
                q_in_exact=q_in_exact,
            )
            total = total + self.lambda_mass * loss_mass
            components['mass_cons'] = loss_mass.item()

        components['total'] = total.item()

        if return_components:
            return total, components
        return total
