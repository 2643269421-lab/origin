#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
# 强制使用 UTF-8 输出，解决 Windows GBK 编码问题
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

"""
极端暴雨外推实验 (Extreme Rainfall Extrapolation Experiment)
============================================================
研究目的：评估模型对超出训练分布降雨事件的预测能力

实验设计：
  - 训练集：仅包含重现期 ≤ 10 年的设计降雨事件
  - 测试集：20年、30年、50年、100年重现期暴雨事件
  - 模拟模型在遭遇超历史经验降雨时的工作场景
  - 检验外推能力（Extrapolation Ability）与极端事件泛化能力

"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

# 添加项目根目录
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import SWMMDataset
from swmm.simulator import SWMMSimulator
from swmm.rainfall.generator import RainfallGenerator
from model import Predictor
from registry import create_model
from attention import CausalAttentionLSTM
from gru import SimpleGRU
from lstm import SimpleLSTM
from attention import AttentionLSTM

# ======================== 全局配置 ========================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 实验参数
N_TRAIN_EVENTS = 200           # 训练事件数
TRAIN_MAX_RETURN_PERIOD = 10   # 训练集最大重现期
N_TEST_PER_RP = 20             # 每个重现期测试事件数
EXTREME_RETURN_PERIODS = [20, 30, 50, 100]  # 测试极端重现期
SEQ_LENGTH = 288               # 序列长度 (24h @ 5min)
TIME_STEP_MIN = 5              # 时间步长（分钟）
EPOCHS = 200                   # 训练轮数
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# 模型配置
MODEL_CONFIGS = [
    {'type': 'SimpleLSTM',          'name': 'LSTM',                    'color': '#f27970',
     'params': {'input_size': 1, 'hidden_size': 128, 'num_layers': 2, 'output_size': 1}},
    {'type': 'SimpleGRU',           'name': 'GRU',                     'color': '#54b345',
     'params': {'input_size': 1, 'hidden_size': 128, 'num_layers': 2, 'output_size': 1}},
    {'type': 'AttentionLSTM',       'name': 'Attention LSTM',          'color': '#8983bf',
     'params': {'input_size': 1, 'hidden_size': 128, 'num_layers': 2, 'output_size': 1}},
    {'type': 'CausalAttentionLSTM', 'name': 'Causal Attention LSTM',   'color': '#c76da2',
     'params': {'input_size': 1, 'hidden_size': 128, 'num_layers': 2, 'output_size': 1}},
]

# 输出根目录
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_ROOT = os.path.join('output', f'extreme_experiment_{TIMESTAMP}')

def create_output_dir():
    """创建输出目录"""
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    print(f"实验输出目录: {OUTPUT_ROOT}")
    return OUTPUT_ROOT


# ======================== 第一阶段：生成训练数据 ========================
def generate_training_data(device='cuda'):
    """
    生成训练数据（仅含重现期 ≤ 10 年的事件）
    """
    print("\n" + "=" * 60)
    print("  第一阶段：生成训练数据（重现期 ≤ 10年）")
    print("=" * 60)

    print(f"\n生成 {N_TRAIN_EVENTS} 个训练降雨事件（重现期 ≤ {TRAIN_MAX_RETURN_PERIOD}年）...")
    t0 = time.time()

    dataset = SWMMDataset(
        n_events=N_TRAIN_EVENTS,
        seq_length=SEQ_LENGTH,
        time_step_min=TIME_STEP_MIN,
        max_return_period=TRAIN_MAX_RETURN_PERIOD
    )

    elapsed = time.time() - t0
    print(f"训练数据生成完成，耗时: {elapsed:.1f}s")
    print(f"有效样本数: {len(dataset)}")

    # 划分数据集
    n_total = len(dataset)
    train_size = int(0.8 * n_total)
    val_size = int(0.1 * n_total)
    test_size = n_total - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"训练集: {len(train_dataset)} | 验证集: {len(val_dataset)} | 测试集(≤10yr): {len(test_dataset)}")

    return {
        'dataset': dataset,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
    }


# ======================== 第二阶段：训练模型 ========================
def train_single_model(model_config, data, device='cuda'):
    """
    训练单个模型（仅使用 ≤10yr 数据）
    """
    model_type = model_config['type']
    model_name = model_config['name']
    model_params = model_config['params']
    model_path = os.path.join(OUTPUT_ROOT, f'extreme_{model_type.lower()}_model.pth')

    print(f"\n{'─' * 50}")
    print(f"  训练 {model_name} 模型...")
    print(f"{'─' * 50}")

    # 创建模型
    model = create_model(model_type, **model_params)
    model = model.to(device)

    # 数据加载器
    train_loader = DataLoader(data['train_dataset'], batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(data['val_dataset'], batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        # 训练
        model.train()
        train_loss = 0
        for batch_data, batch_target in train_loader:
            batch_data, batch_target = batch_data.to(device), batch_target.to(device)
            optimizer.zero_grad()
            output = model(batch_data)
            loss = criterion(output, batch_target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_data, batch_target in val_loader:
                batch_data, batch_target = batch_data.to(device), batch_target.to(device)
                output = model(batch_data)
                val_loss += criterion(output, batch_target).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch [{epoch+1}/{EPOCHS}] Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'rain_scaler': data['dataset'].rain_scaler,
        'water_scaler': data['dataset'].water_scaler,
        'seq_length': data['dataset'].seq_length,
        'n_events': data['dataset'].n_events,
        'time_step_min': data['dataset'].time_step_min,
        'model_type': model_type,
        'model_params': model_params,
        'train_max_return_period': TRAIN_MAX_RETURN_PERIOD,
    }, model_path)

    print(f"  {model_name} 训练完成 | 最佳验证损失: {best_val_loss:.6f} | 模型保存至: {model_path}")

    return {
        'config': model_config,
        'model_path': model_path,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
    }


def train_all_models(data, device='cuda'):
    """训练所有模型"""
    print("\n" + "=" * 60)
    print("  第二阶段：训练模型（仅限 ≤10yr 数据）")
    print("=" * 60)

    results = []
    for config in MODEL_CONFIGS:
        result = train_single_model(config, data, device)
        results.append(result)

    return results


# ======================== 第三阶段：生成极端暴雨测试集 ========================
def generate_extreme_test_sets():
    """
    为每个极端重现期生成测试集
    """
    print("\n" + "=" * 60)
    print("  第三阶段：生成极端暴雨测试集")
    print("=" * 60)

    test_sets = {}

    for rp in EXTREME_RETURN_PERIODS:
        print(f"\n生成 {rp}年一遇 测试事件（共 {N_TEST_PER_RP} 个）...")
        t0 = time.time()

        dataset = SWMMDataset(
            n_events=N_TEST_PER_RP,
            seq_length=SEQ_LENGTH,
            time_step_min=TIME_STEP_MIN,
            return_period=rp
        )

        elapsed = time.time() - t0

        # 记录统计信息
        rainfall_max = dataset.rainfall_events.max()
        rainfall_sum = dataset.rainfall_events.sum() / N_TEST_PER_RP * TIME_STEP_MIN / 60
        water_max = dataset.water_level_events.max()

        test_sets[rp] = {
            'dataset': dataset,
            'n_samples': len(dataset),
            'stats': {
                'max_rainfall_mmh': float(rainfall_max),
                'avg_total_rainfall_mm': float(rainfall_sum),
                'max_water_depth_m': float(water_max),
            }
        }

        print(f"  {rp}年一遇: {len(dataset)} 个样本 | "
              f"最大降雨强度: {rainfall_max:.1f} mm/h | "
              f"平均总降雨量: {rainfall_sum:.1f} mm | "
              f"最大水位: {water_max:.3f} m | "
              f"耗时: {elapsed:.1f}s")

    return test_sets


# ======================== 第四阶段：评估外推能力 ========================
def evaluate_on_extreme_test(model_result, test_sets, device='cuda'):
    """
    评估单个模型在所有极端重现期上的表现
    """
    model_config = model_result['config']
    model_type = model_config['type']
    model_name = model_config['name']
    model_path = model_result['model_path']

    predictor = Predictor(model_path=model_path, output_dir=OUTPUT_ROOT, device=device)

    evaluation_results = {}

    for rp, test_set in test_sets.items():
        dataset = test_set['dataset']

        # 获取原始（非标准化）数据
        rainfall_events = []
        water_targets = []
        for i in range(len(dataset)):
            rain_tensor, water_tensor = dataset[i]
            # 反标准化
            rain = dataset.rain_scaler.inverse_transform(rain_tensor.numpy().reshape(-1, 1)).flatten()
            water = dataset.water_scaler.inverse_transform(water_tensor.numpy().reshape(-1, 1)).flatten()
            rainfall_events.append(rain)
            water_targets.append(water)

        rainfall_events = np.array(rainfall_events)
        water_targets = np.array(water_targets)

        # 批量预测
        water_predictions = predictor.predict_batch(rainfall_events)

        # 计算每个样本的指标
        all_metrics = []
        for i in range(len(dataset)):
            pred = water_predictions[i]
            target = water_targets[i]
            rain = rainfall_events[i]

            # 只在有降雨或水位变化的区域计算MAPE（避免除以0）
            active_mask = target > 0.001  # 水位 > 1mm

            mse = np.mean((pred - target) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(pred - target))

            if active_mask.sum() > 0:
                mape = np.mean(np.abs((pred[active_mask] - target[active_mask]) / (target[active_mask] + 1e-10))) * 100
                max_rel_error = np.max(np.abs((pred[active_mask] - target[active_mask]) / (target[active_mask] + 1e-10))) * 100
            else:
                mape = 0
                max_rel_error = 0

            # R²
            ss_res = np.sum((target - pred) ** 2)
            ss_tot = np.sum((target - np.mean(target)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-10))

            # 峰值误差
            peak_idx = np.argmax(target)
            peak_error = pred[peak_idx] - target[peak_idx]
            peak_rel_error = np.abs(peak_error) / (target[peak_idx] + 1e-10) * 100

            all_metrics.append({
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape,
                'R2': r2,
                'MaxAbsError': np.max(np.abs(pred - target)),
                'MaxRelError': max_rel_error,
                'PeakError': peak_error,
                'PeakRelError': peak_rel_error,
                'MaxRainfall': float(rain.max()),
                'MaxWaterTarget': float(target.max()),
                'MaxWaterPred': float(pred.max()),
            })

        # 汇总统计
        metrics_array = {k: np.array([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
        summary = {
            'return_period': rp,
            'n_samples': len(dataset),
            'MSE_mean': float(metrics_array['MSE'].mean()),
            'MSE_std': float(metrics_array['MSE'].std()),
            'RMSE_mean': float(metrics_array['RMSE'].mean()),
            'RMSE_std': float(metrics_array['RMSE'].std()),
            'MAE_mean': float(metrics_array['MAE'].mean()),
            'MAE_std': float(metrics_array['MAE'].std()),
            'MAPE_mean': float(metrics_array['MAPE'].mean()),
            'MAPE_std': float(metrics_array['MAPE'].std()),
            'R2_mean': float(metrics_array['R2'].mean()),
            'R2_std': float(metrics_array['R2'].std()),
            'PeakError_mean': float(metrics_array['PeakError'].mean()),
            'PeakRelError_mean': float(metrics_array['PeakRelError'].mean()),
            'MaxAbsError_mean': float(metrics_array['MaxAbsError'].mean()),
        }

        evaluation_results[rp] = {
            'summary': summary,
            'per_sample': all_metrics,
            'predictions': water_predictions,
            'targets': water_targets,
            'rainfall': rainfall_events,
        }

        print(f"  {model_name} @ {rp}yr | RMSE: {summary['RMSE_mean']:.4f}+/-{summary['RMSE_std']:.4f} | "
              f"MAE: {summary['MAE_mean']:.4f} | R²: {summary['R2_mean']:.4f} | "
              f"峰值误差: {summary['PeakError_mean']:.4f}m")

    return evaluation_results


def evaluate_all_models(train_results, test_sets, device='cuda'):
    """评估所有模型"""
    print("\n" + "=" * 60)
    print("  第四阶段：极端暴雨外推能力评估")
    print("=" * 60)

    all_evaluations = {}
    for model_result in train_results:
        print(f"\n--- 评估 {model_result['config']['name']} ---")
        all_evaluations[model_result['config']['type']] = evaluate_on_extreme_test(
            model_result, test_sets, device
        )

    return all_evaluations


# ======================== 第五阶段：可视化与分析 ========================
def visualize_training_curves(train_results):
    """绘制所有模型的训练曲线"""
    print("\n--- 绘制训练曲线 ---")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, result in enumerate(train_results):
        ax = axes[idx]
        ax.plot(result['train_losses'], label='训练损失', alpha=0.7, linewidth=1)
        ax.plot(result['val_losses'], label='验证损失', alpha=0.9, linewidth=1.5)
        ax.set_title(result['config']['name'], fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle('模型训练过程（仅 ≤10yr 重现期数据）', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_ROOT, 'training_curves.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  训练曲线已保存: {save_path}")


def visualize_extrapolation_heatmap(all_evaluations):
    """绘制外推能力热力图 —— 各模型在不同重现期下的性能退化"""
    print("\n--- 绘制外推能力热力图 ---")

    metrics_to_plot = ['RMSE_mean', 'MAE_mean', 'MAPE_mean', 'R2_mean']
    metric_labels = ['RMSE (m)', 'MAE (m)', 'MAPE (%)', 'R²']
    model_names = [cfg['name'] for cfg in MODEL_CONFIGS]
    return_periods = [str(rp) for rp in EXTREME_RETURN_PERIODS]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        ax = axes[idx]

        # 构建数据矩阵
        data = np.zeros((len(model_names), len(return_periods)))
        for i, model_type in enumerate([cfg['type'] for cfg in MODEL_CONFIGS]):
            for j, rp in enumerate(EXTREME_RETURN_PERIODS):
                data[i, j] = all_evaluations[model_type][rp]['summary'][metric]

        # 绘制热力图
        im = ax.imshow(data, cmap='RdYlGn_r' if metric != 'R2_mean' else 'RdYlGn', aspect='auto')

        # 添加数值标注
        for i in range(len(model_names)):
            for j in range(len(return_periods)):
                if metric == 'MAPE_mean' or metric == 'R2_mean':
                    text = f'{data[i, j]:.2f}'
                else:
                    text = f'{data[i, j]:.4f}'
                ax.text(j, i, text, ha='center', va='center', fontsize=10,
                       color='white' if abs(data[i, j]) > 0.5 * data.max() else 'black')

        ax.set_xticks(range(len(return_periods)))
        ax.set_xticklabels([f'{rp}年' for rp in EXTREME_RETURN_PERIODS])
        ax.set_yticks(range(len(model_names)))
        ax.set_yticklabels(model_names)
        ax.set_title(label, fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle('极端暴雨外推能力评估 — 各模型在不同重现期下的性能', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_ROOT, 'extrapolation_heatmap.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  外推能力热力图已保存: {save_path}")


def visualize_degradation_curves(all_evaluations):
    """绘制性能退化曲线 —— 性能随重现期增长的变化"""
    print("\n--- 绘制性能退化曲线 ---")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    metrics = ['RMSE_mean', 'MAE_mean', 'MAPE_mean', 'R2_mean']
    metric_labels = ['RMSE (m)', 'MAE (m)', 'MAPE (%)', 'R²']
    metric_stds = ['RMSE_std', 'MAE_std', 'MAPE_std', 'R2_std']

    x = np.arange(len(EXTREME_RETURN_PERIODS))

    for idx, (metric, label, std_key) in enumerate(zip(metrics, metric_labels, metric_stds)):
        ax = axes[idx]

        for cfg in MODEL_CONFIGS:
            model_type = cfg['type']
            y_vals = [all_evaluations[model_type][rp]['summary'][metric] for rp in EXTREME_RETURN_PERIODS]
            y_stds = [all_evaluations[model_type][rp]['summary'][std_key] for rp in EXTREME_RETURN_PERIODS]

            ax.errorbar(x, y_vals, yerr=y_stds, marker='o', linewidth=2, markersize=8,
                       label=cfg['name'], color=cfg['color'], capsize=4, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([f'{rp}年' for rp in EXTREME_RETURN_PERIODS])
        ax.set_xlabel('重现期（年）', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f'{label} 随重现期变化', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('模型外推性能退化分析 — 训练上限为10年重现期', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_ROOT, 'degradation_curves.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  性能退化曲线已保存: {save_path}")


def visualize_extreme_events(all_evaluations, test_sets):
    """为每个极端重现期绘制典型事件对比图"""
    print("\n--- 绘制极端事件预测对比图 ---")

    for rp in EXTREME_RETURN_PERIODS:
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()

        # 取该重现期最极端的一个事件（水位最高）
        test_set = test_sets[rp]['dataset']
        water_maxes = []
        for i in range(len(test_set)):
            _, wt = test_set[i]
            w = test_set.water_scaler.inverse_transform(wt.numpy().reshape(-1, 1)).flatten()
            water_maxes.append(w.max())
        extreme_idx = np.argmax(water_maxes)

        # 获取该事件的真实数据
        _, water_tensor = test_set[extreme_idx]
        rain_tensor, _ = test_set[extreme_idx]
        rain = test_set.rain_scaler.inverse_transform(rain_tensor.numpy().reshape(-1, 1)).flatten()
        water_true = test_set.water_scaler.inverse_transform(water_tensor.numpy().reshape(-1, 1)).flatten()

        time_hours = np.arange(SEQ_LENGTH) * TIME_STEP_MIN / 60

        for idx, cfg in enumerate(MODEL_CONFIGS):
            ax = axes[idx]
            model_type = cfg['type']

            # 获取该模型对该事件的预测
            pred = all_evaluations[model_type][rp]['predictions'][extreme_idx]

            # 双轴：降雨
            ax_rain = ax.twinx()
            ax_rain.bar(time_hours, rain, width=TIME_STEP_MIN/60/1.5,
                       alpha=0.2, color='blue', label=f'降雨 ({rp}年一遇)')
            ax_rain.set_ylabel('降雨强度 (mm/h)', color='blue', fontsize=9)
            ax_rain.tick_params(axis='y', labelcolor='blue')

            # 水位
            ax.plot(time_hours, water_true, 'k--', linewidth=2, label='SWMM 真实水位', alpha=0.8)
            ax.plot(time_hours, pred, color=cfg['color'], linewidth=2.5, label=f'{cfg["name"]} 预测')

            ax.set_xlabel('时间 (h)')
            ax.set_ylabel('水位 (m)', color='black')
            ax.set_title(f'{cfg["name"]} @ {rp}年一遇', fontweight='bold')
            ax.grid(True, alpha=0.3)

            # 合并图例
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax_rain.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

            # 添加误差统计
            mae = all_evaluations[model_type][rp]['per_sample'][extreme_idx]['MAE']
            mape = all_evaluations[model_type][rp]['per_sample'][extreme_idx]['MAPE']
            peak_err = all_evaluations[model_type][rp]['per_sample'][extreme_idx]['PeakError']
            ax.text(0.02, 0.98, f'MAE={mae:.4f}m  MAPE={mape:.1f}%\n峰值误差={peak_err:.4f}m',
                   transform=ax.transAxes, verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

        fig.suptitle(f'{rp}年一遇极端暴雨事件 — 多模型预测对比', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()

        save_path = os.path.join(OUTPUT_ROOT, f'extreme_event_{rp}yr.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  {rp}年一遇事件对比图已保存: {save_path}")


def visualize_comprehensive_comparison(all_evaluations, test_sets):
    """绘制综合对比图 —— 所有模型在所有重现期下的表现总览"""
    print("\n--- 绘制综合对比总览图 ---")

    n_models = len(MODEL_CONFIGS)
    n_rps = len(EXTREME_RETURN_PERIODS)

    fig, axes = plt.subplots(n_models, n_rps, figsize=(5 * n_rps, 4 * n_models))
    fig.suptitle('极端暴雨外推实验 — 完整对比矩阵\n'
                f'(训练数据: ≤{TRAIN_MAX_RETURN_PERIOD}年重现期 | 测试数据: 极端重现期)',
                fontsize=14, fontweight='bold', y=0.995)

    for i, cfg in enumerate(MODEL_CONFIGS):
        model_type = cfg['type']

        for j, rp in enumerate(EXTREME_RETURN_PERIODS):
            ax = axes[i, j] if n_models > 1 else axes[j]

            test_set = test_sets[rp]['dataset']
            # 找水位峰值最大的事件
            best_idx = 0
            best_max = 0
            for k in range(len(test_set)):
                _, wt = test_set[k]
                w = test_set.water_scaler.inverse_transform(wt.numpy().reshape(-1, 1)).flatten()
                if w.max() > best_max:
                    best_max = w.max()
                    best_idx = k

            rain_tensor, water_tensor = test_set[best_idx]
            rain = test_set.rain_scaler.inverse_transform(rain_tensor.numpy().reshape(-1, 1)).flatten()
            water_true = test_set.water_scaler.inverse_transform(water_tensor.numpy().reshape(-1, 1)).flatten()
            water_pred = all_evaluations[model_type][rp]['predictions'][best_idx]

            time_hours = np.arange(SEQ_LENGTH) * TIME_STEP_MIN / 60

            # 双轴绘制降雨
            ax_rain = ax.twinx()
            ax_rain.bar(time_hours, rain, width=TIME_STEP_MIN/60/1.5,
                       alpha=0.15, color='steelblue')
            ax_rain.set_ylim(0, max(rain.max() * 1.3, 10))

            ax.plot(time_hours, water_true, 'k-', linewidth=1.5, label='SWMM', alpha=0.7)
            ax.plot(time_hours, water_pred, color=cfg['color'], linewidth=2, label=cfg['name'])

            ax.set_xlim(0, 24)
            ax.grid(True, alpha=0.2)

            rmse = all_evaluations[model_type][rp]['summary']['RMSE_mean']
            r2 = all_evaluations[model_type][rp]['summary']['R2_mean']

            if i == 0:
                ax.set_title(f'{rp}年一遇\nRMSE={rmse:.4f} R²={r2:.3f}', fontsize=10, fontweight='bold')
            else:
                ax.set_title(f'RMSE={rmse:.4f} R²={r2:.3f}', fontsize=9)

            if i == n_models - 1:
                ax.set_xlabel('时间 (h)', fontsize=8)

    plt.tight_layout()

    save_path = os.path.join(OUTPUT_ROOT, 'comprehensive_comparison_matrix.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  综合对比矩阵已保存: {save_path}")


# ======================== 第六阶段：保存报告 ========================
def save_experiment_report(all_evaluations, train_results, test_sets, device):
    """保存完整的实验报告"""
    print("\n" + "=" * 60)
    print("  第六阶段：生成实验报告")
    print("=" * 60)

    # ---- JSON 详细报告 ----
    report = {
        'experiment_name': '极端暴雨外推实验',
        'timestamp': TIMESTAMP,
        'device': device,
        'config': {
            'n_train_events': N_TRAIN_EVENTS,
            'train_max_return_period': TRAIN_MAX_RETURN_PERIOD,
            'test_return_periods': EXTREME_RETURN_PERIODS,
            'n_test_per_rp': N_TEST_PER_RP,
            'seq_length': SEQ_LENGTH,
            'time_step_min': TIME_STEP_MIN,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
        },
        'models': {},
        'test_sets_stats': {str(rp): test_sets[rp]['stats'] for rp in EXTREME_RETURN_PERIODS},
    }

    for cfg in MODEL_CONFIGS:
        model_type = cfg['type']
        model_eval = all_evaluations[model_type]

        report['models'][cfg['name']] = {
            'type': model_type,
            'best_val_loss': next(r['best_val_loss'] for r in train_results if r['config']['type'] == model_type),
            'extrapolation_results': {
                str(rp): model_eval[rp]['summary'] for rp in EXTREME_RETURN_PERIODS
            }
        }

    json_path = os.path.join(OUTPUT_ROOT, 'experiment_report.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  详细报告 (JSON): {json_path}")

    # ---- CSV 汇总表 ----
    csv_rows = []
    for cfg in MODEL_CONFIGS:
        model_type = cfg['type']
        for rp in EXTREME_RETURN_PERIODS:
            s = all_evaluations[model_type][rp]['summary']
            csv_rows.append({
                '模型': cfg['name'],
                '重现期(年)': rp,
                'RMSE均值': f"{s['RMSE_mean']:.6f}",
                'RMSE标准差': f"{s['RMSE_std']:.6f}",
                'MAE均值': f"{s['MAE_mean']:.6f}",
                'MAE标准差': f"{s['MAE_std']:.6f}",
                'MAPE均值(%)': f"{s['MAPE_mean']:.2f}",
                'MAPE标准差(%)': f"{s['MAPE_std']:.2f}",
                'R²均值': f"{s['R2_mean']:.6f}",
                'R²标准差': f"{s['R2_std']:.6f}",
                '峰值误差均值(m)': f"{s['PeakError_mean']:.6f}",
                '峰值相对误差均值(%)': f"{s['PeakRelError_mean']:.2f}",
                '最大绝对误差均值(m)': f"{s['MaxAbsError_mean']:.6f}",
            })

    df = pd.DataFrame(csv_rows)
    csv_path = os.path.join(OUTPUT_ROOT, 'extrapolation_metrics_summary.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  指标汇总表 (CSV): {csv_path}")

    # ---- 打印控制台摘要 ----
    print("\n" + "=" * 70)
    print("  极端暴雨外推实验 - 结果摘要")
    print("=" * 70)
    print(f"\n训练配置: {N_TRAIN_EVENTS} 个事件，重现期 ≤ {TRAIN_MAX_RETURN_PERIOD}年")
    print(f"测试配置: 每个极端重现期 {N_TEST_PER_RP} 个事件\n")

    header = f"{'模型':<22}" + "".join(f"{'T={}yr'.format(rp):>14}" for rp in EXTREME_RETURN_PERIODS)
    print(f"{'':>22}{'RMSE (m) 随重现期变化':>56}")
    print(header)
    print("-" * (22 + 14 * len(EXTREME_RETURN_PERIODS)))
    for cfg in MODEL_CONFIGS:
        model_type = cfg['type']
        vals = [f"{all_evaluations[model_type][rp]['summary']['RMSE_mean']:.4f}" for rp in EXTREME_RETURN_PERIODS]
        print(f"{cfg['name']:<22}" + "".join(f"{v:>14}" for v in vals))

    print(f"\n{'':>22}{'R² 随重现期变化':>56}")
    print(header)
    print("-" * (22 + 14 * len(EXTREME_RETURN_PERIODS)))
    for cfg in MODEL_CONFIGS:
        model_type = cfg['type']
        vals = [f"{all_evaluations[model_type][rp]['summary']['R2_mean']:.3f}" for rp in EXTREME_RETURN_PERIODS]
        print(f"{cfg['name']:<22}" + "".join(f"{v:>14}" for v in vals))

    return json_path, csv_path


# ======================== 主函数 ========================
def main():
    """主函数：运行完整的极端暴雨外推实验"""
    print("=" * 70)
    print("  极端暴雨外推实验")
    print("  Extreme Rainfall Extrapolation Experiment")
    print("=" * 70)
    print(f"\n实验时间: {TIMESTAMP}")
    print(f"核心问题: 模型在 ≤{TRAIN_MAX_RETURN_PERIOD}年重现期数据上训练，")
    print(f"          能否准确预测 {EXTREME_RETURN_PERIODS} 年重现期的极端暴雨？")

    # 设备检测
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"\n计算设备: GPU - {torch.cuda.get_device_name(0)}")
    else:
        print("\n计算设备: CPU (警告：训练会较慢)")

    # 创建输出目录
    create_output_dir()

    try:
        # --- 第一阶段：生成训练数据 ---
        data = generate_training_data(device)

        # --- 第二阶段：训练模型 ---
        train_results = train_all_models(data, device)

        # --- 第三阶段：生成极端测试集 ---
        test_sets = generate_extreme_test_sets()

        # --- 第四阶段：评估 ---
        all_evaluations = evaluate_all_models(train_results, test_sets, device)

        # --- 第五阶段：可视化 ---
        print("\n" + "=" * 60)
        print("  第五阶段：可视化分析")
        print("=" * 60)
        visualize_training_curves(train_results)
        visualize_extrapolation_heatmap(all_evaluations)
        visualize_degradation_curves(all_evaluations)
        visualize_extreme_events(all_evaluations, test_sets)
        visualize_comprehensive_comparison(all_evaluations, test_sets)

        # --- 第六阶段：报告 ---
        save_experiment_report(all_evaluations, train_results, test_sets, device)

        print("\n" + "=" * 60)
        print(f"  [SUCCESS] 实验完成！所有结果已保存至: {OUTPUT_ROOT}")
        print("=" * 60)

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"  [FAILED] 实验失败: {e}")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
