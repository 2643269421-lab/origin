#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一鲁棒性实验脚本 (Robust Experiment)
=========================================
响应审稿意见的完整实验，修复以下问题：
1. 增加随机种子至10个（统计显著性）
2. 添加 RandomMaskAttentionLSTM 对照实验（排除过拟合替代假设）
3. 统一所有模型（含消融变体）的训练条件：200场×200轮
4. 添加 Wilcoxon 秩和检验
5. 四维评估：精度、极端外推、物理一致性、空间泛化

运行方式:
    python robust_experiment.py

预计时间: CPU模式下约 6-10 小时（10种子 × 8模型变体 × 200轮）
"""

import sys, os, json, time, io, copy
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# Force UTF-8 on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

from dataset import SWMMDataset
from registry import create_model
from physics_loss import PhysicallyConsistentLoss
from swmm.simulator import SWMMSimulator
# 导入模型模块以触发@register_model注册
import lstm, gru, attention

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================================================================
# Configuration
# ================================================================
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
# 使用固定目录以支持断点续跑
OUTPUT_ROOT = os.path.join('output', 'robust_final')
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# --- 核心参数 ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_TRAIN = 200           # 训练事件数（与论文一致）
TRAIN_MAX_RP = 10       # 训练集最大重现期
EXTREME_RPS = [20, 30, 50, 100]  # 极端测试重现期
N_TEST_PER_RP = 15      # 每个重现期的测试事件数
SEQ_LEN = 288           # 序列长度（24h × 12步/h）
DT = 5                  # 时间步长（分钟）
EPOCHS = 200            # 统一训练轮数
BS = 32                 # batch size
LR = 0.001              # 学习率
PATIENCE = 20           # 早停patience

# --- 10个随机种子 ---
N_SEEDS = 10
SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]

# --- 模型配置（统一训练条件） ---
MODEL_CONFIGS = {
    'LSTM': {
        'model_type': 'SimpleLSTM',
        'loss_type': 'mse',
        'lambda_smooth': 0.0, 'lambda_peak': 0.0,
    },
    'GRU': {
        'model_type': 'SimpleGRU',
        'loss_type': 'mse',
        'lambda_smooth': 0.0, 'lambda_peak': 0.0,
    },
    'AttentionLSTM': {
        'model_type': 'AttentionLSTM',
        'loss_type': 'mse',
        'lambda_smooth': 0.0, 'lambda_peak': 0.0,
    },
    'RandomMaskLSTM': {
        'model_type': 'RandomMaskAttentionLSTM',
        'loss_type': 'mse',
        'lambda_smooth': 0.0, 'lambda_peak': 0.0,
    },
    'CA-LSTM': {
        'model_type': 'CausalAttentionLSTM',
        'loss_type': 'mse',
        'lambda_smooth': 0.0, 'lambda_peak': 0.0,
    },
    'CA+SmoothOnly': {
        'model_type': 'CausalAttentionLSTM',
        'loss_type': 'physically_consistent',
        'lambda_smooth': 0.01, 'lambda_peak': 0.0,
    },
    'CA+PeakOnly': {
        'model_type': 'CausalAttentionLSTM',
        'loss_type': 'physically_consistent',
        'lambda_smooth': 0.0, 'lambda_peak': 0.05,
    },
    'PCCA-LSTM': {
        'model_type': 'PCCA-LSTM',
        'loss_type': 'physically_consistent',
        'lambda_smooth': 0.01, 'lambda_peak': 0.05,
    },
}

# 多节点评估
EVAL_NODES = ['SN_001', 'SN_017', 'SN_049']

print(f"{'='*60}")
print(f"  统一鲁棒性实验 (Robust Experiment)")
print(f"{'='*60}")
print(f"  输出目录: {OUTPUT_ROOT}")
print(f"  设备: {DEVICE}")
print(f"  训练: {N_TRAIN}场 × {EPOCHS}轮 (重现期≤{TRAIN_MAX_RP}年)")
print(f"  种子数: {N_SEEDS}")
print(f"  模型数: {len(MODEL_CONFIGS)}")
print(f"  预计训练次数: {N_SEEDS * len(MODEL_CONFIGS)} = {N_SEEDS * len(MODEL_CONFIGS)}")
print(f"{'='*60}\n")

# ================================================================
# Evaluation Metrics
# ================================================================
def compute_metrics(pred, target, return_period=None):
    """计算完整的四维评估指标"""
    pred = np.array(pred).flatten()
    target = np.array(target).flatten()

    # 数值精度
    rmse = np.sqrt(np.mean((pred - target) ** 2))
    mae = np.mean(np.abs(pred - target))
    mape = np.mean(np.abs((pred - target) / (target + 1e-10))) * 100
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-10)

    # 峰值误差
    peak_error = pred[np.argmax(target)] - np.max(target)

    # 物理一致性 — DASR (方向一致率)
    pred_diff = np.diff(pred)
    target_diff = np.diff(target)
    # 排除变化极小的时间步 (噪声)
    significant = np.abs(target_diff) > 1e-6
    if np.sum(significant) > 0:
        same_direction = np.sign(pred_diff[significant]) == np.sign(target_diff[significant])
        dasr = np.mean(same_direction) * 100
    else:
        dasr = 50.0  # 无显著变化时默认50%

    # 峰值索引误差
    peak_idx_error = abs(np.argmax(pred) - np.argmax(target))

    # 径流响应比
    water_range = np.max(target) - np.min(target)
    nrmse = rmse / (water_range + 1e-10)

    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'PeakError': peak_error,
        'DASR': dasr,
        'PeakIdxError': peak_idx_error,
        'NRMSE': nrmse,
    }


# ================================================================
# Training Function (unified for all models)
# ================================================================
def train_model(model_name, config, dataset, seed, save_dir):
    """统一训练函数，所有模型相同条件"""
    model_type = config['model_type']
    loss_type = config['loss_type']
    lambda_smooth = config['lambda_smooth']
    lambda_peak = config['lambda_peak']

    # 断点续跑：如果模型已存在则跳过
    model_path = os.path.join(save_dir, f'{model_name}_seed{seed}.pth')
    if os.path.exists(model_path):
        print(f'    模型已存在，跳过训练: {model_path}')
        model = create_model(model_type, input_size=1, hidden_size=128,
                             num_layers=2, output_size=1, dropout=0.3)
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        model = model.to(DEVICE)
        model.eval()
        info = checkpoint.get('train_info', {'epochs': 0, 'best_val_loss': float('nan')})
        return model, model_path, info

    torch.manual_seed(seed)
    np.random.seed(seed)

    # 创建模型
    model = create_model(model_type, input_size=1, hidden_size=128,
                         num_layers=2, output_size=1, dropout=0.3)
    model = model.to(DEVICE)

    # 数据划分
    n = len(dataset)
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)
    test_size = n - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_ds, batch_size=BS, shuffle=False)

    # 损失函数
    if loss_type == 'physically_consistent':
        criterion = PhysicallyConsistentLoss(
            lambda_smooth=lambda_smooth,
            lambda_peak=lambda_peak,
            lambda_mass=0.0,
        )
    else:
        criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

    # 训练
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        # Train
        model.train()
        epoch_loss = 0
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            if loss_type == 'physically_consistent':
                loss, _ = criterion(output, target, rainfall=data,
                                    rain_scaler=dataset.rain_scaler,
                                    water_scaler=dataset.water_scaler,
                                    return_components=True)
            else:
                loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                if loss_type == 'physically_consistent':
                    vloss, _ = criterion(output, target, rainfall=data,
                                         rain_scaler=dataset.rain_scaler,
                                         water_scaler=dataset.water_scaler,
                                         return_components=True)
                else:
                    vloss = criterion(output, target)
                val_loss += vloss.item()
        val_losses.append(val_loss / len(val_loader))

        scheduler.step(val_losses[-1])

        # Early stopping
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    # 恢复最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)

    # 保存
    model_path = os.path.join(save_dir, f'{model_name}_seed{seed}.pth')
    train_info = {
        'epochs': epoch + 1,
        'best_val_loss': best_val_loss,
        'loss_type': loss_type,
        'lambda_smooth': lambda_smooth,
        'lambda_peak': lambda_peak,
        'seed': seed,
    }
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': model_type,
        'model_params': {'input_size': 1, 'hidden_size': 128,
                         'num_layers': 2, 'output_size': 1},
        'rain_scaler': dataset.rain_scaler,
        'water_scaler': dataset.water_scaler,
        'seq_length': SEQ_LEN,
        'n_events': N_TRAIN,
        'time_step_min': DT,
        'seed': seed,
        'epochs_trained': epoch + 1,
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'loss_type': loss_type,
        'lambda_smooth': lambda_smooth,
        'lambda_peak': lambda_peak,
        'train_info': train_info,
    }, model_path)

    tv_gap = train_losses[-1] - val_losses[-1] if len(train_losses) > 0 else 0

    return model, model_path, {'tv_gap': tv_gap, 'best_val_loss': best_val_loss,
                                'epochs': epoch + 1}

# ================================================================
# Evaluation on Extreme Events
# ================================================================
def evaluate_on_extreme(model, dataset_extreme, water_scaler):
    """在极端事件数据集上评估模型"""
    model.eval()
    loader = DataLoader(dataset_extreme, batch_size=BS, shuffle=False)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            # 反标准化
            pred_np = output.cpu().numpy().reshape(-1, 1)
            target_np = target.cpu().numpy().reshape(-1, 1)
            pred_orig = water_scaler.inverse_transform(pred_np)
            target_orig = water_scaler.inverse_transform(target_np)
            all_preds.append(pred_orig.flatten())
            all_targets.append(target_orig.flatten())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    return compute_metrics(preds, targets)


def evaluate_per_event(model, dataset_extreme, water_scaler):
    """逐事件评估，返回每个事件的指标列表"""
    model.eval()
    metrics_list = []

    for i in range(len(dataset_extreme)):
        data, target = dataset_extreme[i]
        data = data.unsqueeze(0).to(DEVICE)
        target = target.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(data)

        pred_np = output.cpu().numpy().reshape(-1, 1)
        target_np = target.cpu().numpy().reshape(-1, 1)
        pred_orig = water_scaler.inverse_transform(pred_np).flatten()
        target_orig = water_scaler.inverse_transform(target_np).flatten()

        m = compute_metrics(pred_orig, target_orig)
        metrics_list.append(m)

    return metrics_list


# ================================================================
# Real Rainfall Evaluation
# ================================================================
def evaluate_real_rainfall(model, rain_scaler, water_scaler):
    """在实测降雨事件上评估"""
    data_dir = os.path.join(PROJECT_ROOT, 'Actual Rainfall_Water Level')
    if not os.path.exists(data_dir):
        print("  [WARN] 实测降雨目录不存在，跳过")
        return []

    files = sorted([f for f in os.listdir(data_dir)
                    if f.endswith('.xlsx') and not f.startswith('~')])
    if not files:
        print("  [WARN] 无实测降雨文件，跳过")
        return []

    model.eval()
    metrics_list = []

    for fname in files:
        try:
            df = pd.read_excel(os.path.join(data_dir, fname))
            rain_col = water_col = None
            for c in df.columns:
                cs = str(c).lower()
                if 'rain' in cs or '降雨' in str(c):
                    rain_col = c
                if 'water' in cs or '水位' in str(c) or 'depth' in cs:
                    water_col = c

            if rain_col is None or water_col is None:
                continue

            rain_raw = df[rain_col].values.astype(np.float32)
            water_raw = df[water_col].values.astype(np.float32)

            # 补齐/截断到SEQ_LEN
            if len(rain_raw) < SEQ_LEN:
                rain_raw = np.pad(rain_raw, (0, SEQ_LEN - len(rain_raw)))
                water_raw = np.pad(water_raw, (0, SEQ_LEN - len(water_raw)))
            else:
                rain_raw = rain_raw[:SEQ_LEN]
                water_raw = water_raw[:SEQ_LEN]

            # 标准化
            rain_scaled = rain_scaler.transform(rain_raw.reshape(-1, 1)).reshape(SEQ_LEN)
            rain_tensor = torch.FloatTensor(rain_scaled).unsqueeze(0).unsqueeze(-1).to(DEVICE)

            with torch.no_grad():
                pred_scaled = model(rain_tensor)

            pred_orig = water_scaler.inverse_transform(
                pred_scaled.cpu().numpy().reshape(-1, 1)
            ).flatten()

            m = compute_metrics(pred_orig, water_raw)
            metrics_list.append(m)

        except Exception as e:
            continue

    return metrics_list


# ================================================================
# Statistical Tests
# ================================================================
def wilcoxon_test(metrics_a, metrics_b, metric_name='RMSE'):
    """Wilcoxon 符号秩检验"""
    a = [m[metric_name] for m in metrics_a]
    b = [m[metric_name] for m in metrics_b]
    if len(a) < 5 or len(b) < 5:
        return {'statistic': np.nan, 'p_value': np.nan, 'n': min(len(a), len(b))}
    try:
        stat, p = scipy_stats.wilcoxon(a, b)
        return {'statistic': stat, 'p_value': p, 'n': len(a)}
    except Exception:
        return {'statistic': np.nan, 'p_value': np.nan, 'n': len(a)}


def mann_whitney_test(metrics_a, metrics_b, metric_name='RMSE'):
    """Mann-Whitney U 检验 (独立样本)"""
    a = [m[metric_name] for m in metrics_a]
    b = [m[metric_name] for m in metrics_b]
    if len(a) < 3 or len(b) < 3:
        return {'statistic': np.nan, 'p_value': np.nan}
    try:
        stat, p = scipy_stats.mannwhitneyu(a, b, alternative='two-sided')
        return {'statistic': stat, 'p_value': p}
    except Exception:
        return {'statistic': np.nan, 'p_value': np.nan}

# ================================================================
# MAIN EXPERIMENT LOOP
# ================================================================
if __name__ == '__main__':
    results_all = {}  # {model_name: {seed: {rp: metrics}}}
    results_real = {}  # {model_name: {seed: [metrics_per_event]}}
    training_info = {}  # {model_name: {seed: info}}

    # ---- Step 1: 加载预缓存训练数据（跳过 SWMM 模拟） ----
    cache_dir = os.path.join(OUTPUT_ROOT, 'data_cache')
    train_cache = os.path.join(cache_dir, 'train_data.npz')
    if os.path.exists(train_cache):
        print("\n[Step 1] 加载缓存的训练数据...")
        data = np.load(train_cache, allow_pickle=True)
        from sklearn.preprocessing import MinMaxScaler

        # 用原始数据重新 fit scaler（避免版本兼容问题）
        rain_2d = data['rainfall_events'].reshape(-1, 1)
        water_2d = data['water_level_events'].reshape(-1, 1)
        rain_scaler = MinMaxScaler().fit(rain_2d)
        water_scaler = MinMaxScaler().fit(water_2d)

        # 创建数据集对象
        train_dataset = SWMMDataset.__new__(SWMMDataset)
        train_dataset.X = data['X']
        train_dataset.y = data['y']
        train_dataset.rainfall_events = data['rainfall_events']
        train_dataset.water_level_events = data['water_level_events']
        train_dataset.rain_scaler = rain_scaler
        train_dataset.water_scaler = water_scaler
        train_dataset.rainfall_scaled = train_dataset.X
        train_dataset.water_level_scaled = train_dataset.y
        print(f"  训练集加载完成: {len(train_dataset.X)} 个样本 (从缓存)")
    else:
        print("\n[Step 1] 生成训练数据集...")
        t0 = time.time()
        train_dataset = SWMMDataset(
            n_events=N_TRAIN,
            seq_length=SEQ_LEN,
            time_step_min=DT,
            max_return_period=TRAIN_MAX_RP,
        )
        print(f"  训练集生成完成: {len(train_dataset)}场, 用时{time.time()-t0:.1f}s")

    # ---- Step 2: 加载预缓存测试数据 ----
    print("\n[Step 2] 加载极端测试数据...")
    extreme_datasets = {}
    for rp in EXTREME_RPS:
        cache_path = os.path.join(cache_dir, f'extreme_T{rp}.npz')
        if os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=True)
            ds = SWMMDataset.__new__(SWMMDataset)
            ds.X = data['X']
            ds.y = data['y']
            ds.rainfall_events = data['rainfall_events']
            ds.water_level_events = data['water_level_events']
            ds.rain_scaler = train_dataset.rain_scaler
            ds.water_scaler = train_dataset.water_scaler
            ds.rainfall_scaled = ds.X
            ds.water_level_scaled = ds.y
            extreme_datasets[rp] = ds
            print(f"  T={rp}年: {len(ds.X)}场 (从缓存)")
        else:
            t0 = time.time()
            ds = SWMMDataset(
                n_events=N_TEST_PER_RP,
                seq_length=SEQ_LEN,
                time_step_min=DT,
                return_period=rp,
            )
            # 使用训练集的scaler进行标准化
            ds.rain_scaler = train_dataset.rain_scaler
            ds.water_scaler = train_dataset.water_scaler
            # 重新标准化
            ds.rainfall_scaled = train_dataset.rain_scaler.transform(
                ds.rainfall_events.reshape(-1, 1)
            ).reshape(ds.rainfall_events.shape)
            ds.water_level_scaled = train_dataset.water_scaler.transform(
                ds.water_level_events.reshape(-1, 1)
            ).reshape(ds.water_level_events.shape)
            ds.X = ds.rainfall_scaled
            ds.y = ds.water_level_scaled
            extreme_datasets[rp] = ds
            print(f"  T={rp}年: {len(ds)}场, 用时{time.time()-t0:.1f}s")

    # ---- Step 3: 训练和评估 ----
    print(f"\n[Step 3] 开始训练 ({N_SEEDS}种子 × {len(MODEL_CONFIGS)}模型)...")
    model_dir = os.path.join(OUTPUT_ROOT, 'models')
    os.makedirs(model_dir, exist_ok=True)

    total_runs = N_SEEDS * len(MODEL_CONFIGS)
    run_count = 0

    for seed in SEEDS:
        for model_name, config in MODEL_CONFIGS.items():
            run_count += 1
            print(f"\n  [{run_count}/{total_runs}] {model_name} (seed={seed})")
            t0 = time.time()

            # 训练
            model, model_path, info = train_model(
                model_name, config, train_dataset, seed, model_dir
            )
            training_info.setdefault(model_name, {})[seed] = info
            elapsed = time.time() - t0
            print(f"    训练完成: {info['epochs']}轮, val_loss={info['best_val_loss']:.6f}, "
                  f"用时{elapsed:.1f}s")

            # 评估: 极端外推
            if model_name not in results_all:
                results_all[model_name] = {}
            results_all[model_name][seed] = {}

            for rp in EXTREME_RPS:
                metrics = evaluate_per_event(
                    model, extreme_datasets[rp], train_dataset.water_scaler
                )
                results_all[model_name][seed][rp] = metrics

            # 评估: 实测降雨
            real_metrics = evaluate_real_rainfall(
                model, train_dataset.rain_scaler, train_dataset.water_scaler
            )
            results_real.setdefault(model_name, {})[seed] = real_metrics

            print(f"    T=100 RMSE={np.mean([m['RMSE'] for m in results_all[model_name][seed][100]]):.4f}, "
                  f"DASR={np.mean([m['DASR'] for m in results_all[model_name][seed][100]]):.1f}%")
            if real_metrics:
                print(f"    实测 RMSE={np.mean([m['RMSE'] for m in real_metrics]):.4f}, "
                      f"DASR={np.mean([m['DASR'] for m in real_metrics]):.1f}%")

    # ================================================================
    # Step 4: 汇总统计 + 统计检验
    # ================================================================
    print(f"\n[Step 4] 汇总结果与统计检验...")

    # 汇总表: 极端外推 (T=100)
    summary_rows = []
    for model_name in MODEL_CONFIGS.keys():
        all_event_metrics = []
        for seed in SEEDS:
            if seed in results_all.get(model_name, {}):
                all_event_metrics.extend(results_all[model_name][seed].get(100, []))

        if not all_event_metrics:
            continue

        row = {
            'Model': model_name,
            'RMSE_mean': np.mean([m['RMSE'] for m in all_event_metrics]),
            'RMSE_std': np.std([m['RMSE'] for m in all_event_metrics]),
            'MAE_mean': np.mean([m['MAE'] for m in all_event_metrics]),
            'MAE_std': np.std([m['MAE'] for m in all_event_metrics]),
            'MAPE_mean': np.mean([m['MAPE'] for m in all_event_metrics]),
            'R2_mean': np.mean([m['R2'] for m in all_event_metrics]),
            'PeakError_mean': np.mean([m['PeakError'] for m in all_event_metrics]),
            'DASR_mean': np.mean([m['DASR'] for m in all_event_metrics]),
            'DASR_std': np.std([m['DASR'] for m in all_event_metrics]),
            'N_events': len(all_event_metrics),
        }
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_ROOT, 'table3_extreme_T100.csv')
    df_summary.to_csv(summary_path, index=False)
    print(f"  T=100汇总表已保存: {summary_path}")
    print(df_summary[['Model', 'RMSE_mean', 'RMSE_std', 'DASR_mean', 'DASR_std']].to_string())

    # 汇总表: 多重现期
    multi_rp_rows = []
    for model_name in MODEL_CONFIGS.keys():
        for rp in EXTREME_RPS:
            all_event_metrics = []
            for seed in SEEDS:
                if seed in results_all.get(model_name, {}):
                    all_event_metrics.extend(results_all[model_name][seed].get(rp, []))
            if not all_event_metrics:
                continue
            multi_rp_rows.append({
                'Model': model_name,
                'ReturnPeriod': rp,
                'RMSE_mean': np.mean([m['RMSE'] for m in all_event_metrics]),
                'RMSE_std': np.std([m['RMSE'] for m in all_event_metrics]),
                'DASR_mean': np.mean([m['DASR'] for m in all_event_metrics]),
                'DASR_std': np.std([m['DASR'] for m in all_event_metrics]),
                'PeakError_mean': np.mean([m['PeakError'] for m in all_event_metrics]),
                'N': len(all_event_metrics),
            })
    df_multi_rp = pd.DataFrame(multi_rp_rows)
    df_multi_rp.to_csv(os.path.join(OUTPUT_ROOT, 'table_multi_rp.csv'), index=False)

    # 汇总表: 实测降雨
    real_rows = []
    for model_name in MODEL_CONFIGS.keys():
        all_real_metrics = []
        for seed in SEEDS:
            if seed in results_real.get(model_name, {}):
                all_real_metrics.extend(results_real[model_name][seed])
        if not all_real_metrics:
            continue
        real_rows.append({
            'Model': model_name,
            'RMSE_mean': np.mean([m['RMSE'] for m in all_real_metrics]),
            'RMSE_std': np.std([m['RMSE'] for m in all_real_metrics]),
            'MAE_mean': np.mean([m['MAE'] for m in all_real_metrics]),
            'DASR_mean': np.mean([m['DASR'] for m in all_real_metrics]),
            'DASR_std': np.std([m['DASR'] for m in all_real_metrics]),
            'PeakError_mean': np.mean([m['PeakError'] for m in all_real_metrics]),
            'PeakError_std': np.std([m['PeakError'] for m in all_real_metrics]),
            'N': len(all_real_metrics),
        })
    df_real = pd.DataFrame(real_rows)
    real_path = os.path.join(OUTPUT_ROOT, 'table4_real_rainfall.csv')
    df_real.to_csv(real_path, index=False)
    print(f"\n  实测降雨汇总表已保存: {real_path}")
    if not df_real.empty:
        print(df_real[['Model', 'RMSE_mean', 'RMSE_std', 'DASR_mean', 'DASR_std']].to_string())

    # ---- 统计检验 ----
    print(f"\n  统计检验 (Wilcoxon signed-rank):")
    stat_tests = []

    # H1: AttentionLSTM vs LSTM
    for metric_key in ['RMSE', 'DASR']:
        # 收集每个种子的平均指标
        a_seeds = []  # AttentionLSTM
        b_seeds = []  # LSTM
        for seed in SEEDS:
            if seed in results_all.get('AttentionLSTM', {}) and \
               seed in results_all.get('LSTM', {}):
                a_metrics = results_all['AttentionLSTM'][seed].get(100, [])
                b_metrics = results_all['LSTM'][seed].get(100, [])
                if a_metrics and b_metrics:
                    a_seeds.append(np.mean([m[metric_key] for m in a_metrics]))
                    b_seeds.append(np.mean([m[metric_key] for m in b_metrics]))

        if len(a_seeds) >= 5:
            try:
                stat, p = scipy_stats.wilcoxon(a_seeds, b_seeds)
                stat_tests.append({
                    'Comparison': f'AttentionLSTM vs LSTM ({metric_key})',
                    'Statistic': stat, 'p_value': p, 'N_pairs': len(a_seeds),
                    'Significant_0.05': p < 0.05
                })
                print(f"    AttentionLSTM vs LSTM [{metric_key}]: W={stat:.1f}, p={p:.4f} "
                      f"{'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'}")
            except Exception as e:
                print(f"    AttentionLSTM vs LSTM [{metric_key}]: 检验失败 ({e})")

    # H1: RandomMask vs LSTM, RandomMask vs CA-LSTM
    for pair_name, (model_a, model_b) in [
        ('RandomMask vs LSTM', ('RandomMaskLSTM', 'LSTM')),
        ('RandomMask vs CA-LSTM', ('RandomMaskLSTM', 'CA-LSTM')),
        ('CA-LSTM vs LSTM', ('CA-LSTM', 'LSTM')),
        ('PCCA-LSTM vs CA-LSTM', ('PCCA-LSTM', 'CA-LSTM')),
    ]:
        for metric_key in ['RMSE', 'DASR']:
            a_seeds, b_seeds = [], []
            for seed in SEEDS:
                if seed in results_all.get(model_a, {}) and \
                   seed in results_all.get(model_b, {}):
                    a_m = results_all[model_a][seed].get(100, [])
                    b_m = results_all[model_b][seed].get(100, [])
                    if a_m and b_m:
                        a_seeds.append(np.mean([m[metric_key] for m in a_m]))
                        b_seeds.append(np.mean([m[metric_key] for m in b_m]))
            if len(a_seeds) >= 5:
                try:
                    stat, p = scipy_stats.wilcoxon(a_seeds, b_seeds)
                    stat_tests.append({
                        'Comparison': f'{pair_name} ({metric_key})',
                        'Statistic': stat, 'p_value': p, 'N_pairs': len(a_seeds),
                        'Significant_0.05': p < 0.05
                    })
                    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                    print(f"    {pair_name} [{metric_key}]: W={stat:.1f}, p={p:.4f} {sig}")
                except Exception:
                    pass

    df_stats = pd.DataFrame(stat_tests)
    stats_path = os.path.join(OUTPUT_ROOT, 'statistical_tests.csv')
    df_stats.to_csv(stats_path, index=False)
    print(f"\n  统计检验结果已保存: {stats_path}")

    # ================================================================
    # Step 5: 保存完整结果为JSON (供后续论文更新使用)
    # ================================================================
    print(f"\n[Step 5] 保存完整结果...")

    # 将numpy类型转换为Python原生类型
    def to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        return obj

    # 跨种子聚合的摘要
    final_summary = {
        'config': {
            'n_seeds': N_SEEDS,
            'seeds': SEEDS,
            'n_train': N_TRAIN,
            'epochs': EPOCHS,
            'train_max_rp': TRAIN_MAX_RP,
            'extreme_rps': EXTREME_RPS,
            'device': DEVICE,
            'timestamp': TIMESTAMP,
        },
        'extreme_T100': to_serializable(summary_rows),
        'multi_rp': to_serializable(multi_rp_rows),
        'real_rainfall': to_serializable(real_rows),
        'statistical_tests': to_serializable(stat_tests),
        'training_info': to_serializable(training_info),
    }

    json_path = os.path.join(OUTPUT_ROOT, 'results_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)
    print(f"  完整结果已保存: {json_path}")

    # ================================================================
    # Step 6: 打印核心结论
    # ================================================================
    print(f"\n{'='*60}")
    print("  核心结论摘要")
    print(f"{'='*60}")

    if not df_summary.empty:
        lstm_rmse = df_summary[df_summary.Model == 'LSTM']['RMSE_mean'].values
        attn_rmse = df_summary[df_summary.Model == 'AttentionLSTM']['RMSE_mean'].values
        rand_rmse = df_summary[df_summary.Model == 'RandomMaskLSTM']['RMSE_mean'].values
        ca_rmse = df_summary[df_summary.Model == 'CA-LSTM']['RMSE_mean'].values
        pcca_rmse = df_summary[df_summary.Model == 'PCCA-LSTM']['RMSE_mean'].values

        if len(lstm_rmse) > 0 and len(attn_rmse) > 0:
            penalty = (attn_rmse[0] - lstm_rmse[0]) / lstm_rmse[0] * 100
            print(f"\n  H1 注意力惩罚: AttentionLSTM RMSE比LSTM高 {penalty:+.1f}%")
        if len(rand_rmse) > 0 and len(lstm_rmse) > 0:
            rand_vs_lstm = (rand_rmse[0] - lstm_rmse[0]) / lstm_rmse[0] * 100
            print(f"     RandomMask RMSE比LSTM高 {rand_vs_lstm:+.1f}% (对照)")
        if len(ca_rmse) > 0 and len(lstm_rmse) > 0:
            ca_improve = (lstm_rmse[0] - ca_rmse[0]) / lstm_rmse[0] * 100
            print(f"\n  H2 因果约束: CA-LSTM RMSE比LSTM降低 {ca_improve:.1f}%")
        if len(pcca_rmse) > 0 and len(ca_rmse) > 0:
            loss_improve = (ca_rmse[0] - pcca_rmse[0]) / ca_rmse[0] * 100
            print(f"     损失级额外降低: {loss_improve:.1f}%")
            if ca_improve > 0 and loss_improve > 0:
                ratio = ca_improve / loss_improve
                print(f"     架构/损失贡献比: {ratio:.1f}x")

        lstm_dasr = df_summary[df_summary.Model == 'LSTM']['DASR_mean'].values
        ca_dasr = df_summary[df_summary.Model == 'CA-LSTM']['DASR_mean'].values
        if len(lstm_dasr) > 0 and len(ca_dasr) > 0:
            dasr_gain = ca_dasr[0] - lstm_dasr[0]
            print(f"\n  DASR增益: LSTM {lstm_dasr[0]:.1f}% → CA-LSTM {ca_dasr[0]:.1f}% (+{dasr_gain:.1f}pp)")

    print(f"\n{'='*60}")
    print(f"  实验完成！所有结果保存在: {OUTPUT_ROOT}")
    print(f"{'='*60}")
