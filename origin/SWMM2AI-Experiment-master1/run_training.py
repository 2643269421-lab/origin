#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
直接启动训练（跳过数据生成），main 入口
"""
import os, sys, io, json, time, copy
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

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
import lstm, gru, attention

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================================================================
# Configuration (same as robust_experiment.py)
# ================================================================
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_ROOT = os.path.join('output', 'robust_final')
os.makedirs(OUTPUT_ROOT, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_TRAIN = 200
TRAIN_MAX_RP = 10
EXTREME_RPS = [20, 30, 50, 100]
N_TEST_PER_RP = 15
SEQ_LEN = 288
DT = 5
EPOCHS = 200
BS = 32
LR = 0.001
PATIENCE = 20

N_SEEDS = 10
SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]

MODEL_CONFIGS = {
    'LSTM':           {'model_type': 'SimpleLSTM',       'loss_type': 'mse',                 'lambda_smooth': 0.0,  'lambda_peak': 0.0},
    'GRU':            {'model_type': 'SimpleGRU',        'loss_type': 'mse',                 'lambda_smooth': 0.0,  'lambda_peak': 0.0},
    'AttentionLSTM':  {'model_type': 'AttentionLSTM',    'loss_type': 'mse',                 'lambda_smooth': 0.0,  'lambda_peak': 0.0},
    'RandomMaskLSTM': {'model_type': 'RandomMaskAttentionLSTM', 'loss_type': 'mse',           'lambda_smooth': 0.0,  'lambda_peak': 0.0},
    'CA-LSTM':        {'model_type': 'CausalAttentionLSTM', 'loss_type': 'mse',               'lambda_smooth': 0.0,  'lambda_peak': 0.0},
    'CA+SmoothOnly':  {'model_type': 'CausalAttentionLSTM', 'loss_type': 'physically_consistent', 'lambda_smooth': 0.01, 'lambda_peak': 0.0},
    'CA+PeakOnly':    {'model_type': 'CausalAttentionLSTM', 'loss_type': 'physically_consistent', 'lambda_smooth': 0.0,  'lambda_peak': 0.05},
    'PCCA-LSTM':      {'model_type': 'PCCA-LSTM',         'loss_type': 'physically_consistent', 'lambda_smooth': 0.01, 'lambda_peak': 0.05},
}

EVAL_NODES = ['SN_001', 'SN_017', 'SN_049']

# ================================================================
# Metrics (same)
# ================================================================
def compute_metrics(pred, target, return_period=None):
    pred = np.array(pred).flatten()
    target = np.array(target).flatten()
    rmse = np.sqrt(np.mean((pred - target) ** 2))
    mae = np.mean(np.abs(pred - target))
    mape = np.mean(np.abs((pred - target) / (target + 1e-10))) * 100
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-10)
    peak_idx = np.argmax(target)
    peak_error = pred[peak_idx] - target[peak_idx]

    # DASR: 方向一致率 (Direction Alignment Success Rate)
    # 预测与目标水位变化方向一致的时间步百分比
    pred_diff = np.diff(pred)
    target_diff = np.diff(target)
    # 排除变化极小的时间步
    significant = np.abs(target_diff) > 1e-6
    if np.sum(significant) > 0:
        same_dir = np.sign(pred_diff[significant]) == np.sign(target_diff[significant])
        dasr = np.mean(same_dir) * 100
    else:
        dasr = 50.0

    # 峰值索引误差
    peak_idx_error = abs(int(np.argmax(pred)) - int(np.argmax(target)))

    # NRMSE
    water_range = np.max(target) - np.min(target)
    nrmse = rmse / (water_range + 1e-10)

    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2,
            'PeakError': peak_error, 'DASR': dasr, 'PeakIdxError': peak_idx_error,
            'NRMSE': nrmse}

def evaluate_per_event(model, dataset, water_scaler):
    model.eval()
    metrics_list = []
    with torch.no_grad():
        for i in range(len(dataset)):
            data, target = dataset[i]
            data = data.unsqueeze(0).to(DEVICE)
            target = target.unsqueeze(0).to(DEVICE)
            output = model(data)
            pred_np = output.cpu().numpy().reshape(-1, 1)
            target_np = target.cpu().numpy().reshape(-1, 1)
            pred_orig = water_scaler.inverse_transform(pred_np).flatten()
            target_orig = water_scaler.inverse_transform(target_np).flatten()
            metrics_list.append(compute_metrics(pred_orig, target_orig))
    return metrics_list

def evaluate_real_rainfall(model, rain_scaler, water_scaler):
    import glob
    data_dir = os.path.join(PROJECT_ROOT, 'Actual Rainfall_Water Level')
    xlsx_files = sorted(glob.glob(os.path.join(data_dir, '*.xlsx')))
    if not xlsx_files:
        return []
    results = []
    model.eval()
    for fpath in xlsx_files:
        try:
            df = pd.read_excel(fpath)
            # 自动识别降雨和水位列
            rain_col = None
            water_col = None
            for c in df.columns:
                cs = str(c)
                if '雨强' in cs or 'rain' in cs.lower() or '降雨' in cs:
                    rain_col = c
                if 'STORM_L_01' in cs or ('水位' in cs and 'STORM' in cs):
                    water_col = c
            # 备选: 第一个含"液位"的列
            if water_col is None:
                for c in df.columns:
                    if '液位' in str(c) and 'WASTE' not in str(c):
                        water_col = c
                        break
            if rain_col is None or water_col is None:
                continue

            rain = df[rain_col].values[:SEQ_LEN].astype(np.float32)
            true_water = df[water_col].values[:SEQ_LEN].astype(np.float32)

            # 跳过全零或长度不足的事件
            if len(rain) < SEQ_LEN:
                rain = np.pad(rain, (0, SEQ_LEN - len(rain)))
                true_water = np.pad(true_water, (0, SEQ_LEN - len(true_water)))
            if np.max(rain) < 0.1:  # 几乎无降雨
                continue

            rain_scaled = rain_scaler.transform(rain.reshape(-1, 1)).reshape(-1)
            rain_tensor = torch.FloatTensor(rain_scaled).unsqueeze(0).unsqueeze(-1).to(DEVICE)
            with torch.no_grad():
                pred_scaled = model(rain_tensor)
            pred_orig = water_scaler.inverse_transform(pred_scaled.cpu().numpy().reshape(-1, 1)).flatten()
            results.append(compute_metrics(pred_orig, true_water))
        except Exception:
            continue
    return results

# ================================================================
# Training function (same)
# ================================================================
def train_model(model_name, config, train_dataset, seed, model_dir):
    # 跳过已完成的训练（断点续跑）
    save_path = os.path.join(model_dir, f'{model_name}_seed{seed}.pth')
    if os.path.exists(save_path):
        try:
            checkpoint = torch.load(save_path, map_location=DEVICE, weights_only=False)
            info = checkpoint.get('train_info', {'epochs': 0, 'best_val_loss': float('nan')})
            print(f"    [跳过] 已存在, val_loss={info['best_val_loss']:.6f}")
            model = create_model(config['model_type'], input_size=1, hidden_size=128,
                                 num_layers=2, output_size=1, dropout=0.3)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(DEVICE)
            model.eval()
            return model, save_path, info
        except Exception as e:
            print(f"    [重新训练] 读取失败: {e}")
            os.remove(save_path)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = create_model(config['model_type'], input_size=1, hidden_size=128,
                         num_layers=2, output_size=1, dropout=0.3)
    model = model.to(DEVICE)

    n = len(train_dataset)
    indices = list(range(n))
    split = int(n * 0.8)
    np.random.shuffle(indices)
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_subset = torch.utils.data.Subset(train_dataset, train_idx)
    val_subset = torch.utils.data.Subset(train_dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=BS, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BS, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    if config['loss_type'] == 'physically_consistent':
        criterion = PhysicallyConsistentLoss(lambda_smooth=config['lambda_smooth'],
                                              lambda_peak=config['lambda_peak'])
    else:
        criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            if config['loss_type'] == 'physically_consistent':
                loss, _ = criterion(output, target, rainfall=data, rain_scaler=train_dataset.rain_scaler, water_scaler=train_dataset.water_scaler, return_components=True)
            else:
                loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                if config['loss_type'] == 'physically_consistent':
                    vloss, _ = criterion(output, target, rainfall=data, rain_scaler=train_dataset.rain_scaler, water_scaler=train_dataset.water_scaler, return_components=True)
                else:
                    vloss = nn.MSELoss()(output, target)
                val_loss += vloss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    model.load_state_dict(best_state)
    save_path = os.path.join(model_dir, f'{model_name}_seed{seed}.pth')
    torch.save({'model_state_dict': best_state, 'train_info': {'epochs': epoch + 1, 'best_val_loss': best_val_loss}}, save_path)

    return model, save_path, {'epochs': epoch + 1, 'best_val_loss': best_val_loss}

# ================================================================
# MAIN
# ================================================================
if __name__ == '__main__':
    print(f"{'='*60}")
    print(f"  统一鲁棒性实验 - GPU训练模式")
    print(f"{'='*60}")
    print(f"  设备: {DEVICE}")
    print(f"  训练: {N_TRAIN}场 × {EPOCHS}轮")
    print(f"  种子: {N_SEEDS}, 模型: {len(MODEL_CONFIGS)}")
    print(f"  预计训练次数: {N_SEEDS * len(MODEL_CONFIGS)}")
    print(f"{'='*60}\n")

    # Step 1: Load cached data
    cache_dir = os.path.join(OUTPUT_ROOT, 'data_cache')
    train_cache = os.path.join(cache_dir, 'train_data.npz')

    print("[Step 1] 加载缓存训练数据...")
    data = np.load(train_cache, allow_pickle=True)
    from sklearn.preprocessing import MinMaxScaler
    rain_2d = data['rainfall_events'].reshape(-1, 1)
    water_2d = data['water_level_events'].reshape(-1, 1)
    rain_scaler = MinMaxScaler().fit(rain_2d)
    water_scaler = MinMaxScaler().fit(water_2d)

    train_dataset = SWMMDataset.__new__(SWMMDataset)
    train_dataset.X = data['X']
    train_dataset.y = data['y']
    train_dataset.rainfall_events = data['rainfall_events']
    train_dataset.water_level_events = data['water_level_events']
    train_dataset.rain_scaler = rain_scaler
    train_dataset.water_scaler = water_scaler
    train_dataset.rainfall_scaled = train_dataset.X
    train_dataset.water_level_scaled = train_dataset.y
    print(f"  训练集: {len(train_dataset.X)} 样本 (缓存)")

    # Step 2: Load extreme test data
    print("\n[Step 2] 加载极端测试数据...")
    extreme_datasets = {}
    for rp in EXTREME_RPS:
        ext_data = np.load(os.path.join(cache_dir, f'extreme_T{rp}.npz'), allow_pickle=True)
        ds = SWMMDataset.__new__(SWMMDataset)
        ds.X = ext_data['X']
        ds.y = ext_data['y']
        ds.rainfall_events = ext_data['rainfall_events']
        ds.water_level_events = ext_data['water_level_events']
        ds.rain_scaler = rain_scaler
        ds.water_scaler = water_scaler
        ds.rainfall_scaled = ds.X
        ds.water_level_scaled = ds.y
        extreme_datasets[rp] = ds
        print(f"  T={rp}年: {len(ds.X)} 样本")

    # Step 3: Train ALL models
    print(f"\n[Step 3] 开始GPU训练...")
    model_dir = os.path.join(OUTPUT_ROOT, 'models')
    os.makedirs(model_dir, exist_ok=True)

    results_all = {}
    results_real = {}
    training_info = {}

    total_runs = N_SEEDS * len(MODEL_CONFIGS)
    run_count = 0

    for seed in SEEDS:
        for model_name, config in MODEL_CONFIGS.items():
            run_count += 1
            print(f"\n  [{run_count}/{total_runs}] {model_name} seed={seed}...")
            t0 = time.time()

            try:
                model, model_path, info = train_model(model_name, config, train_dataset, seed, model_dir)
                training_info.setdefault(model_name, {})[seed] = info
                print(f"    训练: {info['epochs']}轮, val_loss={info['best_val_loss']:.6f}, {time.time()-t0:.0f}s")

                # Evaluate
                results_all.setdefault(model_name, {})[seed] = {}
                for rp in EXTREME_RPS:
                    results_all[model_name][seed][rp] = evaluate_per_event(model, extreme_datasets[rp], water_scaler)

                real_metrics = evaluate_real_rainfall(model, rain_scaler, water_scaler)
                results_real.setdefault(model_name, {})[seed] = real_metrics

                # Quick summary
                t100 = [m['RMSE'] for m in results_all[model_name][seed][100]]
                print(f"    T=100: RMSE={np.mean(t100):.4f}, DASR={np.mean([m['DASR'] for m in results_all[model_name][seed][100]]):.1f}%")

            except Exception as e:
                print(f"    ERROR: {e}")
                continue

    # Step 4: Summary tables
    print(f"\n[Step 4] 生成汇总报告...")

    summary_rows = []
    for model_name in MODEL_CONFIGS.keys():
        all_m = []
        for seed in SEEDS:
            if seed in results_all.get(model_name, {}):
                all_m.extend(results_all[model_name][seed].get(100, []))
        if all_m:
            summary_rows.append({
                'Model': model_name,
                'RMSE': np.mean([m['RMSE'] for m in all_m]),
                'RMSE_std': np.std([m['RMSE'] for m in all_m]),
                'MAE': np.mean([m['MAE'] for m in all_m]),
                'DASR': np.mean([m['DASR'] for m in all_m]),
                'DASR_std': np.std([m['DASR'] for m in all_m]),
                'PeakErr': np.mean([m['PeakError'] for m in all_m]),
                'R2': np.mean([m['R2'] for m in all_m]),
            })

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        df.to_csv(os.path.join(OUTPUT_ROOT, 'table3_extreme_T100.csv'), index=False)
        print("\n  === T=100 极端外推结果 ===")
        print(df[['Model', 'RMSE', 'RMSE_std', 'DASR', 'DASR_std']].to_string())

    # Real rainfall summary
    real_rows = []
    for model_name in MODEL_CONFIGS.keys():
        all_m = []
        for seed in SEEDS:
            if seed in results_real.get(model_name, {}):
                all_m.extend(results_real[model_name][seed])
        if all_m:
            real_rows.append({
                'Model': model_name,
                'RMSE': np.mean([m['RMSE'] for m in all_m]),
                'MAE': np.mean([m['MAE'] for m in all_m]),
                'DASR': np.mean([m['DASR'] for m in all_m]),
            })

    if real_rows:
        df_real = pd.DataFrame(real_rows)
        df_real.to_csv(os.path.join(OUTPUT_ROOT, 'table4_real_rainfall.csv'), index=False)
        print("\n  === 实测降雨结果 ===")
        print(df_real.to_string())

    # Save full results
    def to_ser(obj):
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: to_ser(v) for k, v in obj.items()}
        if isinstance(obj, list): return [to_ser(v) for v in obj]
        return obj

    json.dump(to_ser({'config': {'device': DEVICE, 'seeds': SEEDS, 'n_seeds': N_SEEDS, 'epochs': EPOCHS},
                       'extreme_T100': summary_rows, 'real_rainfall': real_rows, 'training_info': training_info}),
              open(os.path.join(OUTPUT_ROOT, 'results_summary.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  实验完成！结果: {OUTPUT_ROOT}")
    print(f"{'='*60}")
