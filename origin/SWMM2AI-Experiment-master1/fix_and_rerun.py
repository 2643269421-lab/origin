#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
补充实验脚本：修复 DASR 计算 + 重训物理约束模型 + 实测降雨评估
===============================================================
修复内容：
1. DASR 改为正确的方向一致率计算
2. 物理约束模型 (CA+SmoothOnly, CA+PeakOnly, PCCA-LSTM) 重新训练
3. 实测降雨评估（修正列名识别）
4. 统计检验（Wilcoxon）
5. 生成最终汇总表

运行方式：
    python fix_and_rerun.py
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

# GPU模式
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
from datetime import datetime

from dataset import SWMMDataset
from registry import create_model
from physics_loss import PhysicallyConsistentLoss
import lstm, gru, attention
from sklearn.preprocessing import MinMaxScaler

# ================================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_ROOT = os.path.join('output', 'robust_final')
MODEL_DIR = os.path.join(OUTPUT_ROOT, 'models')
CACHE_DIR = os.path.join(OUTPUT_ROOT, 'data_cache')

SEQ_LEN = 288; DT = 5; BS = 32; LR = 0.001
EPOCHS = 200; PATIENCE = 20
N_SEEDS = 10
SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]
EXTREME_RPS = [20, 30, 50, 100]

# 需要重训的模型
RETRAIN_MODELS = {
    'CA+SmoothOnly': {'model_type': 'CausalAttentionLSTM', 'loss_type': 'physically_consistent',
                      'lambda_smooth': 0.01, 'lambda_peak': 0.0},
    'CA+PeakOnly':   {'model_type': 'CausalAttentionLSTM', 'loss_type': 'physically_consistent',
                      'lambda_smooth': 0.0, 'lambda_peak': 0.05},
    'PCCA-LSTM':     {'model_type': 'PCCA-LSTM', 'loss_type': 'physically_consistent',
                      'lambda_smooth': 0.01, 'lambda_peak': 0.05},
}

# 所有模型（用于重新评估）
ALL_MODELS = {
    'LSTM':           {'model_type': 'SimpleLSTM',              'loss_type': 'mse'},
    'GRU':            {'model_type': 'SimpleGRU',               'loss_type': 'mse'},
    'AttentionLSTM':  {'model_type': 'AttentionLSTM',           'loss_type': 'mse'},
    'RandomMaskLSTM': {'model_type': 'RandomMaskAttentionLSTM', 'loss_type': 'mse'},
    'CA-LSTM':        {'model_type': 'CausalAttentionLSTM',     'loss_type': 'mse'},
    'CA+SmoothOnly':  {'model_type': 'CausalAttentionLSTM',     'loss_type': 'physically_consistent'},
    'CA+PeakOnly':    {'model_type': 'CausalAttentionLSTM',     'loss_type': 'physically_consistent'},
    'PCCA-LSTM':      {'model_type': 'PCCA-LSTM',               'loss_type': 'physically_consistent'},
}

# ================================================================
# 正确的 DASR 计算
# ================================================================
def compute_metrics(pred, target):
    """正确的四维评估指标"""
    pred = np.array(pred).flatten()
    target = np.array(target).flatten()

    rmse = np.sqrt(np.mean((pred - target) ** 2))
    mae = np.mean(np.abs(pred - target))
    mape = np.mean(np.abs((pred - target) / (target + 1e-10))) * 100
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-10)

    # 峰值误差 (m)
    peak_error = pred[np.argmax(target)] - np.max(target)

    # DASR: 方向一致率 — 预测与目标逐步变化方向一致的百分比
    pred_diff = np.diff(pred)
    target_diff = np.diff(target)
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

# ================================================================
# 训练函数
# ================================================================
def train_model(model_name, config, train_dataset, seed):
    """训练单个模型（支持断点续跑：已存在则跳过）"""
    save_path = os.path.join(MODEL_DIR, f'{model_name}_seed{seed}.pth')

    # 断点续跑：如果模型已存在，直接跳过
    if os.path.exists(save_path):
        print(f"  [SKIP] {model_name} seed={seed} 已存在，跳过训练")
        checkpoint = torch.load(save_path, map_location=DEVICE, weights_only=False)
        model = create_model(config['model_type'], input_size=1, hidden_size=128,
                             num_layers=2, output_size=1, dropout=0.0)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(DEVICE)
        return model, checkpoint.get('train_info', {})

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = create_model(config['model_type'], input_size=1, hidden_size=128,
                         num_layers=2, output_size=1, dropout=0.3)
    model = model.to(DEVICE)

    n = len(train_dataset)
    indices = list(range(n))
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    split = int(n * 0.8)
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_subset = torch.utils.data.Subset(train_dataset, train_idx)
    val_subset = torch.utils.data.Subset(train_dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=BS, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BS, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    if config['loss_type'] == 'physically_consistent':
        criterion = PhysicallyConsistentLoss(
            lambda_smooth=config.get('lambda_smooth', 0.01),
            lambda_peak=config.get('lambda_peak', 0.05),
            lambda_mass=0.0,
        )
    else:
        criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            if config['loss_type'] == 'physically_consistent':
                loss, _ = criterion(output, target, rainfall=data,
                                    rain_scaler=train_dataset.rain_scaler,
                                    water_scaler=train_dataset.water_scaler,
                                    return_components=True)
            else:
                loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                if config['loss_type'] == 'physically_consistent':
                    vloss, _ = criterion(output, target, rainfall=data,
                                         rain_scaler=train_dataset.rain_scaler,
                                         water_scaler=train_dataset.water_scaler,
                                         return_components=True)
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

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save({
        'model_state_dict': model.state_dict(),
        'train_info': {'epochs': epoch + 1, 'best_val_loss': best_val_loss}
    }, save_path)

    return model, {'epochs': epoch + 1, 'best_val_loss': best_val_loss}


# ================================================================
# 评估函数
# ================================================================
def evaluate_per_event(model, dataset, water_scaler):
    model.eval()
    metrics_list = []
    with torch.no_grad():
        for i in range(len(dataset)):
            data, target = dataset[i]
            data = data.unsqueeze(0).to(DEVICE)
            target_t = target.unsqueeze(0).to(DEVICE)
            output = model(data)
            pred_np = output.cpu().numpy().reshape(-1, 1)
            target_np = target_t.cpu().numpy().reshape(-1, 1)
            pred_orig = water_scaler.inverse_transform(pred_np).flatten()
            target_orig = water_scaler.inverse_transform(target_np).flatten()
            metrics_list.append(compute_metrics(pred_orig, target_orig))
    return metrics_list


def evaluate_real_rainfall(model, rain_scaler, water_scaler):
    """实测降雨评估（正确列名识别）"""
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
            rain_col = water_col = None
            for c in df.columns:
                cs = str(c)
                if '雨强' in cs or 'rain' in cs.lower():
                    rain_col = c
                if 'STORM_L_01' in cs:
                    water_col = c
            if water_col is None:
                for c in df.columns:
                    if '液位' in str(c) and 'WASTE' not in str(c):
                        water_col = c
                        break
            if rain_col is None or water_col is None:
                continue

            rain = df[rain_col].values[:SEQ_LEN].astype(np.float32)
            true_water = df[water_col].values[:SEQ_LEN].astype(np.float32)
            if len(rain) < SEQ_LEN:
                rain = np.pad(rain, (0, SEQ_LEN - len(rain)))
                true_water = np.pad(true_water, (0, SEQ_LEN - len(true_water)))
            if np.max(rain) < 0.1:
                continue

            rain_scaled = rain_scaler.transform(rain.reshape(-1, 1)).reshape(-1)
            rain_tensor = torch.FloatTensor(rain_scaled).unsqueeze(0).unsqueeze(-1).to(DEVICE)
            with torch.no_grad():
                pred_scaled = model(rain_tensor)
            pred_orig = water_scaler.inverse_transform(
                pred_scaled.cpu().numpy().reshape(-1, 1)).flatten()
            results.append(compute_metrics(pred_orig, true_water))
        except Exception:
            continue
    return results

# ================================================================
# MAIN
# ================================================================
if __name__ == '__main__':
    print(f"{'='*60}")
    print(f"  补充实验: 修复DASR + 重训物理约束模型 + 实测评估")
    print(f"{'='*60}")
    print(f"  设备: {DEVICE}")
    print(f"{'='*60}\n")

    # Load cached data
    print("[1] 加载缓存数据...")
    train_data = np.load(os.path.join(CACHE_DIR, 'train_data.npz'), allow_pickle=True)
    rain_scaler = MinMaxScaler().fit(train_data['rainfall_events'].reshape(-1, 1))
    water_scaler = MinMaxScaler().fit(train_data['water_level_events'].reshape(-1, 1))

    train_dataset = SWMMDataset.__new__(SWMMDataset)
    train_dataset.X = train_data['X']
    train_dataset.y = train_data['y']
    train_dataset.rainfall_events = train_data['rainfall_events']
    train_dataset.water_level_events = train_data['water_level_events']
    train_dataset.rain_scaler = rain_scaler
    train_dataset.water_scaler = water_scaler
    print(f"  训练集: {len(train_dataset.X)} 样本")

    extreme_datasets = {}
    for rp in EXTREME_RPS:
        ext_data = np.load(os.path.join(CACHE_DIR, f'extreme_T{rp}.npz'), allow_pickle=True)
        ds = SWMMDataset.__new__(SWMMDataset)
        ds.X = ext_data['X']
        ds.y = ext_data['y']
        ds.rain_scaler = rain_scaler
        ds.water_scaler = water_scaler
        extreme_datasets[rp] = ds
        print(f"  T={rp}年: {len(ds.X)} 样本")

    # Step 2: 重训3个失败的物理约束模型
    print(f"\n[2] 重训物理约束模型 (3模型 × {N_SEEDS}种子 = {3*N_SEEDS}次)...")
    for model_name, config in RETRAIN_MODELS.items():
        for seed in SEEDS:
            t0 = time.time()
            print(f"  {model_name} seed={seed}...", end=' ')
            try:
                model, info = train_model(model_name, config, train_dataset, seed)
                print(f"OK ({info['epochs']}轮, val={info['best_val_loss']:.6f}, {time.time()-t0:.0f}s)")
            except Exception as e:
                print(f"FAILED: {e}")

    # Step 3: 重新评估所有模型（使用正确的DASR）
    print(f"\n[3] 重新评估所有模型（正确DASR）...")
    results_extreme = {}  # {model: {seed: {rp: [metrics]}}}
    results_real = {}     # {model: {seed: [metrics]}}

    for model_name, config in ALL_MODELS.items():
        results_extreme[model_name] = {}
        results_real[model_name] = {}

        for seed in SEEDS:
            model_path = os.path.join(MODEL_DIR, f'{model_name}_seed{seed}.pth')
            if not os.path.exists(model_path):
                continue

            try:
                checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
                model = create_model(config['model_type'], input_size=1, hidden_size=128,
                                     num_layers=2, output_size=1, dropout=0.0)
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(DEVICE)
                model.eval()

                # 极端事件评估
                results_extreme[model_name][seed] = {}
                for rp in EXTREME_RPS:
                    results_extreme[model_name][seed][rp] = evaluate_per_event(
                        model, extreme_datasets[rp], water_scaler)

                # 实测降雨评估
                results_real[model_name][seed] = evaluate_real_rainfall(
                    model, rain_scaler, water_scaler)

            except Exception as e:
                print(f"  [WARN] {model_name} seed={seed}: {e}")
                continue

        # Progress
        n_seeds_done = len(results_extreme[model_name])
        if n_seeds_done > 0:
            t100_rmse = []
            t100_dasr = []
            for s in results_extreme[model_name]:
                for m in results_extreme[model_name][s].get(100, []):
                    t100_rmse.append(m['RMSE'])
                    t100_dasr.append(m['DASR'])
            n_real = sum(len(results_real[model_name].get(s, [])) for s in SEEDS)
            print(f"  {model_name:18s}: {n_seeds_done} seeds, "
                  f"T100 RMSE={np.mean(t100_rmse):.4f}±{np.std(t100_rmse):.4f}, "
                  f"DASR={np.mean(t100_dasr):.1f}%±{np.std(t100_dasr):.1f}%, "
                  f"实测={n_real}事件")

    # Step 4: 生成汇总表
    print(f"\n[4] 生成最终汇总表...")

    # Table 3: T=100 极端外推
    t3_rows = []
    for model_name in ALL_MODELS.keys():
        all_m = []
        for seed in SEEDS:
            all_m.extend(results_extreme.get(model_name, {}).get(seed, {}).get(100, []))
        if not all_m:
            continue
        t3_rows.append({
            'Model': model_name,
            'RMSE_mean': np.mean([m['RMSE'] for m in all_m]),
            'RMSE_std': np.std([m['RMSE'] for m in all_m]),
            'MAE_mean': np.mean([m['MAE'] for m in all_m]),
            'MAPE_mean': np.mean([m['MAPE'] for m in all_m]),
            'R2_mean': np.mean([m['R2'] for m in all_m]),
            'PeakError_mean': np.mean([m['PeakError'] for m in all_m]),
            'DASR_mean': np.mean([m['DASR'] for m in all_m]),
            'DASR_std': np.std([m['DASR'] for m in all_m]),
            'N': len(all_m),
        })
    df_t3 = pd.DataFrame(t3_rows)
    df_t3.to_csv(os.path.join(OUTPUT_ROOT, 'table3_final.csv'), index=False)
    print("\n  === Table 3: T=100 极端外推 ===")
    print(df_t3[['Model', 'RMSE_mean', 'RMSE_std', 'DASR_mean', 'DASR_std', 'PeakError_mean', 'R2_mean']].to_string())

    # Table 4: 实测降雨
    t4_rows = []
    for model_name in ALL_MODELS.keys():
        all_m = []
        for seed in SEEDS:
            all_m.extend(results_real.get(model_name, {}).get(seed, []))
        if not all_m:
            continue
        t4_rows.append({
            'Model': model_name,
            'RMSE_mean': np.mean([m['RMSE'] for m in all_m]),
            'RMSE_std': np.std([m['RMSE'] for m in all_m]),
            'MAE_mean': np.mean([m['MAE'] for m in all_m]),
            'DASR_mean': np.mean([m['DASR'] for m in all_m]),
            'DASR_std': np.std([m['DASR'] for m in all_m]),
            'PeakError_mean': np.mean([m['PeakError'] for m in all_m]),
            'PeakError_std': np.std([m['PeakError'] for m in all_m]),
            'N': len(all_m),
        })
    df_t4 = pd.DataFrame(t4_rows)
    df_t4.to_csv(os.path.join(OUTPUT_ROOT, 'table4_final.csv'), index=False)
    print("\n  === Table 4: 实测降雨 ===")
    if not df_t4.empty:
        print(df_t4[['Model', 'RMSE_mean', 'RMSE_std', 'DASR_mean', 'DASR_std']].to_string())

    # Multi-RP table
    mrp_rows = []
    for model_name in ALL_MODELS.keys():
        for rp in EXTREME_RPS:
            all_m = []
            for seed in SEEDS:
                all_m.extend(results_extreme.get(model_name, {}).get(seed, {}).get(rp, []))
            if not all_m:
                continue
            mrp_rows.append({
                'Model': model_name, 'RP': rp,
                'RMSE_mean': np.mean([m['RMSE'] for m in all_m]),
                'RMSE_std': np.std([m['RMSE'] for m in all_m]),
                'DASR_mean': np.mean([m['DASR'] for m in all_m]),
                'DASR_std': np.std([m['DASR'] for m in all_m]),
            })
    pd.DataFrame(mrp_rows).to_csv(os.path.join(OUTPUT_ROOT, 'table_multi_rp_final.csv'), index=False)

    # Step 5: 统计检验
    print(f"\n[5] 统计检验 (Wilcoxon signed-rank)...")
    stat_tests = []
    comparisons = [
        ('H1: Attn vs LSTM', 'AttentionLSTM', 'LSTM'),
        ('H1: RandomMask vs LSTM', 'RandomMaskLSTM', 'LSTM'),
        ('H1: CA vs Attn', 'CA-LSTM', 'AttentionLSTM'),
        ('H2: CA vs LSTM', 'CA-LSTM', 'LSTM'),
        ('H2: PCCA vs CA', 'PCCA-LSTM', 'CA-LSTM'),
        ('Ablation: Smooth vs CA', 'CA+SmoothOnly', 'CA-LSTM'),
        ('Ablation: Peak vs CA', 'CA+PeakOnly', 'CA-LSTM'),
    ]

    for desc, model_a, model_b in comparisons:
        for metric in ['RMSE', 'DASR']:
            a_vals, b_vals = [], []
            for seed in SEEDS:
                a_m = results_extreme.get(model_a, {}).get(seed, {}).get(100, [])
                b_m = results_extreme.get(model_b, {}).get(seed, {}).get(100, [])
                if a_m and b_m:
                    a_vals.append(np.mean([m[metric] for m in a_m]))
                    b_vals.append(np.mean([m[metric] for m in b_m]))

            if len(a_vals) >= 5:
                try:
                    stat, p = scipy_stats.wilcoxon(a_vals, b_vals)
                    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                    stat_tests.append({
                        'Comparison': f'{desc} [{metric}]',
                        'W': stat, 'p': p, 'N': len(a_vals), 'sig': sig,
                        'A_mean': np.mean(a_vals), 'B_mean': np.mean(b_vals),
                    })
                    print(f"  {desc} [{metric}]: W={stat:.1f}, p={p:.4f} {sig}")
                except Exception as e:
                    print(f"  {desc} [{metric}]: failed ({e})")

    df_stats = pd.DataFrame(stat_tests)
    df_stats.to_csv(os.path.join(OUTPUT_ROOT, 'statistical_tests_final.csv'), index=False)

    # Step 6: 保存完整JSON
    def to_ser(obj):
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: to_ser(v) for k, v in obj.items()}
        if isinstance(obj, list): return [to_ser(v) for v in obj]
        return obj

    final = {
        'config': {'device': DEVICE, 'seeds': SEEDS, 'n_seeds': N_SEEDS, 'epochs': EPOCHS},
        'table3_T100': to_ser(t3_rows),
        'table4_real': to_ser(t4_rows),
        'multi_rp': to_ser(mrp_rows),
        'statistical_tests': to_ser(stat_tests),
    }
    with open(os.path.join(OUTPUT_ROOT, 'results_final.json'), 'w', encoding='utf-8') as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    # 打印核心结论
    print(f"\n{'='*60}")
    print("  核心结论")
    print(f"{'='*60}")
    if df_t3 is not None and not df_t3.empty:
        for _, row in df_t3.iterrows():
            print(f"  {row['Model']:18s}: RMSE={row['RMSE_mean']:.4f}±{row['RMSE_std']:.4f}, "
                  f"DASR={row['DASR_mean']:.1f}%±{row['DASR_std']:.1f}%")

    print(f"\n  实验完成！结果: {OUTPUT_ROOT}")
    print(f"{'='*60}")