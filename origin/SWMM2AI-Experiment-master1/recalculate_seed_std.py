#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Recalculate per-seed standard deviations for paper reporting.
This script loads each trained model, evaluates on T=100 extreme events,
and outputs per-seed RMSE/DASR means, then computes seed-level std.

Also evaluates real rainfall to get RMSE (previously missing in table4).

Run: python recalculate_seed_std.py
Output: output/robust_final/per_seed_metrics.csv
        output/robust_final/table3_corrected.csv
        output/robust_final/table4_corrected.csv
"""
import os, sys, io, json, copy
import numpy as np
import pandas as pd

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Force CPU to avoid page file issues
os.environ['CUDA_VISIBLE_DEVICES'] = ''

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SWMMDataset
from registry import create_model
import lstm, gru, attention
from sklearn.preprocessing import MinMaxScaler
import glob

DEVICE = 'cpu'
OUTPUT_ROOT = os.path.join('output', 'robust_final')
MODEL_DIR = os.path.join(OUTPUT_ROOT, 'models')
CACHE_DIR = os.path.join(OUTPUT_ROOT, 'data_cache')
SEQ_LEN = 288

SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]

ALL_MODELS = {
    'LSTM':           {'model_type': 'SimpleLSTM'},
    'GRU':            {'model_type': 'SimpleGRU'},
    'AttentionLSTM':  {'model_type': 'AttentionLSTM'},
    'RandomMaskLSTM': {'model_type': 'RandomMaskAttentionLSTM'},
    'CA-LSTM':        {'model_type': 'CausalAttentionLSTM'},
    'CA+SmoothOnly':  {'model_type': 'CausalAttentionLSTM'},
    'CA+PeakOnly':    {'model_type': 'CausalAttentionLSTM'},
    'PCCA-LSTM':      {'model_type': 'PCCA-LSTM'},
}


def compute_metrics(pred, target):
    pred = np.array(pred).flatten()
    target = np.array(target).flatten()
    rmse = np.sqrt(np.mean((pred - target) ** 2))
    mae = np.mean(np.abs(pred - target))
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-10)
    peak_error = pred[np.argmax(target)] - np.max(target)
    # DASR
    pred_diff = np.diff(pred)
    target_diff = np.diff(target)
    significant = np.abs(target_diff) > 1e-6
    if np.sum(significant) > 0:
        same_dir = np.sign(pred_diff[significant]) == np.sign(target_diff[significant])
        dasr = np.mean(same_dir) * 100
    else:
        dasr = 50.0
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'PeakError': peak_error, 'DASR': dasr}


def evaluate_dataset(model, dataset, water_scaler):
    model.eval()
    metrics_list = []
    with torch.no_grad():
        for i in range(len(dataset)):
            data, target = dataset[i]
            data = data.unsqueeze(0).to(DEVICE)
            output = model(data)
            pred_np = output.cpu().numpy().reshape(-1, 1)
            target_np = target.numpy().reshape(-1, 1)
            pred_orig = water_scaler.inverse_transform(pred_np).flatten()
            target_orig = water_scaler.inverse_transform(target_np).flatten()
            metrics_list.append(compute_metrics(pred_orig, target_orig))
    return metrics_list


def evaluate_real_rainfall(model, rain_scaler, water_scaler):
    data_dir = os.path.join(PROJECT_ROOT, 'Actual Rainfall_Water Level')
    xlsx_files = sorted(glob.glob(os.path.join(data_dir, '*.xlsx')))
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


if __name__ == '__main__':
    print("=" * 60)
    print("  Recalculating per-seed metrics for corrected reporting")
    print("=" * 60)

    # Load data
    print("\n[1] Loading cached data...")
    train_data = np.load(os.path.join(CACHE_DIR, 'train_data.npz'), allow_pickle=True)
    rain_scaler = MinMaxScaler().fit(train_data['rainfall_events'].reshape(-1, 1))
    water_scaler = MinMaxScaler().fit(train_data['water_level_events'].reshape(-1, 1))

    # Load T=100 test set
    ext_data = np.load(os.path.join(CACHE_DIR, 'extreme_T100.npz'), allow_pickle=True)
    test_dataset = SWMMDataset.__new__(SWMMDataset)
    test_dataset.X = ext_data['X']
    test_dataset.y = ext_data['y']
    test_dataset.rain_scaler = rain_scaler
    test_dataset.water_scaler = water_scaler
    print(f"  T=100 test events: {len(test_dataset.X)}")

    # Evaluate all models per-seed
    print("\n[2] Evaluating per-seed...")
    per_seed_rows = []

    for model_name, config in ALL_MODELS.items():
        print(f"\n  {model_name}:")
        for seed in SEEDS:
            model_path = os.path.join(MODEL_DIR, f'{model_name}_seed{seed}.pth')
            if not os.path.exists(model_path):
                print(f"    seed {seed}: MISSING")
                continue

            checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
            model = create_model(config['model_type'], input_size=1, hidden_size=128,
                                 num_layers=2, output_size=1, dropout=0.0)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(DEVICE)
            model.eval()

            # T=100 evaluation
            t100_metrics = evaluate_dataset(model, test_dataset, water_scaler)
            seed_rmse = np.mean([m['RMSE'] for m in t100_metrics])
            seed_dasr = np.mean([m['DASR'] for m in t100_metrics])
            seed_r2 = np.mean([m['R2'] for m in t100_metrics])
            seed_peak = np.mean([m['PeakError'] for m in t100_metrics])

            # Real rainfall evaluation
            real_metrics = evaluate_real_rainfall(model, rain_scaler, water_scaler)
            real_rmse = np.mean([m['RMSE'] for m in real_metrics]) if real_metrics else None
            real_dasr = np.mean([m['DASR'] for m in real_metrics]) if real_metrics else None
            real_n = len(real_metrics)

            per_seed_rows.append({
                'Model': model_name, 'Seed': seed,
                'T100_RMSE': seed_rmse, 'T100_DASR': seed_dasr,
                'T100_R2': seed_r2, 'T100_PeakError': seed_peak,
                'T100_N_events': len(t100_metrics),
                'Real_RMSE': real_rmse, 'Real_DASR': real_dasr,
                'Real_N_events': real_n,
            })
            real_rmse_str = f"{real_rmse:.5f}" if real_rmse is not None else "N/A"
            real_dasr_str = f"{real_dasr:.1f}" if real_dasr is not None else "N/A"
            print(f"    seed {seed}: T100 RMSE={seed_rmse:.5f} DASR={seed_dasr:.1f}% | Real RMSE={real_rmse_str} DASR={real_dasr_str}% (n={real_n})")

    # Save per-seed data
    df_seeds = pd.DataFrame(per_seed_rows)
    df_seeds.to_csv(os.path.join(OUTPUT_ROOT, 'per_seed_metrics.csv'), index=False)
    print(f"\n  Saved: per_seed_metrics.csv ({len(df_seeds)} rows)")

    # Generate corrected tables with seed-level std
    print("\n[3] Generating corrected tables...")

    # Table 3 corrected
    t3_rows = []
    for model_name in ALL_MODELS.keys():
        mdf = df_seeds[df_seeds['Model'] == model_name]
        if mdf.empty:
            continue
        t3_rows.append({
            'Model': model_name,
            'RMSE_mean': mdf['T100_RMSE'].mean(),
            'RMSE_seed_std': mdf['T100_RMSE'].std(),
            'DASR_mean': mdf['T100_DASR'].mean(),
            'DASR_seed_std': mdf['T100_DASR'].std(),
            'R2_mean': mdf['T100_R2'].mean(),
            'PeakError_mean': mdf['T100_PeakError'].mean(),
            'N_seeds': len(mdf),
        })
    df_t3 = pd.DataFrame(t3_rows)
    df_t3.to_csv(os.path.join(OUTPUT_ROOT, 'table3_corrected.csv'), index=False)
    print("\n  === Table 3 (corrected, seed-level std) ===")
    print(df_t3.to_string(index=False))

    # Table 4 corrected (with RMSE!)
    t4_rows = []
    for model_name in ALL_MODELS.keys():
        mdf = df_seeds[df_seeds['Model'] == model_name].dropna(subset=['Real_RMSE'])
        if mdf.empty:
            continue
        t4_rows.append({
            'Model': model_name,
            'RMSE_mean': mdf['Real_RMSE'].mean(),
            'RMSE_seed_std': mdf['Real_RMSE'].std(),
            'DASR_mean': mdf['Real_DASR'].mean(),
            'DASR_seed_std': mdf['Real_DASR'].std(),
            'N_seeds': len(mdf),
            'N_events_per_seed': int(mdf['Real_N_events'].mean()),
        })
    df_t4 = pd.DataFrame(t4_rows)
    df_t4.to_csv(os.path.join(OUTPUT_ROOT, 'table4_corrected.csv'), index=False)
    print("\n  === Table 4 (corrected, with RMSE) ===")
    print(df_t4.to_string(index=False))

    print("\n" + "=" * 60)
    print("  DONE. Check output/robust_final/per_seed_metrics.csv")
    print("  and table3_corrected.csv, table4_corrected.csv")
    print("=" * 60)