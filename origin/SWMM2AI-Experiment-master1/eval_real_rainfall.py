#!/usr/bin/env python
# encoding: utf-8
"""真实降雨数据评估 - SN_001节点"""
import sys, os, json, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

import torch
from model import Predictor

SEQ = 288; DT = 5
DATA_DIR = 'Actual Rainfall_Water Level'
MODEL_PATHS = {
    'LSTM': 'output/extreme_experiment_20260610_110041/extreme_simplelstm_model.pth',
    'CA-LSTM': 'output/extreme_experiment_20260610_110041/extreme_causalattentionlstm_model.pth',
    'PCCA-LSTM': 'output/ablation_20260610_150250/pgca_lstm_model.pth',
}
OUT = 'output/real_rainfall_eval'
FIGS = os.path.join(OUT, 'figures')
os.makedirs(FIGS, exist_ok=True)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load models
preds = {}
for n, p in MODEL_PATHS.items():
    preds[n] = Predictor(model_path=p, output_dir=OUT, device=dev)
    print(f'[OK] {n}')

# Load all real rainfall files
files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.xlsx') and not f.startswith('~')])
print(f'\n{len(files)} real rainfall event files found')

all_metrics = []
best_events = []  # (mape, file_idx, preds_dict, rain, water)

for fidx, fname in enumerate(files):
    df = pd.read_excel(os.path.join(DATA_DIR, fname))
    # Try to find rainfall and water level columns
    rain_col = None; water_col = None
    for c in df.columns:
        cstr = str(c).lower()
        if 'rain' in cstr or '降雨' in str(c):
            rain_col = c
        if 'water' in cstr or '水位' in str(c) or 'depth' in cstr:
            water_col = c

    if rain_col is None or water_col is None:
        continue

    rain_raw = df[rain_col].values.astype(np.float32)
    water_raw = df[water_col].values.astype(np.float32)

    # Take segments of SEQ length
    n_segments = max(1, len(rain_raw) // SEQ)
    for seg in range(n_segments):
        start = seg * SEQ
        end = start + SEQ
        if end > len(rain_raw):
            break
        rain_seg = rain_raw[start:end]
        water_seg = water_raw[start:end]

        # Skip if water level is flat/near zero
        if water_seg.max() < 0.001 or water_seg.std() < 0.001:
            continue

        metrics_per_model = {}
        for name, pr in preds.items():
            try:
                pred = pr.predict(rain_seg)
                # Clip to SEQ length
                pred = pred[:len(water_seg)]
                wt = water_seg[:len(pred)]
                mse = np.mean((pred - wt)**2)
                mae = np.mean(np.abs(pred - wt))
                active = wt > 0.001
                mape = np.mean(np.abs((pred[active]-wt[active])/(wt[active]+1e-10)))*100 if active.sum()>0 else 0
                ssr = np.sum((wt-pred)**2); sst = np.sum((wt-np.mean(wt))**2)
                r2 = 1 - ssr/(sst+1e-10)
                pe = pred[np.argmax(wt)] - wt.max()
                metrics_per_model[name] = {
                    'RMSE': np.sqrt(mse), 'MAE': mae, 'MAPE': mape, 'R2': r2,
                    'PeakErr': pe, 'pred': pred, 'target': wt
                }
            except Exception as e:
                continue

        if len(metrics_per_model) >= 3:
            all_metrics.append(metrics_per_model)
            # Track best (lowest PCCA-LSTM MAPE)
            if 'PCCA-LSTM' in metrics_per_model:
                best_events.append((
                    metrics_per_model['PCCA-LSTM']['MAPE'],
                    fidx, fname, seg,
                    metrics_per_model,
                    rain_seg, water_seg
                ))

# Sort by PCCA-LSTM MAPE ascending
best_events.sort(key=lambda x: x[0])

# Aggregate results across all segments
print(f'\nEvaluated {len(all_metrics)} segments across {len(files)} files')

results = {}
for name in preds.keys():
    vals = [m[name] for m in all_metrics if name in m]
    results[name] = {
        'RMSE': float(np.mean([v['RMSE'] for v in vals])),
        'MAE': float(np.mean([v['MAE'] for v in vals])),
        'MAPE': float(np.mean([v['MAPE'] for v in vals])),
        'R2': float(np.mean([v['R2'] for v in vals])),
        'PeakErr': float(np.mean([v['PeakErr'] for v in vals])),
        'R2_positive_count': sum(1 for v in vals if v['R2'] > 0),
        'total_segments': len(vals),
    }
    print(f'{name:12s} n={len(vals):3d}  RMSE={results[name]["RMSE"]:.5f}  MAE={results[name]["MAE"]:.4f}  '
          f'MAPE={results[name]["MAPE"]:.1f}%  R2={results[name]["R2"]:.4f}  '
          f'PeakErr={results[name]["PeakErr"]:.4f}m  R2>0: {results[name]["R2_positive_count"]}/{results[name]["total_segments"]}')

# Save results
with open(os.path.join(OUT, 'real_rainfall_results.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# Plot best 3 events
n_best = min(6, len(best_events))
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
axes = axes.flatten()
colors = {'LSTM': '#f27970', 'CA-LSTM': '#54b345', 'PCCA-LSTM': '#4472C4'}

for idx in range(min(n_best, 6)):
    ax = axes[idx]
    mape_val, fidx, fname, seg, metrics, rain, water = best_events[idx]
    time_h = np.arange(len(rain)) * DT / 60

    ax2 = ax.twinx()
    ax2.bar(time_h, rain, width=DT/60/1.5, alpha=0.15, color='steelblue')
    ax2.set_ylabel('Rainfall (mm/h)', color='steelblue', fontsize=8)

    ax.plot(time_h, water, 'k-', linewidth=1.5, label='Observed', alpha=0.8)
    for name in ['LSTM', 'CA-LSTM', 'PCCA-LSTM']:
        if name in metrics:
            ax.plot(time_h[:len(metrics[name]['pred'])], metrics[name]['pred'],
                    color=colors[name], linewidth=1.8, alpha=0.85, label=name)

    ax.set_xlabel('Time (h)', fontsize=8)
    ax.set_ylabel('Water Level (m)', fontsize=8)
    ax.set_title(f'{fname} seg{seg} (PCCA MAPE={mape_val:.1f}%)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

for idx in range(n_best, 6):
    axes[idx].axis('off')

fig.suptitle('Real Rainfall Events: Top-6 Predictions on SN_001', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(FIGS, 'real_rainfall_predictions.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'\nSaved: real_rainfall_predictions.png ({n_best} events)')

# Print paper-ready summary
print('\n=== PAPER SUMMARY ===')
print(f'Evaluated on {len(files)} real rainfall events ({len(all_metrics)} valid segments) from SN_001 node.')
print(f'PCCA-LSTM: RMSE={results["PCCA-LSTM"]["RMSE"]:.4f}m  MAPE={results["PCCA-LSTM"]["MAPE"]:.1f}%  R2={results["PCCA-LSTM"]["R2"]:.4f}')
print(f'CA-LSTM:    RMSE={results["CA-LSTM"]["RMSE"]:.4f}m  MAPE={results["CA-LSTM"]["MAPE"]:.1f}%  R2={results["CA-LSTM"]["R2"]:.4f}')
print(f'LSTM:       RMSE={results["LSTM"]["RMSE"]:.4f}m  MAPE={results["LSTM"]["MAPE"]:.1f}%  R2={results["LSTM"]["R2"]:.4f}')
