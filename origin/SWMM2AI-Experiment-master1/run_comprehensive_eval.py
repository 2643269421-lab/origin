#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
综合评估脚本：5模型 × 4节点 × T=100yr + 峰值误差分布
"""
import sys, os, json, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from model import Predictor
from swmm.simulator import SWMMSimulator
from swmm.rainfall.generator import RainfallGenerator

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

SEQ_LEN = 288; DT = 5; N_TEST = 15
EXTREME_RP = 100  # T=100yr

NODES = ['SN_001', 'SN_017', 'SN_049', 'SN_011']
NODE_LABELS = {
    'SN_001': '下游排口 (1.46m)',
    'SN_017': '中游 (1.85m)',
    'SN_049': '中上游 (2.70m)',
    'SN_011': '上游 (3.08m)',
}

MODEL_PATHS = {
    'LSTM': 'output/extreme_experiment_20260610_110041/extreme_simplelstm_model.pth',
    'GRU': 'output/extreme_experiment_20260610_110041/extreme_simplegru_model.pth',
    'Attn-LSTM': 'output/extreme_experiment_20260610_110041/extreme_attentionlstm_model.pth',
    'CA-LSTM': 'output/extreme_experiment_20260610_110041/extreme_causalattentionlstm_model.pth',
    'PCCA-LSTM': 'output/ablation_20260610_150250/pgca_lstm_model.pth',
}
MODEL_COLORS = {
    'LSTM': '#f27970', 'GRU': '#b07d62', 'Attn-LSTM': '#8983bf',
    'CA-LSTM': '#54b345', 'PCCA-LSTM': '#4472C4'
}

OUTPUT_DIR = 'output/comprehensive_eval'
os.makedirs(OUTPUT_DIR, exist_ok=True)
FIGS_DIR = os.path.join(OUTPUT_DIR, 'figures')
os.makedirs(FIGS_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ============ Load models ============
predictors = {}
for name, path in MODEL_PATHS.items():
    if os.path.exists(path):
        predictors[name] = Predictor(model_path=path, output_dir=OUTPUT_DIR, device=device)
        print(f"Loaded {name}")
    else:
        print(f"WARN: {path} not found")

# ============ Generate test data for each node ============
test_data = {}
for node_id in NODES:
    print(f"\n=== Generating T=100yr test data for {node_id} ===")
    rg = RainfallGenerator(time_step_min=DT)
    sim = SWMMSimulator(template_inp_path='template.inp', output_dir=OUTPUT_DIR,
                       output_element=node_id, output_type='node', output_variable='depth')
    rains, waters = [], []
    for i in range(N_TEST):
        rain = rg.generate_rainfall_event(seq_length=SEQ_LEN, rain_type='chicago',
                                          duration_hours=np.random.uniform(1,6),
                                          return_period=EXTREME_RP,
                                          peak_position=np.random.uniform(0.3,0.7),
                                          start_idx=np.random.randint(0,36))
        try:
            res = sim.run_swmm_simulation(rainfall_mm_h=rain)
            if res and len(res['values']) == SEQ_LEN:
                rains.append(rain); waters.append(res['values'])
        except Exception as e:
            pass
    test_data[node_id] = {'rainfall': np.array(rains), 'water_swmm': np.array(waters)}
    print(f"  {len(rains)}/{N_TEST} valid events, max water depth: {waters[0].max() if waters else 0:.3f}m")

# ============ Evaluate all models × all nodes ============
all_results = {}
for node_id in NODES:
    all_results[node_id] = {}
    rain = test_data[node_id]['rainfall']
    swmm_w = test_data[node_id]['water_swmm']

    for name, pred in predictors.items():
        preds = pred.predict_batch(rain)
        metrics = []
        peak_errors = []
        for i in range(len(rain)):
            p = preds[i]; t = swmm_w[i]
            mse = np.mean((p-t)**2)
            mae = np.mean(np.abs(p-t))
            active = t > 0.001
            mape = np.mean(np.abs((p[active]-t[active])/(t[active]+1e-10)))*100 if active.sum()>0 else 0
            ss_r = np.sum((t-p)**2); ss_t = np.sum((t-np.mean(t))**2)
            r2 = 1 - ss_r/(ss_t+1e-10)
            peak_err = p[np.argmax(t)] - t.max()
            metrics.append({'RMSE': np.sqrt(mse), 'MAE': mae, 'MAPE': mape, 'R2': r2, 'PeakErr': peak_err})
            peak_errors.append(peak_err)

        all_results[node_id][name] = {
            'RMSE': (np.mean([m['RMSE'] for m in metrics]), np.std([m['RMSE'] for m in metrics])),
            'MAE': (np.mean([m['MAE'] for m in metrics]), np.std([m['MAE'] for m in metrics])),
            'MAPE': (np.mean([m['MAPE'] for m in metrics]), np.std([m['MAPE'] for m in metrics])),
            'R2': (np.mean([m['R2'] for m in metrics]), np.std([m['R2'] for m in metrics])),
            'PeakErr': (np.mean([m['PeakErr'] for m in metrics]), np.std([m['PeakErr'] for m in metrics])),
            'peak_error_list': peak_errors,
            'swmm_max_depth': float(swmm_w.max(axis=1).mean()),
        }
        v = all_results[node_id][name]
        print(f"  {node_id} {name:12s}: RMSE={v['RMSE'][0]:.5f}±{v['RMSE'][1]:.5f}  R²={v['R2'][0]:.4f}  PeakErr={v['PeakErr'][0]:.4f}m  MaxDepth={v['swmm_max_depth']:.3f}m")

# ============ Save results ============
summary = {}
for node_id in NODES:
    summary[node_id] = {}
    for name in predictors:
        v = all_results[node_id][name]
        summary[node_id][name] = {
            'RMSE_mean': v['RMSE'][0], 'RMSE_std': v['RMSE'][1],
            'MAE_mean': v['MAE'][0], 'MAE_std': v['MAE'][1],
            'R2_mean': v['R2'][0], 'R2_std': v['R2'][1],
            'PeakErr_mean': v['PeakErr'][0], 'PeakErr_std': v['PeakErr'][1],
            'swmm_max_depth': v['swmm_max_depth'],
        }

with open(os.path.join(OUTPUT_DIR, 'multinode_results.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# ============ Generate figures ============
print("\n=== Generating figures ===")

# Fig 1: Multi-node bar chart (RMSE per model × node)
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(NODES))
w = 0.15
models_list = list(predictors.keys())
for j, name in enumerate(models_list):
    vals = [all_results[node_id][name]['RMSE'][0] for node_id in NODES]
    stds = [all_results[node_id][name]['RMSE'][1] for node_id in NODES]
    bars = ax.bar(x + j*w, vals, w, yerr=stds, capsize=3, color=MODEL_COLORS[name], label=name, alpha=0.9)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0003, f'{val:.4f}',
                ha='center', va='bottom', fontsize=7, rotation=90)

ax.set_xticks(x + w*2)
ax.set_xticklabels([f'{NODE_LABELS[n]}' for n in NODES], fontsize=9)
ax.set_ylabel('RMSE (m)', fontsize=12)
ax.set_title('T=100yr: 5 Models × 4 Nodes — RMSE Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, 'multinode_rmse_bars.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  -> multinode_rmse_bars.png")

# Fig 2: Peak error box plot (all models)
fig, ax = plt.subplots(figsize=(14, 6))
box_data = []
box_labels = []
box_colors = []
for name in models_list:
    all_peaks = []
    for node_id in NODES:
        all_peaks.extend(all_results[node_id][name]['peak_error_list'])
    box_data.append(all_peaks)
    box_labels.append(name)
    box_colors.append(MODEL_COLORS[name])

bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.5)
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color); patch.set_alpha(0.7)

ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax.set_ylabel('Peak Water Level Error (m)', fontsize=12)
ax.set_title('T=100yr: Peak Error Distribution — All Models × All Nodes (60 samples each)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
# Add mean annotations
for i, d in enumerate(box_data):
    mean_val = np.mean(d)
    ax.annotate(f'μ={mean_val:.3f}', xy=(i+1, mean_val), xytext=(i+1.35, mean_val+0.003),
                fontsize=8, color='darkred', fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, 'peak_error_boxplot.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  -> peak_error_boxplot.png")

# Fig 3: Per-node peak error box plots (separate subplots)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()
for idx, node_id in enumerate(NODES):
    ax = axes[idx]
    box_data = [all_results[node_id][name]['peak_error_list'] for name in models_list]
    bp = ax.boxplot(box_data, labels=models_list, patch_artist=True, widths=0.5)
    for patch, name in zip(bp['boxes'], models_list):
        patch.set_facecolor(MODEL_COLORS[name]); patch.set_alpha(0.7)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_title(f'{node_id}: {NODE_LABELS[node_id]}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Peak Error (m)')
    ax.grid(True, alpha=0.3, axis='y')
fig.suptitle('T=100yr: Peak Error Distribution by Node (15 samples each)', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, 'peak_error_per_node.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  -> peak_error_per_node.png")

# ============ Print paper-ready table ============
print("\n" + "="*80)
print("TABLE: Multi-Node RMSE at T=100yr (for paper)")
print("="*80)
header = f"{'Model':<14}" + "".join(f"{NODE_LABELS[n]:>20}" for n in NODES)
print(header)
print("-" * (14 + 20*4))
for name in models_list:
    vals = "".join(f"{all_results[node_id][name]['RMSE'][0]:.4f}±{all_results[node_id][name]['RMSE'][1]:.4f}    "[:20] for node_id in NODES)
    print(f"{name:<14}{vals}")

print("\n" + "="*80)
print("TABLE: Multi-Node Peak Error at T=100yr (for paper)")
print("="*80)
print(f"{'Model':<14}" + "".join(f"{NODE_LABELS[n]:>20}" for n in NODES))
print("-" * (14 + 20*4))
for name in models_list:
    vals = "".join(f"{all_results[node_id][name]['PeakErr'][0]:.4f}±{all_results[node_id][name]['PeakErr'][1]:.4f}    "[:20] for node_id in NODES)
    print(f"{name:<14}{vals}")

print("\n[DONE] Results saved to:", OUTPUT_DIR)
