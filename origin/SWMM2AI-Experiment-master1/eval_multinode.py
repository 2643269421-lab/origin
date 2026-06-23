#!/usr/bin/env python
# encoding: utf-8
"""Multi-node evaluation: 5 models x 4 nodes at T=100yr"""
import sys, os, json, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

import torch
from model import Predictor
from swmm.simulator import SWMMSimulator
from swmm.rainfall.generator import RainfallGenerator

SEQ = 288; DT = 5; N_TEST = 15
NODES = ['SN_001', 'SN_017', 'SN_049', 'SN_011']
MODEL_PATHS = {
    'LSTM': 'output/extreme_experiment_20260610_110041/extreme_simplelstm_model.pth',
    'GRU': 'output/extreme_experiment_20260610_110041/extreme_simplegru_model.pth',
    'Attn-LSTM': 'output/extreme_experiment_20260610_110041/extreme_attentionlstm_model.pth',
    'CA-LSTM': 'output/extreme_experiment_20260610_110041/extreme_causalattentionlstm_model.pth',
    'PCCA-LSTM': 'output/ablation_20260610_150250/pgca_lstm_model.pth',
}
COLORS = {'LSTM': '#f27970', 'GRU': '#b07d62', 'Attn-LSTM': '#8983bf',
          'CA-LSTM': '#54b345', 'PCCA-LSTM': '#4472C4'}

OUT = 'output/comprehensive_eval'
FIGS = os.path.join(OUT, 'figures')
os.makedirs(FIGS, exist_ok=True)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---- Load models ----
preds = {}
for n, p in MODEL_PATHS.items():
    preds[n] = Predictor(model_path=p, output_dir=OUT, device=dev)
    print(f'[OK] {n}')

# ---- Generate test data per node ----
test = {}
for nid in NODES:
    print(f'[SIM] Node {nid}...')
    rg = RainfallGenerator(time_step_min=DT)
    sim = SWMMSimulator(template_inp_path='template.inp', output_dir=OUT,
                        output_element=nid, output_type='node', output_variable='depth')
    rains, waters = [], []
    for i in range(N_TEST):
        rain = rg.generate_rainfall_event(
            seq_length=SEQ, rain_type='chicago',
            duration_hours=np.random.uniform(1, 6),
            return_period=100,
            peak_position=np.random.uniform(0.3, 0.7),
            start_idx=np.random.randint(0, 36))
        try:
            res = sim.run_swmm_simulation(rainfall_mm_h=rain)
            if res and res['values'] is not None and len(res['values']) == SEQ:
                rains.append(rain)
                waters.append(res['values'])
        except Exception:
            pass
    test[nid] = {'rain': np.array(rains), 'swmm': np.array(waters)}
    mm = waters[0].max() if waters else 0
    print(f'  {len(rains)}/{N_TEST} valid, max_depth={mm:.3f}m')

# ---- Evaluate ----
results = {}
for nid in NODES:
    results[nid] = {}
    rain = test[nid]['rain']
    swmm = test[nid]['swmm']
    is_dry = swmm.max(axis=1).mean() < 0.001

    for name, pr in preds.items():
        pred = pr.predict_batch(rain)
        ms = []
        pes = []
        for i in range(len(rain)):
            p = pred[i]
            t = swmm[i]
            if t.max() < 1e-6:
                continue
            mse = np.mean((p - t) ** 2)
            mae = np.mean(np.abs(p - t))
            active = t > 0.001
            mape = np.mean(np.abs((p[active]-t[active])/(t[active]+1e-10)))*100 if active.sum()>0 else 0
            ssr = np.sum((t-p)**2)
            sst = np.sum((t-np.mean(t))**2)
            r2 = 1 - ssr/(sst+1e-10)
            pe = p[np.argmax(t)] - t.max()
            ms.append({'RMSE': np.sqrt(mse), 'MAE': mae, 'MAPE': mape, 'R2': r2, 'PeakErr': pe})
            pes.append(pe)

        if ms:
            results[nid][name] = {
                'RMSE': (np.mean([m['RMSE'] for m in ms]), np.std([m['RMSE'] for m in ms])),
                'MAE': (np.mean([m['MAE'] for m in ms]), np.std([m['MAE'] for m in ms])),
                'MAPE': (np.mean([m['MAPE'] for m in ms]), np.std([m['MAPE'] for m in ms])),
                'R2': (np.mean([m['R2'] for m in ms]), np.std([m['R2'] for m in ms])),
                'PeakErr': (np.mean(pes), np.std(pes)),
                'peaks': pes,
                'swmm_max': float(swmm.max(axis=1).mean()),
                'is_dry': is_dry,
            }
        else:
            results[nid][name] = {
                'RMSE': (0, 0), 'MAE': (0, 0), 'MAPE': (0, 0), 'R2': (0, 0),
                'PeakErr': (0, 0), 'peaks': [], 'swmm_max': 0, 'is_dry': True,
            }

        v = results[nid][name]
        print(f'  {nid} {name:12s} RMSE={v["RMSE"][0]:.5f} R2={v["R2"][0]:.4f} '
              f'PeakErr={v["PeakErr"][0]:.4f}m MaxD={v["swmm_max"]:.3f}m {"[DRY]" if v["is_dry"] else ""}')

# ---- Save JSON ----
summary = {}
for nid in NODES:
    summary[nid] = {}
    for name in preds:
        v = results[nid][name]
        summary[nid][name] = {
            'RMSE_mean': v['RMSE'][0], 'RMSE_std': v['RMSE'][1],
            'MAE_mean': v['MAE'][0], 'MAE_std': v['MAE'][1],
            'R2_mean': v['R2'][0], 'R2_std': v['R2'][1],
            'PeakErr_mean': v['PeakErr'][0], 'PeakErr_std': v['PeakErr'][1],
            'swmm_max_depth': v['swmm_max'], 'is_dry': bool(v['is_dry']),
        }

with open(os.path.join(OUT, 'multinode_results.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# ---- Print tables ----
models = list(preds.keys())

print('\n' + '=' * 110)
print('TABLE: Multi-Node RMSE at T=100yr (for paper)')
print('=' * 110)
hdr = f"{'Model':<14}" + ''.join(f'  {nid:<22}' for nid in NODES)
print(hdr)
print('-' * 110)
for name in models:
    row = f'{name:<14}'
    for nid in NODES:
        v = results[nid][name]
        row += f'  {v["RMSE"][0]:.5f}+-{v["RMSE"][1]:.5f}'
    print(row)

print('\n' + '=' * 110)
print('TABLE: Multi-Node Peak Error at T=100yr (for paper)')
print('=' * 110)
for name in models:
    row = f'{name:<14}'
    for nid in NODES:
        v = results[nid][name]
        row += f'  {v["PeakErr"][0]:.4f}+-{v["PeakErr"][1]:.4f}'
    print(row)

# ---- FIG 1: Multi-node RMSE bar chart ----
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(NODES))
w = 0.15
active_nodes = [n for n in NODES if not results[n][models[0]]['is_dry']]
for j, name in enumerate(models):
    vals = [results[nid][name]['RMSE'][0] if not results[nid][name]['is_dry'] else 0 for nid in NODES]
    stds = [results[nid][name]['RMSE'][1] if not results[nid][name]['is_dry'] else 0 for nid in NODES]
    bars = ax.bar(x + j*w, vals, w, yerr=stds, capsize=3, color=COLORS[name], label=name, alpha=0.9)
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=7, rotation=90)
ax.set_xticks(x + w*2)
labels = []
for nid in NODES:
    d = results[nid][models[0]]['swmm_max']
    labels.append(f'{nid}\n(depth={d:.2f}m)' if d > 0.001 else f'{nid}\n[DRY]')
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('RMSE (m)', fontsize=12)
ax.set_title('T=100yr: 5 Models x 4 Nodes -- RMSE Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig.savefig(os.path.join(FIGS, 'multinode_rmse_bars.png'), dpi=150, bbox_inches='tight')
plt.close()
print('\nSaved: multinode_rmse_bars.png')

# ---- FIG 2: Peak error box plot (all nodes combined) ----
fig, ax = plt.subplots(figsize=(14, 6))
bd = []
bl = []
for name in models:
    ap = []
    for nid in NODES:
        if not results[nid][name]['is_dry']:
            ap.extend(results[nid][name]['peaks'])
    bd.append(ap)
    bl.append(name)
bp = ax.boxplot(bd, labels=bl, patch_artist=True, widths=0.5)
for p, (name, c) in zip(bp['boxes'], COLORS.items()):
    p.set_facecolor(c)
    p.set_alpha(0.7)
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax.set_ylabel('Peak Water Level Error (m)', fontsize=12)
ax.set_title('T=100yr: Peak Error Distribution -- All Models Across Active Nodes', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for i, d in enumerate(bd):
    if d:
        ax.annotate(f'mean={np.mean(d):.3f}m', xy=(i+1, np.mean(d)),
                    xytext=(i+1.35, np.mean(d)+0.005),
                    fontsize=8, color='darkred', fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(FIGS, 'peak_error_boxplot.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved: peak_error_boxplot.png')

# ---- FIG 3: Per-node peak error box ----
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()
for idx, nid in enumerate(NODES):
    ax = axes[idx]
    bd = []
    for name in models:
        bd.append(results[nid][name]['peaks'])
    bp = ax.boxplot(bd, labels=models, patch_artist=True, widths=0.5)
    for p, name in zip(bp['boxes'], models):
        p.set_facecolor(COLORS[name])
        p.set_alpha(0.7)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    dmax = results[nid][models[0]]['swmm_max']
    ax.set_title(f'{nid} (max SWMM depth={dmax:.3f}m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Peak Error (m)')
    ax.grid(True, alpha=0.3, axis='y')
fig.suptitle('T=100yr: Peak Error Distribution by Node', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(FIGS, 'peak_error_per_node.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved: peak_error_per_node.png')

print('\n[DONE] All results in:', OUT)
