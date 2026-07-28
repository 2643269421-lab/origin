#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Complete remaining training + evaluation + generate supplementary results
"""
import sys, os, io, json, warnings, time
warnings.filterwarnings('ignore')
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

os.chdir(r'C:\Users\26432\Desktop\origin\origin\SWMM2AI-Experiment-master1')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import torch
from model import Trainer, Predictor
from dataset import SWMMDataset

PKG = r'C:\Users\26432\Desktop\FINAL_PACKAGE\supplementary_models'
PROJ = r'C:\Users\26432\Desktop\origin\origin\SWMM2AI-Experiment-master1'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SEQ = 288
DATA_DIR = 'Actual Rainfall_Water Level'

os.makedirs(PKG, exist_ok=True)

# ============================================================
# 1. Train CA-LSTM+PeakOnly (CA-LSTM + lambda_peak=0.05, no smooth)
# ============================================================
mpath = os.path.join(PKG, 'CA-LSTM+PeakOnly.pth')
if not os.path.exists(mpath) or os.path.getsize(mpath) < 1000:
    print('=== Training CA-LSTM+PeakOnly ===')
    torch.manual_seed(42); np.random.seed(42)
    t = Trainer(
        model_type='CausalAttentionLSTM',
        model_params={'input_size': 1, 'hidden_size': 128, 'num_layers': 2, 'output_size': 1},
        model_path=mpath, device=DEV
    )
    model, ds = t.train(
        n_events=30, seq_length=SEQ, epochs=30,
        loss_type='physically_consistent',
        lambda_smooth=0.0, lambda_peak=0.05,
        max_return_period=10,
        template_inp_path='template.inp'
    )
    print(f'CA-LSTM+PeakOnly saved: {os.path.getsize(mpath)/1024:.0f} KB')
else:
    print('CA-LSTM+PeakOnly already exists')

# ============================================================
# 2. Train PCCA-LSTM (full: lambda_smooth=0.01, lambda_peak=0.05)
# ============================================================
mpath2 = os.path.join(PKG, 'pcca_lstm_full.pth')
if not os.path.exists(mpath2) or os.path.getsize(mpath2) < 1000:
    print('\n=== Training PCCA-LSTM (full physics loss) ===')
    torch.manual_seed(42); np.random.seed(42)
    t2 = Trainer(
        model_type='PCCA-LSTM',
        model_params={'input_size': 1, 'hidden_size': 128, 'num_layers': 2, 'output_size': 1},
        model_path=mpath2, device=DEV
    )
    model2, ds2 = t2.train(
        n_events=30, seq_length=SEQ, epochs=30,
        loss_type='physically_consistent',
        lambda_smooth=0.01, lambda_peak=0.05,
        max_return_period=10,
        template_inp_path='template.inp'
    )
    print(f'PCCA-LSTM full saved: {os.path.getsize(mpath2)/1024:.0f} KB')
else:
    print('PCCA-LSTM full already exists')

# ============================================================
# 3. Evaluate all models on real rainfall events
# ============================================================
print('\n=== Evaluating on real rainfall events ===')

# Model paths
model_paths = {
    'PCCA-LSTM': os.path.join(PKG, 'pcca_lstm_quick.pth'),
    'CA+SmoothOnly': os.path.join(PKG, 'CA-LSTM+SmoothOnly.pth'),
    'CA+PeakOnly': os.path.join(PKG, 'CA-LSTM+PeakOnly.pth'),
    'PCCA-LSTM_full': os.path.join(PKG, 'pcca_lstm_full.pth'),
}

# Also check the original project-level models
project_models = {
    'LSTM': os.path.join(PROJ, 'lstm_model.pth') if 'PROJ' in dir() else None,
}

# Find files
files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.xlsx') and not f.startswith('~')])
print(f'Found {len(files)} rainfall data files, using up to 70')

results = {}

for model_name, path in model_paths.items():
    if not os.path.exists(path):
        print(f'  {model_name}: model file not found at {path}')
        continue

    print(f'  Evaluating {model_name}...')
    try:
        pred_obj = Predictor(model_path=path, output_dir=PKG, device=DEV)
    except Exception as e:
        print(f'    Failed to load: {e}')
        continue

    rmses, dasrs, maes, pks, nrmse_list = [], [], [], [], []
    skipped = 0
    for i, f in enumerate(files[:70]):
        try:
            df = pd.read_excel(os.path.join(DATA_DIR, f))
            rc = [c for c in df.columns if '雨强' in str(c)]
            wc = [c for c in df.columns if 'STORM_L_01' in str(c) or '液位' in str(c)]
            if rc and wc:
                rain = df[rc[0]].values[:SEQ].astype(float)
                water = df[wc[0]].values[:SEQ].astype(float)
                # Filter: need valid rainfall and water level range
                if len(rain) == SEQ and len(water) == SEQ and np.max(rain) > 5 and np.std(water) > 0.005:
                    preds = pred_obj.predict_batch(rain.reshape(1, SEQ, 1))
                    p, t = preds[0], water

                    rmse = np.sqrt(np.mean((p - t) ** 2))
                    mae = np.mean(np.abs(p - t))
                    dasr = np.mean(np.sign(np.diff(p)) == np.sign(np.diff(t))) * 100

                    # Peak error
                    peak_idx = np.argmax(t)
                    pk = p[peak_idx] - t.max()

                    # NRMSE (normalized by range)
                    t_range = t.max() - t.min()
                    nrmse = rmse / t_range if t_range > 0.001 else rmse

                    rmses.append(rmse)
                    maes.append(mae)
                    dasrs.append(dasr)
                    pks.append(pk)
                    nrmse_list.append(nrmse)
        except Exception as e:
            skipped += 1

    if rmses:
        results[model_name] = {
            'RMSE': (float(np.mean(rmses)), float(np.std(rmses))),
            'MAE': (float(np.mean(maes)), float(np.std(maes))),
            'DASR': (float(np.mean(dasrs)), float(np.std(dasrs))),
            'PeakErr': (float(np.mean(pks)), float(np.std(pks))),
            'NRMSE': (float(np.mean(nrmse_list)), float(np.std(nrmse_list))),
            'n_events': len(rmses),
        }
        print(f'    RMSE={np.mean(rmses):.4f}±{np.std(rmses):.4f} m, '
              f'MAE={np.mean(maes):.4f}±{np.std(maes):.4f} m, '
              f'DASR={np.mean(dasrs):.1f}±{np.std(dasrs):.1f}%, '
              f'PeakErr={np.mean(pks):.4f}±{np.std(pks):.4f} m, '
              f'NRMSE={np.mean(nrmse_list):.4f}±{np.std(nrmse_list):.4f} '
              f'({len(rmses)} events)')

# Save results
out_path = os.path.join(PKG, 'supplementary_results.json')
with open(out_path, 'w', encoding='utf-8') as ff:
    json.dump(results, ff, indent=2, ensure_ascii=False)
print(f'\nResults saved to {out_path}')

# ============================================================
# 4. Print summary table
# ============================================================
print('\n=== SUMMARY ===')
print(f'{"Model":<20s} {"RMSE":>10s} {"MAE":>10s} {"DASR%":>10s} {"PeakErr":>10s} {"NRMSE":>10s} {"N":>5s}')
print('-' * 75)
for name, r in sorted(results.items()):
    print(f'{name:<20s} {r["RMSE"][0]:10.4f} {r["MAE"][0]:10.4f} {r["DASR"][0]:10.1f} {r["PeakErr"][0]:10.4f} {r["NRMSE"][0]:10.4f} {r["n_events"]:5d}')

print('\nDone!')
