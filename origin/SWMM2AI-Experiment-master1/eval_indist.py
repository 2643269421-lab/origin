#!/usr/bin/env python
# encoding: utf-8
"""跑分布内测试 (≤10yr) 保存结果 """
import sys, os, json, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
from swmm.simulator import SWMMSimulator
from swmm.rainfall.generator import RainfallGenerator
from model import Predictor

SEQ = 288; DT = 5; N_TEST = 20; MAX_RP = 10
MODEL_PATHS = {
    'LSTM': 'output/extreme_experiment_20260610_110041/extreme_simplelstm_model.pth',
    'GRU': 'output/extreme_experiment_20260610_110041/extreme_simplegru_model.pth',
    'Attn-LSTM': 'output/extreme_experiment_20260610_110041/extreme_attentionlstm_model.pth',
    'CA-LSTM': 'output/extreme_experiment_20260610_110041/extreme_causalattentionlstm_model.pth',
    'PCCA-LSTM': 'output/ablation_20260610_150250/pgca_lstm_model.pth',
}
OUT = 'output/indist_eval'
os.makedirs(OUT, exist_ok=True)
dev = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
print(f'Device: {dev}')

# ---- Load ----
preds = {}
for n, p in MODEL_PATHS.items():
    preds[n] = Predictor(model_path=p, output_dir=OUT, device=dev)
    print(f'[OK] {n}')

# ---- Generate ≤10yr test data (SN_001) ----
print(f'\nGenerating {N_TEST} events (≤{MAX_RP}yr)...')
rg = RainfallGenerator(time_step_min=DT)
sim = SWMMSimulator(template_inp_path='template.inp', output_dir=OUT,
                    output_element='SN_001', output_type='node', output_variable='depth')
rains, waters = [], []
for i in range(N_TEST):
    rp = np.random.choice([1, 2, 3, 5, 10])
    rain = rg.generate_rainfall_event(
        seq_length=SEQ, rain_type='chicago',
        duration_hours=np.random.uniform(1, 6),
        return_period=rp,
        peak_position=np.random.uniform(0.3, 0.7),
        start_idx=np.random.randint(0, 36))
    try:
        res = sim.run_swmm_simulation(rainfall_mm_h=rain)
        if res and res['values'] is not None and len(res['values']) == SEQ:
            rains.append(rain)
            waters.append(res['values'])
    except Exception:
        pass
rains = np.array(rains)
waters = np.array(waters)
print(f'{len(rains)}/{N_TEST} valid, max depth={waters.max():.3f}m')

# ---- Evaluate ----
results = {}
for name, pr in preds.items():
    pred = pr.predict_batch(rains)
    ms = []
    for i in range(len(rains)):
        p = pred[i]; t = waters[i]
        mse = np.mean((p - t)**2)
        mae = np.mean(np.abs(p - t))
        active = t > 0.001
        mape = np.mean(np.abs((p[active]-t[active])/(t[active]+1e-10)))*100 if active.sum()>0 else 0
        ssr = np.sum((t-p)**2); sst = np.sum((t - np.mean(t))**2)
        r2 = 1 - ssr/(sst+1e-10)
        pe = p[np.argmax(t)] - t.max()
        ms.append({'RMSE': np.sqrt(mse), 'MAE': mae, 'MAPE': mape, 'R2': r2, 'PeakErr': pe})

    results[name] = {
        'RMSE_mean': float(np.mean([m['RMSE'] for m in ms])),
        'RMSE_std': float(np.std([m['RMSE'] for m in ms])),
        'MAE_mean': float(np.mean([m['MAE'] for m in ms])),
        'MAE_std': float(np.std([m['MAE'] for m in ms])),
        'MAPE_mean': float(np.mean([m['MAPE'] for m in ms])),
        'R2_mean': float(np.mean([m['R2'] for m in ms])),
        'PeakErr_mean': float(np.mean([m['PeakErr'] for m in ms])),
    }
    v = results[name]
    print(f'{name:12s} RMSE={v["RMSE_mean"]:.5f} MAE={v["MAE_mean"]:.4f} '
          f'MAPE={v["MAPE_mean"]:.2f}% R2={v["R2_mean"]:.4f} PeakErr={v["PeakErr_mean"]:.4f}m')

# ---- Save ----
with open(os.path.join(OUT, 'indist_results.json'), 'w', encoding='utf-8') as f:
    json.dump({
        'config': {'n_test': N_TEST, 'max_return_period': MAX_RP, 'node': 'SN_001'},
        'results': results,
    }, f, ensure_ascii=False, indent=2)

print(f'\nSaved: {OUT}/indist_results.json')
print('\n=== 论文表格可引用数值 ===')
for name, v in results.items():
    print(f"  {name}: RMSE={v['RMSE_mean']:.4f}  MAE={v['MAE_mean']:.4f}  "
          f"MAPE={v['MAPE_mean']:.2f}%  R2={v['R2_mean']:.4f}  PeakErr={v['PeakErr_mean']:.4f}m")
