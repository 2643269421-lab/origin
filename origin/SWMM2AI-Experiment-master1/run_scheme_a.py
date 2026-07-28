#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pragmatic full experiment run for Scheme A:
- Trains on ORIGINAL network only (ring network experiments take too long to run live)
- Generates ring network INP file for later use
- Adds real-rainfall validation
- Outputs all paper-ready results
"""

import sys, os, json, time, io, shutil
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

PROJECT_ROOT = r'C:\Users\26432\Desktop\origin\origin\SWMM2AI-Experiment-master1'
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from datetime import datetime
import torch
import torch.nn as nn
from model import Trainer, Predictor
from dataset import SWMMDataset
from registry import create_model
from physics_loss import PhysicallyConsistentLoss
from swmm.simulator import SWMMSimulator
from swmm.rainfall.generator import RainfallGenerator

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join('output', f'schemeA_{TIMESTAMP}')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_TRAIN, TRAIN_MAX_RP = 500, 10
EXTREME_RPS = [20, 30, 50, 100]
N_TEST_PER_RP = 15
SEQ_LEN, DT = 288, 5
EPOCHS, BS, LR = 200, 32, 0.001
SEEDS = [42, 123, 456]
N_REAL_EVENTS = 30  # real rainfall events to use

print(f"=== 方案A 实验运行 ===")
print(f"输出目录: {OUTPUT_DIR}")
print(f"设备: {DEVICE} | 训练事件: {N_TRAIN} (≤{TRAIN_MAX_RP}yr)")
print(f"极端测试重现期: {EXTREME_RPS} | 每种: {N_TEST_PER_RP} 事件")
print(f"随机种子: {SEEDS}")
print()

# ============================================================
# 1. CREATE RING NETWORK
# ============================================================
print("="*60)
print(" STEP 1: 创建环状管网模板")
print("="*60)

RING_INP = os.path.join(OUTPUT_DIR, 'template_ring.inp')
ORIG_INP = os.path.join(PROJECT_ROOT, 'template.inp')

with open(ORIG_INP, 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')
new_lines = []
section = None

for line in lines:
    stripped = line.strip()
    if stripped.startswith('[CONDUITS]'): section = 'conduits'
    elif stripped.startswith('[XSECTIONS]'): section = 'xsections'
    elif stripped.startswith('[SUBCATCHMENTS]'): section = 'subcatchments'
    elif stripped.startswith('[JUNCTIONS]'): section = 'junctions'
    elif stripped == '' or stripped.startswith('['):
        if stripped and stripped != '[CONDUITS]' and stripped != '[XSECTIONS]' and stripped != '[SUBCATCHMENTS]' and stripped != '[JUNCTIONS]':
            section = None

    if section == 'xsections' and stripped and not stripped.startswith(';'):
        parts = line.split()
        if len(parts) >= 3:
            try:
                d = float(parts[2])
                parts[2] = f'{d * 1.5:.2f}'
                line = parts[0].ljust(16) + parts[1].ljust(15) + parts[2].ljust(18) + '  '.join(parts[3:]) if len(parts) > 3 else f'{parts[0]:<16}{parts[1]:<15}{parts[2]:<18}'
            except: pass

    if section == 'subcatchments' and stripped and not stripped.startswith(';'):
        parts = line.split()
        if len(parts) >= 5:
            try:
                a = float(parts[3])
                parts[3] = f'{a * 2.0:.3f}'
            except: pass

    new_lines.append(line)

# Add ring links after last conduit
ring_conduits = [
    "SL_RING1           SN_049           SN_007           45.00      0.013       0          0          0          0",
    "SL_RING2           SN_007           SN_020           38.50      0.013       0          0          0          0",
    "SL_RING3           SN_020           SN_028           52.30      0.013       0          0          0          0",
]
ring_xsects = [
    "SL_RING1           CIRCULAR     0.60             0          0          0          1",
    "SL_RING2           CIRCULAR     0.60             0          0          0          1",
    "SL_RING3           CIRCULAR     0.60             0          0          0          1",
]

final_lines = []
section = None
added_ring_conduits = False
added_ring_xsects = False
for line in new_lines:
    stripped = line.strip()
    if stripped.startswith('[CONDUITS]'): section = 'conduits'
    elif stripped.startswith('[XSECTIONS]'): section = 'xsections'
    elif stripped.startswith('['):
        if stripped != '[CONDUITS]' and stripped != '[XSECTIONS]':
            section = stripped
        else:
            section = stripped[1:-1].lower()

    if section == 'conduits' and not added_ring_conduits and stripped.startswith('SL_001'):
        final_lines.append(line)
        for rc in ring_conduits:
            final_lines.append(rc)
        added_ring_conduits = True
        continue

    if section == 'xsections' and not added_ring_xsects and stripped.startswith('SL_001'):
        final_lines.append(line)
        for rx in ring_xsects:
            final_lines.append(rx)
        added_ring_xsects = True
        continue

    final_lines.append(line)

with open(RING_INP, 'w', encoding='utf-8') as f:
    f.write('\n'.join(final_lines))

print(f"[OK] 环状管网: {RING_INP}")

# ============================================================
# 2. GENERATE EXTREME TEST DATA (once, for original network)
# ============================================================
print("\n" + "="*60)
print(" STEP 2: 生成极端测试数据 (原始管网)")
print("="*60)

test_data = {}
for rp in EXTREME_RPS:
    print(f"  T={rp}年 x {N_TEST_PER_RP}...")
    t0 = time.time()
    rg = RainfallGenerator(time_step_min=DT)
    sim = SWMMSimulator(template_inp_path=ORIG_INP, output_dir=OUTPUT_DIR,
                       output_element='SN_001', output_type='node', output_variable='depth')
    rains, waters = [], []
    for _ in range(N_TEST_PER_RP):
        rain = rg.generate_rainfall_event(
            seq_length=SEQ_LEN, rain_type='chicago',
            duration_hours=np.random.uniform(1, 6),
            return_period=rp,
            peak_position=np.random.uniform(0.3, 0.7),
            start_idx=np.random.randint(0, 36))
        res = sim.run_swmm_simulation(rainfall_mm_h=rain)
        if res and len(res['values']) == SEQ_LEN:
            rains.append(rain); waters.append(res['values'])
    test_data[rp] = {'rainfall': np.array(rains), 'water_swmm': np.array(waters)}
    print(f"    {len(rains)}/{N_TEST_PER_RP} 有效, {time.time()-t0:.1f}s")

# Save test data for reuse
test_data_path = os.path.join(OUTPUT_DIR, 'test_data.npz')
np.savez(test_data_path, **{f'r{rp}_rain': test_data[rp]['rainfall'] for rp in EXTREME_RPS},
         **{f'r{rp}_water': test_data[rp]['water_swmm'] for rp in EXTREME_RPS})
print(f"[OK] 测试数据已保存: {test_data_path}")

# ============================================================
# 3. TRAIN ALL MODELS (original network, all seeds)
# ============================================================
print("\n" + "="*60)
print(" STEP 3: 训练所有模型 (原始管网)")
print("="*60)

MODEL_CONFIGS = [
    {'name': 'LSTM', 'type': 'SimpleLSTM', 'phys_loss': False},
    {'name': 'CA-LSTM', 'type': 'CausalAttentionLSTM', 'phys_loss': False},
    {'name': 'PCCA-LSTM', 'type': 'PCCA-LSTM', 'phys_loss': True},
]

INTERMEDIATE_ABLATION = [
    {'name': 'CA-LSTM+SmoothOnly', 'type': 'CausalAttentionLSTM',
     'phys_loss': True, 'lam_s': 0.01, 'lam_p': 0.0},
    {'name': 'CA-LSTM+PeakOnly', 'type': 'CausalAttentionLSTM',
     'phys_loss': True, 'lam_s': 0.0, 'lam_p': 0.05},
]

model_paths = {}

for cfg in MODEL_CONFIGS:
    name = cfg['name']
    model_paths[name] = {}
    for seed in SEEDS:
        model_dir = os.path.join(OUTPUT_DIR, 'models', name, f'seed_{seed}')
        os.makedirs(model_dir, exist_ok=True)
        mpath = os.path.join(model_dir, 'model.pth')

        if os.path.exists(mpath):
            print(f"  [SKIP] {name} seed={seed}")
            model_paths[name][seed] = mpath
            continue

        print(f"  Training {name} seed={seed}...")
        torch.manual_seed(seed)
        np.random.seed(seed)

        try:
            trainer = Trainer(
                model_type=cfg['type'],
                model_params={'input_size': 1, 'hidden_size': 128, 'num_layers': 2, 'output_size': 1},
                model_path=mpath,
                device=DEVICE
            )
            trainer.train(
                n_events=N_TRAIN, seq_length=SEQ_LEN, epochs=EPOCHS,
                loss_type='physically_consistent' if cfg['phys_loss'] else 'mse',
                lambda_smooth=0.01 if cfg['phys_loss'] else 0,
                lambda_peak=0.05 if cfg['phys_loss'] else 0,
                max_return_period=TRAIN_MAX_RP,
                template_inp_path=ORIG_INP
            )
            model_paths[name][seed] = mpath
            print(f"  [OK] {name} seed={seed}")
        except Exception as e:
            print(f"  [FAIL] {name} seed={seed}: {e}")
            import traceback; traceback.print_exc()

# Intermediate ablation (M4) - 1 seed only to save time
for cfg in INTERMEDIATE_ABLATION:
    name = cfg['name']
    model_paths[name] = {}
    seed = SEEDS[0]
    model_dir = os.path.join(OUTPUT_DIR, 'models', name, f'seed_{seed}')
    os.makedirs(model_dir, exist_ok=True)
    mpath = os.path.join(model_dir, 'model.pth')

    if os.path.exists(mpath):
        print(f"  [SKIP] {name} seed={seed}")
        model_paths[name][seed] = mpath
        continue

    print(f"  Training {name} seed={seed}...")
    torch.manual_seed(seed)
    np.random.seed(seed)

    try:
        trainer = Trainer(
            model_type=cfg['type'],
            model_params={'input_size': 1, 'hidden_size': 128, 'num_layers': 2, 'output_size': 1},
            model_path=mpath,
            device=DEVICE
        )
        trainer.train(
            n_events=N_TRAIN, seq_length=SEQ_LEN, epochs=EPOCHS,
            loss_type='physically_consistent',
            lambda_smooth=cfg['lam_s'], lambda_peak=cfg['lam_p'],
            max_return_period=TRAIN_MAX_RP,
            template_inp_path=ORIG_INP
        )
        model_paths[name][seed] = mpath
        print(f"  [OK] {name}")
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        import traceback; traceback.print_exc()

# ============================================================
# 4. EVALUATE ALL MODELS
# ============================================================
print("\n" + "="*60)
print(" STEP 4: 评估所有模型")
print("="*60)

def evaluate_models(model_paths_dict, test_data_dict):
    """Full evaluation across all models, seeds, return periods"""
    results = {}
    all_model_names = list(model_paths_dict.keys())

    for name in all_model_names:
        print(f"\n  评估 {name}...")
        results[name] = {}

        for rp in EXTREME_RPS:
            rain = test_data_dict[rp]['rainfall']
            swmm_w = test_data_dict[rp]['water_swmm']

            per_seed_metrics = {m: [] for m in
                ['RMSE', 'MAE', 'MAPE', 'R2', 'PeakErr', 'DASR', 'PeakIdx', 'Phi']}

            for seed, mpath in model_paths_dict[name].items():
                if mpath is None or not os.path.exists(mpath):
                    continue
                try:
                    predictor = Predictor(model_path=mpath, output_dir=OUTPUT_DIR, device=DEVICE)
                except Exception as e:
                    print(f"    [WARN] Cannot load {name} seed={seed}: {e}")
                    continue

                preds = predictor.predict_batch(rain)

                for i in range(len(rain)):
                    p, t = preds[i], swmm_w[i]
                    mse = np.mean((p - t)**2)
                    mae = np.mean(np.abs(p - t))
                    active = t > 0.001
                    mape = (np.mean(np.abs((p[active]-t[active])/(t[active]+1e-10)))*100
                            if active.sum() > 0 else 0)
                    ss_r, ss_t = np.sum((t-p)**2), np.sum((t-np.mean(t))**2)
                    r2 = 1 - ss_r/(ss_t+1e-10)
                    pk_err = p[np.argmax(t)] - t.max()
                    dasr = np.mean(np.sign(np.diff(p)) == np.sign(np.diff(t))) * 100
                    pk_idx = np.abs(np.argmax(p) - np.argmax(t)) * DT
                    phi = np.sum(p) / (np.sum(rain[i])*DT/60 + 1e-10)

                    per_seed_metrics['RMSE'].append(np.sqrt(mse))
                    per_seed_metrics['MAE'].append(mae)
                    per_seed_metrics['MAPE'].append(mape)
                    per_seed_metrics['R2'].append(r2)
                    per_seed_metrics['PeakErr'].append(pk_err)
                    per_seed_metrics['DASR'].append(dasr)
                    per_seed_metrics['PeakIdx'].append(pk_idx)
                    per_seed_metrics['Phi'].append(phi)

            results[name][rp] = {}
            for m, vals in per_seed_metrics.items():
                if vals:
                    results[name][rp][m] = (float(np.mean(vals)), float(np.std(vals)))

            if 'RMSE' in results[name][rp]:
                v = results[name][rp]['RMSE']
                print(f"    T={rp}yr: RMSE={v[0]:.5f}±{v[1]:.5f}m, DASR={results[name][rp].get('DASR', ('N/A',))[0]:.1f}%")

    return results

all_results = evaluate_models(model_paths, test_data)

# ============================================================
# 5. REAL RAINFALL VALIDATION
# ============================================================
print("\n" + "="*60)
print(" STEP 5: 真实降雨验证")
print("="*60)

data_dir = os.path.join(PROJECT_ROOT, 'Actual Rainfall_Water Level')
xlsx_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.xlsx') and not f.startswith('~')])

real_events = []
for f in xlsx_files[:N_REAL_EVENTS]:
    fpath = os.path.join(data_dir, f)
    try:
        df = pd.read_excel(fpath)
        rain_col = [c for c in df.columns if '雨强' in str(c)]
        water_col = [c for c in df.columns if 'STORM_L_01' in str(c) or '液位' in str(c)]

        if rain_col and water_col:
            rain = df[rain_col[0]].values[:SEQ_LEN].astype(float)
            water = df[water_col[0]].values[:SEQ_LEN].astype(float)
            if len(rain) == SEQ_LEN and len(water) == SEQ_LEN:
                # Filter: skip near-zero rainfall events
                if np.max(rain) > 5.0 and np.std(water) > 0.005:
                    real_events.append({'name': f.replace('.xlsx', ''),
                                        'rain': rain, 'water': water})
    except Exception as e:
        print(f"  [WARN] Cannot read {f}: {e}")

print(f"  加载 {len(real_events)} 个有效真实降雨事件")

real_results = {}
# Use only PCCA-LSTM for real validation
for name in ['LSTM', 'CA-LSTM', 'PCCA-LSTM']:
    real_results[name] = {'RMSE': [], 'MAE': [], 'DASR': [], 'PeakErr': []}

    for evt in real_events:
        seed_vals = {k: [] for k in real_results[name]}
        for seed, mpath in model_paths[name].items():
            if mpath is None or not os.path.exists(mpath):
                continue
            try:
                predictor = Predictor(model_path=mpath, output_dir=OUTPUT_DIR, device=DEVICE)
                rain_arr = evt['rain'].reshape(1, -1, 1)
                water_arr = evt['water']
                pred = predictor.predict(rain_arr)[0]

                mse = np.mean((pred - water_arr)**2)
                mae = np.mean(np.abs(pred - water_arr))
                dasr = np.mean(np.sign(np.diff(pred)) == np.sign(np.diff(water_arr))) * 100
                pk_err = pred[np.argmax(water_arr)] - water_arr.max()

                seed_vals['RMSE'].append(np.sqrt(mse))
                seed_vals['MAE'].append(mae)
                seed_vals['DASR'].append(dasr)
                seed_vals['PeakErr'].append(pk_err)
            except: pass

        for k in real_results[name]:
            if seed_vals[k]:
                real_results[name][k].append(np.mean(seed_vals[k]))

    vals = real_results[name]
    if vals['RMSE']:
        print(f"  {name}: RMSE={np.mean(vals['RMSE']):.4f}±{np.std(vals['RMSE']):.4f} m, "
              f"DASR={np.mean(vals['DASR']):.1f}±{np.std(vals['DASR']):.1f}%, "
              f"PeakErr={np.mean(vals['PeakErr']):.4f}±{np.std(vals['PeakErr']):.4f} m")

# ============================================================
# 6. SAVE ALL RESULTS & GENERATE PAPER TABLES
# ============================================================
print("\n" + "="*60)
print(" STEP 6: 保存结果 & 生成论文表格")
print("="*60)

# Convert to JSON-safe format
def to_json(obj):
    if isinstance(obj, dict):
        return {str(k): to_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_json(v) for v in obj]
    elif isinstance(obj, (np.integer,)): return int(obj)
    elif isinstance(obj, (np.floating,)): return float(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    return obj

full_report = {
    'timestamp': TIMESTAMP,
    'config': {
        'n_train': N_TRAIN, 'train_max_rp': TRAIN_MAX_RP,
        'test_return_periods': EXTREME_RPS, 'n_test_per_rp': N_TEST_PER_RP,
        'seq_len': SEQ_LEN, 'dt_min': DT, 'epochs': EPOCHS, 'lr': LR,
        'seeds': SEEDS,
    },
    'model_results': to_json(all_results),
    'real_rainfall': to_json(real_results),
}

report_path = os.path.join(OUTPUT_DIR, 'full_report.json')
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(full_report, f, ensure_ascii=False, indent=2)

# ---- Generate paper-ready tables ----
table_dir = os.path.join(OUTPUT_DIR, 'paper_tables')
os.makedirs(table_dir, exist_ok=True)

print("\n" + "-"*60)
print(" 核心结果表 — T=100yr (原文中格式)")
print("-"*60)

# Table A: Full comparison, all 5 models
hdr = f"{'Model':<28} {'RMSE(m)':>14} {'MAE(m)':>12} {'MAPE(%)':>10} {'R2':>8} {'DASR(%)':>10} {'PeakErr(m)':>12}"
print(hdr)
print('-'*len(hdr))
for name in MODEL_CONFIGS + INTERMEDIATE_ABLATION:
    n = name['name']
    if n in all_results and 100 in all_results[n]:
        r = all_results[n][100]
        print(f"{n:<28} {r['RMSE'][0]:.4f}±{r['RMSE'][1]:.4f}  {r['MAE'][0]:.4f}±{r['MAE'][1]:.4f}  "
              f"{r['MAPE'][0]:.2f}±{r['MAPE'][1]:.2f}  {r['R2'][0]:.3f}  {r['DASR'][0]:.1f}±{r['DASR'][1]:.1f}  "
              f"{r['PeakErr'][0]:.4f}±{r['PeakErr'][1]:.4f}")

# Table B: Cross-return-period DASR
print("\n\n" + "-"*60)
print(" DASR 多重现期对比 (%)")
print("-"*60)
hdr2 = f"{'Model':<28} " + "".join(f"{'T='+str(r)+'yr':>14}" for r in EXTREME_RPS)
print(hdr2)
print('-'*len(hdr2))
for name in MODEL_CONFIGS:
    n = name['name']
    vals = []
    for rp in EXTREME_RPS:
        if n in all_results and rp in all_results[n] and 'DASR' in all_results[n][rp]:
            v = all_results[n][rp]['DASR']
            vals.append(f"{v[0]:.1f}±{v[1]:.1f}")
        else:
            vals.append('N/A')
    print(f"{n:<28} " + "".join(f"{v:>14}" for v in vals))

# Table C: Real rainfall
print("\n\n" + "-"*60)
print(" 真实降雨验证")
print("-"*60)
hdr3 = f"{'Model':<28} {'RMSE(m)':>14} {'MAE(m)':>12} {'DASR(%)':>10} {'PeakErr(m)':>12}"
print(hdr3)
print('-'*len(hdr3))
for name in ['LSTM', 'CA-LSTM', 'PCCA-LSTM']:
    if name in real_results and real_results[name]['RMSE']:
        v = real_results[name]
        print(f"{name:<28} {np.mean(v['RMSE']):.4f}±{np.std(v['RMSE']):.4f}  "
              f"{np.mean(v['MAE']):.4f}±{np.std(v['MAE']):.4f}  "
              f"{np.mean(v['DASR']):.1f}±{np.std(v['DASR']):.1f}  "
              f"{np.mean(v['PeakErr']):.4f}±{np.std(v['PeakErr']):.4f}")

print(f"\n\n{'='*60}")
print(f" 实验完成！")
print(f" 结果目录: {OUTPUT_DIR}")
print(f" JSON报告: {report_path}")
print(f" 环状管网: {RING_INP}")
print(f"{'='*60}")
