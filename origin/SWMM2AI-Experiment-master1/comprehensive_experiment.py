#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
方案A 完整实验脚本
===================
1. 创建第二个SWMM管网（环状拓扑 + 大尺度）
2. 两个管网上的消融实验 + 中间消融 + 权重敏感性
3. 真实降雨验证
4. 生成全部结果表
"""

import sys, os, json, time, io, copy, shutil
import numpy as np
import pandas as pd

# Force UTF-8 on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

PROJECT_ROOT = r'C:\Users\26432\Desktop\origin\origin\SWMM2AI-Experiment-master1'
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

from model import Trainer, Predictor
from dataset import SWMMDataset
from registry import create_model
from physics_loss import PhysicallyConsistentLoss
from swmm.simulator import SWMMSimulator
from swmm.rainfall.generator import RainfallGenerator

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_ROOT = os.path.join('output', f'comprehensive_{TIMESTAMP}')
os.makedirs(OUTPUT_ROOT, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_TRAIN = 500
TRAIN_MAX_RP = 10
EXTREME_RPS = [20, 30, 50, 100]
N_TEST_RP = 15
SEQ_LEN = 288; DT = 5
EPOCHS = 200; BS = 32; LR = 0.001
N_SEEDS = 3
SEEDS = [42, 123, 456]

# ---- Model configs ----
MODEL_NAMES = ['LSTM', 'CA-LSTM', 'PCCA-LSTM']
MODEL_TYPES = {'LSTM': 'SimpleLSTM', 'CA-LSTM': 'CausalAttentionLSTM', 'PCCA-LSTM': 'PCCA-LSTM'}

# Intermediate ablation (M4)
INTERMEDIATE = {
    'CA-LSTM+Smooth': ('CausalAttentionLSTM', 'physically_consistent', 0.01, 0.0),
    'CA-LSTM+PeakTime': ('CausalAttentionLSTM', 'physically_consistent', 0.0, 0.05),
}

print(f"=== 方案A 完整实验 ===")
print(f"输出目录: {OUTPUT_ROOT}")
print(f"设备: {DEVICE}")
print(f"训练事件数: {N_TRAIN} (≤{TRAIN_MAX_RP}yr)")
print(f"测试重现期: {EXTREME_RPS}")
print(f"随机种子: {SEEDS}")


# ================================================================
# STEP 1: Create 2nd SWMM network (ring topology + larger scale)
# ================================================================
def create_ring_network():
    """创建环状管网 template.inp"""
    ring_dir = os.path.join(OUTPUT_ROOT, 'ring_network')
    os.makedirs(ring_dir, exist_ok=True)
    ring_inp = os.path.join(ring_dir, 'template_ring.inp')

    if os.path.exists(ring_inp):
        print(f"[SKIP] Ring network already exists: {ring_inp}")
        return ring_inp

    # Read original template
    orig = os.path.join(PROJECT_ROOT, 'template.inp')
    with open(orig, 'r', encoding='utf-8') as f:
        content = f.read()

    # Modifications to create a ring-topology + larger-scale network:
    # 1. Add 3 loop connections creating ring topology
    # 2. Increase pipe diameters by 1.5x (larger scale)
    # 3. Double subcatchment areas (larger scale)
    # 4. Change node elevations to create mild reverse slopes (ring feature)

    lines = content.split('\n')
    new_lines = []
    in_conduits = False
    in_xsections = False
    in_subcatchments = False
    in_junctions = False
    conduit_section_end = False

    for i, line in enumerate(lines):
        # Track sections
        if line.strip().startswith('[CONDUITS]'):
            in_conduits = True
            in_xsections = False
            in_subcatchments = False
            in_junctions = False
            new_lines.append(line)
            continue
        elif line.strip().startswith('[XSECTIONS]'):
            in_conduits = False
            in_xsections = True
            in_subcatchments = False
            in_junctions = False
            new_lines.append(line)
            continue
        elif line.strip().startswith('[SUBCATCHMENTS]'):
            in_conduits = False
            in_xsections = False
            in_subcatchments = True
            in_junctions = False
            new_lines.append(line)
            continue
        elif line.strip().startswith('[JUNCTIONS]'):
            in_conduits = False
            in_xsections = False
            in_subcatchments = False
            in_junctions = True
            new_lines.append(line)
            continue
        elif line.strip().startswith('[') and line.strip() != '[CONDUITS]' and line.strip() != '[XSECTIONS]' and line.strip() != '[SUBCATCHMENTS]' and line.strip() != '[JUNCTIONS]':
            in_conduits = False
            in_xsections = False
            in_subcatchments = False
            in_junctions = False

        # Modify conduit pipe diameters (1.5x)
        if in_xsections and line.strip() and not line.strip().startswith(';'):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    diam = float(parts[2])
                    parts[2] = f'{diam * 1.5:.2f}'
                    line = '    '.join(parts)
                except ValueError:
                    pass

        # Modify conduit roughness (more varied)
        if in_conduits and line.strip() and not line.strip().startswith(';'):
            parts = line.split()
            if len(parts) >= 6:
                try:
                    rou = float(parts[4])
                    # Vary roughness: 0.008 to 0.015
                    parts[4] = f'{rou * 1.3:.3f}'
                    line = '    '.join(parts)
                except ValueError:
                    pass

        # Double subcatchment areas
        if in_subcatchments and line.strip() and not line.strip().startswith(';'):
            parts = line.split()
            if len(parts) >= 5:
                try:
                    area = float(parts[3])
                    parts[3] = f'{area * 2.0:.3f}'
                    line = '    '.join(parts)
                except ValueError:
                    pass

        # Modify some junction elevations (mild variation)
        if in_junctions and line.strip() and not line.strip().startswith(';'):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    elev = float(parts[1])
                    # Add mild variation ±0.1m
                    import hashlib
                    h = int(hashlib.md5(parts[0].encode()).hexdigest()[:4], 16) % 200 - 100
                    parts[1] = f'{elev + h/1000:.2f}'
                    line = '    '.join(parts)
                except (ValueError, IndexError):
                    pass

        new_lines.append(line)

    # Add ring connections at the end of [CONDUITS]
    ring_links = [
        "SL_RING1    SN_049           SN_007           45.0       0.013       0          0          0          0",
        "SL_RING2    SN_007           SN_020           38.5       0.013       0          0          0          0",
        "SL_RING3    SN_020           SN_028           52.3       0.013       0          0          0          0",
    ]
    ring_xsects = [
        "SL_RING1    CIRCULAR     0.6              0          0          0          1",
        "SL_RING2    CIRCULAR     0.6              0          0          0          1",
        "SL_RING3    CIRCULAR     0.6              0          0          0          1",
    ]

    # Insert ring conduits before XSECTIONS for existing links
    # First, find conduits section and add ring links
    final_lines = []
    for line in new_lines:
        final_lines.append(line)
        if line.strip().startswith('SL_001'):
            # Add ring links after last original conduit
            for rl in ring_links:
                final_lines.append(rl)

    # Add ring xsects
    content2 = '\n'.join(final_lines)
    # Find last original xsect and add ring xsects after it
    for rl in ring_xsects:
        content2 = content2.replace('SL_001           CIRCULAR     0.7', 'SL_001           CIRCULAR     0.7\n' + '\n'.join(ring_xsects))
        break

    with open(ring_inp, 'w', encoding='utf-8') as f:
        f.write(content2)

    print(f"[OK] Ring network created: {ring_inp}")
    return ring_inp


# ================================================================
# STEP 2: Training function (multi-seed)
# ================================================================
def train_model(model_name, seed, template_inp, output_subdir):
    """Train one model with specific seed"""
    model_type = MODEL_TYPES[model_name]
    model_dir = os.path.join(output_subdir, model_name, f'seed_{seed}')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'model.pth')

    if os.path.exists(model_path):
        print(f"  [SKIP] {model_name} seed={seed} already trained")
        return model_path

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_params = {'input_size': 1, 'hidden_size': 128, 'num_layers': 2, 'output_size': 1}

    use_physics = (model_name == 'PCCA-LSTM')

    trainer = Trainer(
        model_type=model_type,
        model_params=model_params,
        model_path=model_path,
        device=DEVICE
    )

    try:
        model, dataset = trainer.train(
            n_events=N_TRAIN, seq_length=SEQ_LEN, epochs=EPOCHS,
            loss_type='physically_consistent' if use_physics else 'mse',
            lambda_smooth=0.01 if use_physics else 0,
            lambda_peak=0.05 if use_physics else 0,
            max_return_period=TRAIN_MAX_RP
        )
        print(f"  [OK] {model_name} seed={seed} trained")
    except Exception as e:
        print(f"  [FAIL] {model_name} seed={seed}: {e}")
        return None

    return model_path


def train_intermediate(name, model_type_str, loss_type, lam_s, lam_p, seed, output_subdir):
    """Train intermediate ablation model"""
    model_dir = os.path.join(output_subdir, name, f'seed_{seed}')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'model.pth')

    if os.path.exists(model_path):
        print(f"  [SKIP] {name} seed={seed}")
        return model_path

    torch.manual_seed(seed)
    np.random.seed(seed)

    model_params = {'input_size': 1, 'hidden_size': 128, 'num_layers': 2, 'output_size': 1}

    trainer = Trainer(
        model_type=model_type_str,
        model_params=model_params,
        model_path=model_path,
        device=DEVICE
    )

    try:
        model, dataset = trainer.train(
            n_events=N_TRAIN, seq_length=SEQ_LEN, epochs=EPOCHS,
            loss_type=loss_type,
            lambda_smooth=lam_s, lambda_peak=lam_p,
            max_return_period=TRAIN_MAX_RP
        )
        print(f"  [OK] {name} seed={seed} trained")
    except Exception as e:
        print(f"  [FAIL] {name} seed={seed}: {e}")
        return None

    return model_path


# ================================================================
# STEP 3: Evaluation function
# ================================================================
def evaluate_model(model_paths_by_seed, test_data):
    """
    Evaluate across seeds. model_paths_by_seed is dict: {seed: path}
    Returns dict of {metric: (mean, std)}
    """
    all_metrics = {rp: {} for rp in EXTREME_RPS}

    for rp in EXTREME_RPS:
        rain = test_data[rp]['rainfall']
        swmm_w = test_data[rp]['water_swmm']

        seed_results = {m: [] for m in ['RMSE', 'MAE', 'MAPE', 'R2', 'PeakErr', 'DH_agree', 'PeakIdxErr', 'Phi']}

        for seed, mpath in model_paths_by_seed.items():
            if mpath is None or not os.path.exists(mpath):
                continue
            predictor = Predictor(model_path=mpath, output_dir=OUTPUT_ROOT, device=DEVICE)
            preds = predictor.predict_batch(rain)

            per_sample = {m: [] for m in seed_results}
            for i in range(len(rain)):
                p_ = preds[i]; t_ = swmm_w[i]
                mse = np.mean((p_ - t_)**2)
                mae = np.mean(np.abs(p_ - t_))
                active = t_ > 0.001
                mape = np.mean(np.abs((p_[active]-t_[active])/(t_[active]+1e-10)))*100 if active.sum()>0 else 0
                ss_r, ss_t = np.sum((t_-p_)**2), np.sum((t_-np.mean(t_))**2)
                r2 = 1 - ss_r/(ss_t+1e-10)
                peak_err = p_[np.argmax(t_)] - t_.max()

                # DASR
                swmm_dh = np.diff(t_); pred_dh = np.diff(p_)
                dh_agree = np.mean(np.sign(pred_dh) == np.sign(swmm_dh)) * 100

                # Peak index error
                pk_idx_err = np.abs(np.argmax(p_) - np.argmax(t_)) * DT

                # Phi
                phi_pred = np.sum(p_) / (np.sum(rain[i]) * DT / 60 + 1e-10)
                phi_swmm = np.sum(t_) / (np.sum(rain[i]) * DT / 60 + 1e-10)

                per_sample['RMSE'].append(np.sqrt(mse))
                per_sample['MAE'].append(mae)
                per_sample['MAPE'].append(mape)
                per_sample['R2'].append(r2)
                per_sample['PeakErr'].append(peak_err)
                per_sample['DH_agree'].append(dh_agree)
                per_sample['PeakIdxErr'].append(pk_idx_err)
                per_sample['Phi'].append(phi_pred)

            for m in seed_results:
                seed_results[m].append(np.mean(per_sample[m]))

        for m in seed_results:
            if seed_results[m]:
                all_metrics[rp][m] = (float(np.mean(seed_results[m])), float(np.std(seed_results[m])))

    return all_metrics


def generate_test_data_for_network(template_inp, output_node='SN_001'):
    """Generate test data for a specific SWMM network"""
    print(f"\n生成测试数据 (node={output_node})...")
    test_data = {}
    for rp in EXTREME_RPS:
        print(f"  T={rp}yr x {N_TEST_RP}..."); t0 = time.time()
        rg = RainfallGenerator(time_step_min=DT)
        sim = SWMMSimulator(template_inp_path=template_inp, output_dir=OUTPUT_ROOT,
                           output_element=output_node, output_type='node', output_variable='depth')
        rains, waters = [], []
        for _ in range(N_TEST_RP):
            rain = rg.generate_rainfall_event(seq_length=SEQ_LEN, rain_type='chicago',
                                              duration_hours=np.random.uniform(1, 6),
                                              return_period=rp,
                                              peak_position=np.random.uniform(0.3, 0.7),
                                              start_idx=np.random.randint(0, 36))
            res = sim.run_swmm_simulation(rainfall_mm_h=rain)
            if res and len(res['values']) == SEQ_LEN:
                rains.append(rain); waters.append(res['values'])
        test_data[rp] = {'rainfall': np.array(rains), 'water_swmm': np.array(waters)}
        print(f"    {len(rains)}/{N_TEST_RP} valid, {time.time()-t0:.1f}s")
    return test_data


# ================================================================
# STEP 4: Train & evaluate ALL models on BOTH networks
# ================================================================
def run_all_experiments():
    # Create networks
    ring_inp = create_ring_network()
    orig_inp = os.path.join(PROJECT_ROOT, 'template.inp')

    networks = {
        'Original_16ha_Branch': orig_inp,
        'Ring_32ha_Loop': ring_inp,
    }

    all_results = {}

    for net_name, inp_path in networks.items():
        print(f"\n{'='*70}")
        print(f"  NETWORK: {net_name}")
        print(f"{'='*70}")

        net_dir = os.path.join(OUTPUT_ROOT, 'experiments', net_name.replace(' ', '_'))
        os.makedirs(net_dir, exist_ok=True)

        # ---- Train all models across all seeds ----
        model_paths = {}  # {name: {seed: path}}

        for name in MODEL_NAMES:
            model_paths[name] = {}
            for seed in SEEDS:
                print(f"\n  Training {name} seed={seed}...")
                mpath = train_model(name, seed, inp_path, net_dir)
                model_paths[name][seed] = mpath

        # ---- Train intermediate ablation (M4) ----
        for int_name, (mtype, ltype, ls, lp) in INTERMEDIATE.items():
            model_paths[int_name] = {}
            for seed in SEEDS[:1]:  # Only 1 seed for intermediate (save time)
                print(f"\n  Training {int_name} seed={seed}...")
                mpath = train_intermediate(int_name, mtype, ltype, ls, lp, seed, net_dir)
                model_paths[int_name][seed] = mpath

        # ---- Generate test data ----
        test_data = generate_test_data_for_network(inp_path)

        # ---- Evaluate all models ----
        net_results = {}
        for name, paths_by_seed in model_paths.items():
            print(f"\n  Evaluating {name}...")
            metrics = evaluate_model(paths_by_seed, test_data)
            net_results[name] = metrics

            # Print RMSE summary
            for rp in EXTREME_RPS:
                if 'RMSE' in metrics[rp]:
                    v = metrics[rp]['RMSE']
                    print(f"    T={rp}yr RMSE={v[0]:.5f}±{v[1]:.5f} m")

        all_results[net_name] = net_results

    return all_results


# ================================================================
# STEP 5: Real-rainfall validation
# ================================================================
def real_rainfall_validation(model_paths_per_seed):
    """
    Test trained models on real rainfall data.
    model_paths_per_seed is {model_name: {seed: path}} for SN_001 models.
    """
    print(f"\n{'='*70}")
    print(f"  REAL RAINFALL VALIDATION")
    print(f"{'='*70}")

    data_dir = os.path.join(PROJECT_ROOT, 'Actual Rainfall_Water Level')
    xlsx_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.xlsx') and not f.startswith('~')])

    if not xlsx_files:
        print("  [WARN] No real rainfall data found!")
        return None

    # Column mapping: '雨强(mm/hr)' = rainfall, 'HUZU_STORM_L_01_液位' = water level at storm drain 1
    all_events = []
    for f in xlsx_files[:30]:  # Use first 30 real events
        fpath = os.path.join(data_dir, f)
        df = pd.read_excel(fpath)
        rain_col = [c for c in df.columns if '雨强' in str(c)]
        water_col = [c for c in df.columns if 'STORM_L_01' in str(c) or '液位' in str(c)]

        if rain_col and water_col:
            rain = df[rain_col[0]].values[:SEQ_LEN].astype(float)
            water = df[water_col[0]].values[:SEQ_LEN].astype(float)
            if len(rain) == SEQ_LEN and len(water) == SEQ_LEN:
                all_events.append({'name': f.replace('.xlsx', ''), 'rain': rain, 'water': water})

    print(f"  Loaded {len(all_events)} real rainfall events")

    if not all_events:
        return None

    # Evaluate
    real_results = {}
    for name, paths_by_seed in model_paths_per_seed.items():
        print(f"\n  Evaluating {name} on real data...")
        real_results[name] = {'RMSE': [], 'MAE': [], 'DASR': [], 'PeakErr': [], 'R2': []}

        for evt in all_events:
            seed_metrics = {k: [] for k in real_results[name]}
            for seed, mpath in paths_by_seed.items():
                if mpath is None or not os.path.exists(mpath):
                    continue
                predictor = Predictor(model_path=mpath, output_dir=OUTPUT_ROOT, device=DEVICE)
                rain_arr = evt['rain'].reshape(1, -1, 1)
                # Normalize same as training
                water_arr = evt['water']

                pred = predictor.predict(rain_arr)[0]

                mse = np.mean((pred - water_arr)**2)
                mae = np.mean(np.abs(pred - water_arr))
                dh_agree = np.mean(np.sign(np.diff(pred)) == np.sign(np.diff(water_arr))) * 100
                peak_err = pred[np.argmax(water_arr)] - water_arr.max()
                ss_r, ss_t = np.sum((water_arr-pred)**2), np.sum((water_arr-np.mean(water_arr))**2)
                r2 = 1 - ss_r/(ss_t+1e-10)

                seed_metrics['RMSE'].append(np.sqrt(mse))
                seed_metrics['MAE'].append(mae)
                seed_metrics['DASR'].append(dh_agree)
                seed_metrics['PeakErr'].append(peak_err)
                seed_metrics['R2'].append(r2)

            for k in real_results[name]:
                if seed_metrics[k]:
                    real_results[name][k].append(np.mean(seed_metrics[k]))

        # Summarize
        for k in real_results[name]:
            vals = real_results[name][k]
            if vals:
                print(f"    {k}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    return real_results


# ================================================================
# STEP 6: Weight sensitivity on both networks
# ================================================================
def run_sensitivity(template_inp, net_name, output_dir):
    """M6: Lambda sensitivity on one representative seed"""
    print(f"\n{'='*70}")
    print(f"  M6: Weight Sensitivity for {net_name}")
    print(f"{'='*70}")

    lam_s_vals = [0.001, 0.005, 0.01, 0.05, 0.1]
    lam_p_vals = [0.005, 0.01, 0.05, 0.1, 0.2]

    # Use 5 test events for quick eval
    rg = RainfallGenerator(time_step_min=DT)
    sim = SWMMSimulator(template_inp_path=template_inp, output_dir=output_dir,
                       output_element='SN_001', output_type='node', output_variable='depth')
    val_rains, val_waters = [], []
    for _ in range(5):
        rain = rg.generate_rainfall_event(seq_length=SEQ_LEN, rain_type='chicago',
                                          duration_hours=np.random.uniform(2, 5),
                                          return_period=100,
                                          peak_position=np.random.uniform(0.3, 0.7))
        res = sim.run_swmm_simulation(rainfall_mm_h=rain)
        if res and len(res['values']) == SEQ_LEN:
            val_rains.append(rain); val_waters.append(res['values'])
    val_rains, val_waters = np.array(val_rains), np.array(val_waters)

    results = []

    for ls in lam_s_vals:
        for lp in lam_p_vals:
            tag = f'sens_s{ls}_p{lp}'.replace('.', '_')
            mpath = os.path.join(output_dir, f'sensitivity_{tag}.pth')

            if os.path.exists(mpath):
                print(f"  [SKIP] λs={ls}, λp={lp}")
            else:
                torch.manual_seed(42)
                np.random.seed(42)
                trainer = Trainer(
                    model_type='PCCA-LSTM',
                    model_params={'input_size': 1, 'hidden_size': 128, 'num_layers': 2, 'output_size': 1},
                    model_path=mpath, device=DEVICE
                )
                trainer.train(n_events=200, seq_length=SEQ_LEN, epochs=100,  # fewer epochs for sensitivity
                             loss_type='physically_consistent',
                             lambda_smooth=ls, lambda_peak=lp,
                             max_return_period=TRAIN_MAX_RP)

            # Evaluate
            predictor = Predictor(model_path=mpath, output_dir=output_dir, device=DEVICE)
            preds = predictor.predict_batch(val_rains)
            rmses = [np.sqrt(np.mean((p-t)**2)) for p, t in zip(preds, val_waters)]
            dhs = [np.mean(np.sign(np.diff(p))==np.sign(np.diff(t)))*100 for p, t in zip(preds, val_waters)]

            results.append({
                'lambda_smooth': ls, 'lambda_peak': lp,
                'RMSE': float(np.mean(rmses)), 'DASR': float(np.mean(dhs)),
            })
            print(f"  λs={ls:.3f} λp={lp:.3f} → RMSE={results[-1]['RMSE']:.5f} DASR={results[-1]['DASR']:.1f}%")

    # Save
    with open(os.path.join(output_dir, 'sensitivity.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ================================================================
# MAIN
# ================================================================
def main():
    print(f"\n开始时间: {datetime.now()}")
    t_start = time.time()

    # ---- Phase 1: Run experiments on both networks ----
    all_results = run_all_experiments()

    # ---- Phase 2: Sensitivity ----
    orig_inp = os.path.join(PROJECT_ROOT, 'template.inp')
    ring_inp = create_ring_network()

    sens_orig = run_sensitivity(orig_inp, 'Original_16ha_Branch',
                                os.path.join(OUTPUT_ROOT, 'experiments', 'Original_16ha_Branch'))
    sens_ring = run_sensitivity(ring_inp, 'Ring_32ha_Loop',
                                os.path.join(OUTPUT_ROOT, 'experiments', 'Ring_32ha_Loop'))

    # ---- Phase 3: Real-rainfall validation ----
    # Use the original network models
    orig_model_paths = {}
    for name in MODEL_NAMES:
        orig_model_paths[name] = {}
        net_dir = os.path.join(OUTPUT_ROOT, 'experiments', 'Original_16ha_Branch')
        for seed in SEEDS:
            mpath = os.path.join(net_dir, name, f'seed_{seed}', 'model.pth')
            if os.path.exists(mpath):
                orig_model_paths[name][seed] = mpath

    real_results = real_rainfall_validation(orig_model_paths)

    # ---- Phase 4: Save all results ----
    full_report = {
        'timestamp': TIMESTAMP,
        'config': {'n_train': N_TRAIN, 'train_max_rp': TRAIN_MAX_RP,
                   'test_rps': EXTREME_RPS, 'n_test_rp': N_TEST_RP, 'seeds': SEEDS},
        'results': all_results,
        'sensitivity_original': sens_orig,
        'sensitivity_ring': sens_ring,
        'real_rainfall': real_results,
    }

    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    full_report = convert(full_report)

    report_path = os.path.join(OUTPUT_ROOT, 'full_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)

    # ---- Phase 5: Generate result tables for paper ----
    generate_paper_tables(full_report)

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  ALL DONE! ({elapsed/60:.1f} min)")
    print(f"  Results: {OUTPUT_ROOT}")
    print(f"  Report: {report_path}")
    print(f"{'='*70}")


def generate_paper_tables(report):
    """Generate LaTeX-ready result tables from the full report"""
    tables_dir = os.path.join(OUTPUT_ROOT, 'paper_tables')
    os.makedirs(tables_dir, exist_ok=True)

    results = report['results']

    # Table: Cross-network comparison (T=100yr)
    print("\n=== TABLE: Cross-Network Comparison (T=100yr) ===")
    header = f"{'Model':<25} {'Orig_RMSE':>12} {'Ring_RMSE':>12} {'Orig_DASR':>10} {'Ring_DASR':>10} {'Orig_R2':>8} {'Ring_R2':>8}"
    print(header)
    print('-' * len(header))

    for name in MODEL_NAMES:
        try:
            o = results['Original_16ha_Branch'][name][100]
            r = results['Ring_32ha_Loop'][name][100]
            print(f"{name:<25} {o['RMSE'][0]:>10.5f}±{o['RMSE'][1]:.5f} {r['RMSE'][0]:>10.5f}±{r['RMSE'][1]:.5f} "
                  f"{o['DH_agree'][0]:>8.1f}% {r['DH_agree'][0]:>8.1f}% "
                  f"{o['R2'][0]:>6.3f} {r['R2'][0]:>6.3f}")
        except (KeyError, IndexError):
            print(f"{name:<25} [data missing]")

    # Table: Intermediate ablation (M4)
    print("\n=== TABLE: M4 Intermediate Ablation (T=100yr, Original) ===")
    for name in MODEL_NAMES + list(INTERMEDIATE.keys()):
        try:
            r = results['Original_16ha_Branch'][name][100]
            print(f"{name:<25} RMSE={r['RMSE'][0]:.5f}±{r['RMSE'][1]:.5f} "
                  f"MAE={r['MAE'][0]:.5f}±{r['MAE'][1]:.5f} "
                  f"DASR={r['DH_agree'][0]:.1f}±{r['DH_agree'][1]:.1f}%")
        except (KeyError, IndexError):
            print(f"{name:<25} [data missing]")


if __name__ == '__main__':
    main()
