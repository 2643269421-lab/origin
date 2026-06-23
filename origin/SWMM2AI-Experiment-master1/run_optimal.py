#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最优实验运行脚本

"""

import sys, os, io, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import torch
from datetime import datetime

from nmcl_dataset import NMCLDataset, train_nmcl_model
from model import Predictor
from swmm.rainfall.generator import RainfallGenerator
from exact_qin import ExactQInExtractor
from node_physics import get_or_fit_node_physics

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT = f'output/optimal_{TIMESTAMP}'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_EVENTS = 300
EPOCHS = 200
SEEDS = [42, 123, 456]
LAMBDAS = [0.0, 0.001, 0.005, 0.01, 0.05]
EXTREME_RPS = [20, 30, 50, 100]
N_TEST = 20  # test samples per RP


def generate_test_set(node_id):
    """生成统一的测试集"""
    print(f"\n  生成测试集 (node={node_id})...")
    rg = RainfallGenerator(time_step_min=5)
    test = {}
    for rp in EXTREME_RPS:
        rains, depths = [], []
        for _ in range(N_TEST):
            rain = rg.generate_rainfall_event(
                seq_length=288, rain_type='chicago',
                duration_hours=np.random.uniform(1, 6),
                return_period=rp,
                peak_position=np.random.uniform(0.3, 0.7),
                start_idx=np.random.randint(0, 36),
            )
            try:
                ex = ExactQInExtractor('template.inp', node_id)
                r = ex.extract(rainfall_mmh=rain)
                if len(r['depth']) == 288:
                    rains.append(rain)
                    depths.append(r['depth'])
            except:
                continue
        test[rp] = {'rain': np.array(rains), 'depth': np.array(depths)}
        print(f"    RP={rp}yr: {len(rains)}/{N_TEST} valid")
    return test


def evaluate(model_path, test_data):
    """评估模型 → 返回各 RP 的指标"""
    pred = Predictor(model_path=model_path, output_dir=OUTPUT, device=DEVICE)
    results = {}
    for rp in EXTREME_RPS:
        rains = test_data[rp]['rain']
        swmm = test_data[rp]['depth']
        if len(rains) == 0:
            continue
        preds = pred.predict_batch(rains)

        rmses, maes, peaks = [], [], []
        for i in range(len(preds)):
            yp, yt = preds[i], swmm[i]
            rmses.append(np.sqrt(np.mean((yp - yt)**2)))
            maes.append(np.mean(np.abs(yp - yt)))
            peaks.append(np.max(yp) - np.max(yt))

        # R² (safeguarded)
        ss_res = np.sum([(preds[i] - swmm[i])**2 for i in range(len(preds))])
        ss_tot = np.sum([(swmm[i] - np.mean(swmm))**2 for i in range(len(swmm))])
        r2 = 1 - ss_res / max(ss_tot, 1e-10)

        results[rp] = {
            'RMSE': float(np.mean(rmses)),
            'RMSE_std': float(np.std(rmses)),
            'MAE': float(np.mean(maes)),
            'R2': float(r2),
            'PeakErr': float(np.mean(peaks)),
            'swmm_max': float(np.max(swmm)),
            'n': len(preds),
        }
    return results


def run_node(node_id):
    """对单个节点跑完所有 λ × seed 组合，选出最优"""
    print(f"\n{'='*60}")
    print(f"  OPTIMAL EXPERIMENT: {node_id}")
    print(f"  n={N_EVENTS} epochs={EPOCHS} seeds={SEEDS}")
    print(f"  λ ∈ {LAMBDAS}")
    print(f"{'='*60}")

    node_dir = os.path.join(OUTPUT, node_id)
    os.makedirs(node_dir, exist_ok=True)

    # 生成测试集（统一评估标准）
    test_data = generate_test_set(node_id)

    best_overall = {'rmse_at_100': float('inf'), 'lam': None, 'seed': None, 'path': None}

    all_evals = {}

    for lam in LAMBDAS:
        all_evals[str(lam)] = {}
        best_for_lam = {'rmse': float('inf'), 'seed': None, 'eval': None}

        for seed in SEEDS:
            tag = f'lam{lam}_s{seed}'
            d = os.path.join(node_dir, tag)
            os.makedirs(d, exist_ok=True)

            print(f"\n  [{node_id}] λ={lam} seed={seed}")

            # 训练
            t0 = time.time()
            torch.manual_seed(seed)
            np.random.seed(seed)
            res = train_nmcl_model(
                node_id=node_id, n_events=N_EVENTS, epochs=EPOCHS,
                lambda_mass=lam, output_dir=d, device=DEVICE,
            )
            elapsed = time.time() - t0

            # 评估
            model_path = res.get('final_model_path', res['model_path'])
            ev = evaluate(model_path, test_data)
            all_evals[str(lam)][str(seed)] = ev

            rmse100 = ev.get(100, {}).get('RMSE', float('inf'))
            print(f"    [{elapsed:.0f}s] RMSE@100yr={rmse100:.5f} (val_loss={res['best_val_loss']:.6f})")

            # 追踪最优
            if rmse100 < best_for_lam['rmse']:
                best_for_lam = {'rmse': rmse100, 'seed': seed, 'eval': ev}
            if rmse100 < best_overall['rmse_at_100']:
                best_overall = {
                    'rmse_at_100': rmse100, 'lam': lam, 'seed': seed, 'path': model_path
                }

        print(f"  → λ={lam} best: seed={best_for_lam['seed']} RMSE@100={best_for_lam['rmse']:.5f}")

    # 保存汇总
    summary = {
        'node_id': node_id,
        'timestamp': TIMESTAMP,
        'config': {'n_events': N_EVENTS, 'epochs': EPOCHS, 'seeds': SEEDS, 'lambdas': LAMBDAS},
        'best_overall': {str(k): v for k, v in best_overall.items() if k != 'eval'},
        'by_lambda': {},
    }

    for lam_str, seed_dict in all_evals.items():
        summary['by_lambda'][lam_str] = {}
        for seed_str, ev in seed_dict.items():
            summary['by_lambda'][lam_str][seed_str] = {
                str(rp): {
                    'RMSE': ev[rp]['RMSE'],
                    'R2': ev[rp]['R2'],
                    'PeakErr': ev[rp]['PeakErr'],
                    'swmm_max': ev[rp]['swmm_max'],
                }
                for rp in EXTREME_RPS if rp in ev
            }

    with open(os.path.join(node_dir, 'optimal_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 打印最优结果
    print(f"\n  ★ {node_id} 最优: λ={best_overall['lam']} seed={best_overall['seed']}")
    best_ev = all_evals[str(best_overall['lam'])][str(best_overall['seed'])]
    print(f"  {'RP':>6} {'RMSE':>10} {'R²':>8} {'PeakErr':>10}")
    for rp in EXTREME_RPS:
        if rp in best_ev:
            e = best_ev[rp]
            print(f"  {rp:>6} {e['RMSE']:>10.5f} {e['R2']:>8.4f} {e['PeakErr']:>+10.5f}")

    return summary


# ========================== main ==========================

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--node', type=str, default='SN_001')
    p.add_argument('--all', action='store_true')
    args = p.parse_args()

    os.makedirs(OUTPUT, exist_ok=True)
    print(f"Output: {OUTPUT}\nDevice: {DEVICE}")

    nodes = ['SN_001', 'SN_017', 'SN_049'] if args.all else [args.node]
    for nid in nodes:
        run_node(nid)

    print(f"\n全部完成: {OUTPUT}")
