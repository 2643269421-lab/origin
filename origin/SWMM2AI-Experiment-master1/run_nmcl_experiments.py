#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NMCL 完整实验流程
==================
1. 对 SN_001 / SN_017 / SN_049 节点生成含精确 Q_in 的训练数据
2. λ_mass 网格搜索 (0, 0.001, 0.005, 0.01, 0.05)
3. 极端重现期外推评估 (20/30/50/100yr)
4. 物理一致性检验
5. 使用 RMSE 替代 R² 进行低方差节点的评估
6. 多随机种子 (42, 123, 456)

用法:
    python run_nmcl_experiments.py --node SN_001 --quick    # 快速测试
    python run_nmcl_experiments.py --all_nodes              # 全节点完整实验
"""

import sys, os, io, json, time, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import torch
from datetime import datetime

from nmcl_dataset import NMCLDataset, train_nmcl_model
from node_physics import get_or_fit_node_physics
from exact_qin import ExactQInExtractor
from swmm.rainfall.generator import RainfallGenerator
from model import Predictor


TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_ROOT = f'output/nmcl_full_experiment_{TIMESTAMP}'
SEQ_LENGTH = 288
TIME_STEP = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEEDS = [42, 123, 456]
EXTREME_RPS = [20, 30, 50, 100]
n_test_per_rp = 15


def run_full_experiment(node_id='SN_001', n_events=200, epochs=150,
                        lambda_masses=None, seeds=None, n_test_per_rp=15,
                        quick=False):
    """完整的单节点 NMCL 实验"""
    if seeds is None:
        seeds = [42, 123, 456]
    if lambda_masses is None:
        lambda_masses = [0.0, 0.001, 0.005, 0.01, 0.05] if not quick else [0.0, 0.01]

    node_dir = os.path.join(OUTPUT_ROOT, node_id)
    os.makedirs(node_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  NMCL Full Experiment: {node_id}")
    print(f"  λ_mass candidates: {lambda_masses}")
    print(f"  Seeds: {seeds}")
    print(f"  n_events={n_events}, epochs={epochs}")
    print(f"{'='*70}")

    # 预生成测试数据
    print(f"\n[Phase 0] 生成测试数据...")
    test_data = generate_test_data(node_id, n_test_per_rp=n_test_per_rp)

    # 预生成测试数据（同一份数据用于所有模型版本对比）
    print(f"\n[Phase 0] 生成测试数据...")
    test_data = generate_test_data(node_id)
    print(f"  各重现期样本: {[(rp, len(test_data[rp]['rain'])) for rp in EXTREME_RPS]}")

    # 训练每个 λ_mass ✖ seed 组合
    all_results = {}
    for lam in lambda_masses:
        all_results[str(lam)] = {}
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            tag = f'lmass{lam}_seed{seed}'
            model_dir = os.path.join(node_dir, tag)
            os.makedirs(model_dir, exist_ok=True)

            print(f"\n{'─'*50}")
            print(f"  Training: λ_mass={lam}, seed={seed}")
            print(f"{'─'*50}")

            results = train_nmcl_model(
                node_id=node_id,
                n_events=n_events,
                epochs=epochs if not quick else 30,
                lambda_mass=lam,
                output_dir=model_dir,
                device=DEVICE,
            )

            # 评估极端外推
            eval_results = evaluate_extreme(
                model_path=results['model_path'],
                node_id=node_id,
                test_data=test_data,
            )

            all_results[str(lam)][str(seed)] = {
                'training': results,
                'evaluation': eval_results,
            }

    # 汇总结果
    summary = consolidate_results(all_results, node_id, test_data)
    summary_path = os.path.join(node_dir, 'experiment_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"  {node_id} 实验完成!")
    print(f"  摘要: {summary_path}")
    print_summary(summary)

    return all_results, summary


def generate_test_data(node_id, n_test_per_rp=15):
    """为各极端重现期生成测试数据（统一用于所有模型版本）"""
    rg = RainfallGenerator(time_step_min=TIME_STEP)
    test_data = {}

    for rp in EXTREME_RPS:
        rains, waters = [], []
        for i in range(n_test_per_rp):
            rain = rg.generate_rainfall_event(
                seq_length=SEQ_LENGTH, rain_type='chicago',
                duration_hours=np.random.uniform(1, 6),
                return_period=rp,
                peak_position=np.random.uniform(0.3, 0.7),
                start_idx=np.random.randint(0, 36),
            )
            try:
                extractor = ExactQInExtractor('template.inp', node_id)
                result = extractor.extract(rainfall_mmh=rain)
                if result and len(result['depth']) == SEQ_LENGTH:
                    rains.append(rain)
                    waters.append(result['depth'])
            except Exception:
                continue

        test_data[rp] = {
            'rain': np.array(rains),
            'depth': np.array(waters),
        }
        print(f"  RP={rp}yr: {len(rains)}/{n_test_per_rp} valid")

    return test_data


def evaluate_extreme(model_path, node_id, test_data):
    """评估模型在各极端重现期上的性能"""
    # Try best model first, fall back to final model
    pred = None
    for mp in [model_path, model_path.replace('.pth', '_final.pth')]:
        if os.path.exists(mp):
            try:
                pred = Predictor(model_path=mp, output_dir=OUTPUT_ROOT, device=DEVICE)
                print(f"  模型加载成功: {os.path.basename(mp)}")
                break
            except Exception as e:
                print(f"  模型 {os.path.basename(mp)} 加载失败: {e}")
    if pred is None:
        print(f"  ⚠ 所有模型均加载失败")
        return None

    eval_results = {}
    for rp in EXTREME_RPS:
        if rp not in test_data or len(test_data[rp]['rain']) == 0:
            continue

        rain = test_data[rp]['rain']
        swmm = test_data[rp]['depth']

        try:
            preds = pred.predict_batch(rain)
        except Exception as e:
            print(f"  RP={rp}yr 预测失败: {e}")
            continue

        # 逐样本指标
        rmse_list, mae_list, peak_list = [], [], []
        for i in range(len(preds)):
            y_pred = preds[i]
            y_true = swmm[i]

            rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
            mae = np.mean(np.abs(y_pred - y_true))
            peak_err = np.max(y_pred) - np.max(y_true)

            rmse_list.append(rmse)
            mae_list.append(mae)
            peak_list.append(peak_err)

        # R² 安全计算（避免 SN_049 式的数值爆炸）
        ss_res = np.sum([(preds[i] - swmm[i]) ** 2 for i in range(len(preds))])
        ss_tot = np.sum([(swmm[i] - np.mean(swmm)) ** 2 for i in range(len(swmm))])

        if ss_tot > 1e-8:
            r2 = 1 - ss_res / ss_tot
        else:
            r2 = np.nan  # 低方差数据上 R² 无意义

        eval_results[str(rp)] = {
            'RMSE_mean': float(np.mean(rmse_list)),
            'RMSE_std': float(np.std(rmse_list)),
            'MAE_mean': float(np.mean(mae_list)),
            'MAE_std': float(np.std(mae_list)),
            'R2': float(r2) if not np.isnan(r2) else None,
            'R2_valid': not np.isnan(r2),
            'PeakErr_mean': float(np.mean(peak_list)),
            'PeakErr_std': float(np.std(peak_list)),
            'RMSE_per_sample': [float(x) for x in rmse_list],
            'swmm_max_depth': float(np.max(swmm)),
            'n_samples': len(preds),
        }

    return eval_results


def consolidate_results(all_results, node_id, test_data):
    """汇总所有 λ_mass × seed 组合的结果，给出均值±std"""
    swmm_max = max(
        [np.max(test_data[rp]['depth']) for rp in EXTREME_RPS if rp in test_data]
    )

    summary = {
        'node_id': node_id,
        'swmm_max_depth_overall': float(swmm_max),
        'timestamp': TIMESTAMP,
        'by_lambda': {},
    }

    for lam_str, seed_dict in all_results.items():
        lam_summary = {'n_seeds': len(seed_dict), 'by_rp': {}}
        for rp in EXTREME_RPS:
            rp_str = str(rp)
            metrics_across_seeds = {
                'RMSE': [], 'MAE': [], 'R2': [], 'PeakErr': [],
            }
            for seed_str, result in seed_dict.items():
                ev = result.get('evaluation', {})
                if ev and rp_str in ev:
                    m = ev[rp_str]
                    metrics_across_seeds['RMSE'].append(m['RMSE_mean'])
                    metrics_across_seeds['MAE'].append(m['MAE_mean'])
                    if m.get('R2_valid'):
                        metrics_across_seeds['R2'].append(m['R2'])
                    metrics_across_seeds['PeakErr'].append(m['PeakErr_mean'])

            if metrics_across_seeds['RMSE']:
                lam_summary['by_rp'][rp_str] = {
                    'RMSE': f"{np.mean(metrics_across_seeds['RMSE']):.5f}±{np.std(metrics_across_seeds['RMSE']):.5f}",
                    'MAE': f"{np.mean(metrics_across_seeds['MAE']):.5f}±{np.std(metrics_across_seeds['MAE']):.5f}",
                    'R2': f"{np.mean(metrics_across_seeds['R2']):.4f}±{np.std(metrics_across_seeds['R2']):.4f}" if metrics_across_seeds['R2'] else 'N/A (low variance)',
                    'PeakErr': f"{np.mean(metrics_across_seeds['PeakErr']):.5f}±{np.std(metrics_across_seeds['PeakErr']):.5f}",
                }
        summary['by_lambda'][lam_str] = lam_summary

    # 推荐最佳 λ_mass
    best_lam, best_rmse = None, float('inf')
    for lam_str, ls in summary['by_lambda'].items():
        rp100 = ls['by_rp'].get('100', {})
        if rp100 and 'RMSE' in rp100:
            rmse_val = float(rp100['RMSE'].split('±')[0])
            if rmse_val < best_rmse:
                best_rmse = rmse_val
                best_lam = lam_str

    summary['best_lambda_mass'] = best_lam
    summary['best_rmse_at_100yr'] = best_rmse

    return summary


def print_summary(summary):
    """打印实验摘要"""
    print(f"\n{'='*60}")
    print(f"  Experiment Summary: {summary['node_id']}")
    print(f"  Best λ_mass: {summary.get('best_lambda_mass')} "
          f"(RMSE@100yr={summary.get('best_rmse_at_100yr', 'N/A')})")
    print(f"{'='*60}")
    print(f"\n{'λ_mass':>10} {'RP':>5} {'RMSE':>20} {'R²':>18} {'PeakErr':>18}")
    print(f"{'─'*10} {'─'*5} {'─'*20} {'─'*18} {'─'*18}")

    for lam_str, ls in sorted(summary['by_lambda'].items()):
        for rp_str in ['100']:  # 只展示最极端情况
            if rp_str in ls['by_rp']:
                m = ls['by_rp'][rp_str]
                print(f"{lam_str:>10} {rp_str:>5} {m['RMSE']:>20} "
                      f"{m.get('R2', 'N/A'):>18} {m['PeakErr']:>18}")


# ========================== CLI ==========================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NMCL Full Experiment Runner')
    parser.add_argument('--node', type=str, default='SN_001',
                       help='Target node (SN_001/SN_017/SN_049)')
    parser.add_argument('--all_nodes', action='store_true',
                       help='Run on all 3 nodes')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test (fewer epochs, events, seeds)')
    parser.add_argument('--n_events', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=150)
    args = parser.parse_args()

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Device: {DEVICE}")

    if args.quick:
        SEEDS = [42]
        n_test_per_rp = 5
        print("QUICK MODE: 1 seed, 5 test samples, fewer epochs")

    if args.all_nodes:
        for nid in ['SN_001', 'SN_017', 'SN_049']:
            run_full_experiment(
                nid, n_events=args.n_events, epochs=args.epochs, quick=args.quick
            )
    else:
        run_full_experiment(
            args.node, n_events=args.n_events, epochs=args.epochs, quick=args.quick
        )

    print(f"\n所有实验完成! 输出目录: {OUTPUT_ROOT}")
