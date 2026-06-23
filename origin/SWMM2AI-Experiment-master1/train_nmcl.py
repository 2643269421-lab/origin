#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NMCL (Node Mass Conservation Loss) 训练脚本
===========================================
在 PCCA-LSTM 基础上引入节点质量守恒约束进行训练。

用法:
    # 使用默认参数训练 (SN_001, λ_mass=0.01)
    python train_nmcl.py

    # 指定 λ_mass 和其他超参数
    python train_nmcl.py --node SN_001 --lambda_mass 0.01 --epochs 200

    # 网格搜索最佳 λ_mass
    python train_nmcl.py --grid_search

输出:
    模型保存至 output/nmcl_experiment_{timestamp}/
    包含 .pth 权重文件和训练日志。
"""

import sys, os, io, json, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import torch
from datetime import datetime

from model import Trainer
from node_physics import get_or_fit_node_physics


def train_single_node(node_id, lambda_mass, epochs=200, n_events=200,
                      lambda_smooth=0.01, lambda_peak=0.05,
                      output_dir=None):
    """为单个节点训练带 NMCL 的 PCCA-LSTM 模型"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = f'output/nmcl_experiment_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, f'pcca_lstm_nmcl_{node_id}_lmass{lambda_mass}.pth')

    print(f"{'='*60}")
    print(f"  NMCL 训练: node={node_id}, λ_mass={lambda_mass}")
    print(f"{'='*60}")

    # 获取节点物理参数 (含 rating curve 拟合)
    print(f"\n[1/3] 提取节点物理参数 + 拟合 rating curve...")
    physics = get_or_fit_node_physics(
        'template.inp', node_id,
        cache_dir=output_dir,
    )
    print(f"  H₀={physics.H0:.3f}m, A_surf={physics.A_surface:.3f}m²")
    print(f"  catchment={physics.catchment_area_ha:.4f}ha, C={physics.runoff_coeff:.3f}")
    print(f"  rating curve: α={physics.alpha:.4f}, β={physics.beta:.4f}")

    # 初始化训练器
    print(f"\n[2/3] 初始化 PCCA-LSTM 训练器...")
    trainer = Trainer(
        model_type='PCCA-LSTM',
        model_params={
            'input_size': 1,
            'hidden_size': 128,
            'num_layers': 2,
            'output_size': 1,
            'dropout': 0.3,
        },
        model_path=model_path,
    )

    # 训练
    print(f"\n[3/3] 开始训练 (loss=physically_consistent + NMCL)...")
    print(f"  λ_smooth={lambda_smooth}, λ_peak={lambda_peak}, λ_mass={lambda_mass}")
    print(f"  n_events={n_events}, epochs={epochs}")

    model, dataset = trainer.train(
        n_events=n_events,
        seq_length=288,
        time_step_min=5,
        epochs=epochs,
        lr=0.001,
        max_return_period=10,
        loss_type='physically_consistent',
        lambda_smooth=lambda_smooth,
        lambda_peak=lambda_peak,
        lambda_mass=lambda_mass,
        node_physics_params=physics,
    )

    # 保存训练日志
    log = {
        'node_id': node_id,
        'lambda_mass': lambda_mass,
        'lambda_smooth': lambda_smooth,
        'lambda_peak': lambda_peak,
        'epochs': epochs,
        'n_events': n_events,
        'physics_params': {
            'H0': physics.H0,
            'A_surface': physics.A_surface,
            'catchment_ha': physics.catchment_area_ha,
            'runoff_coeff': physics.runoff_coeff,
            'alpha': physics.alpha,
            'beta': physics.beta,
        },
        'model_path': model_path,
    }
    with open(os.path.join(output_dir, 'training_log.json'), 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print(f"\n训练完成! 模型保存至: {model_path}")
    return model, dataset, physics


def grid_search_lambda_mass(node_id='SN_001', lambdas=None, epochs=150):
    """网格搜索最佳 λ_mass 值"""
    if lambdas is None:
        lambdas = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'output/nmcl_gridsearch_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"  NMCL λ_mass 网格搜索: node={node_id}")
    print(f"  候选值: {lambdas}")
    print(f"{'='*60}")

    # 获取物理参数（只拟合一次 rating curve）
    print(f"\n[1] 提取物理参数...")
    physics = get_or_fit_node_physics(
        'template.inp', node_id,
        cache_dir=output_dir,
    )

    results = {}
    for lam in lambdas:
        print(f"\n{'─'*50}")
        print(f"  训练 λ_mass = {lam}")
        print(f"{'─'*50}")

        model_path = os.path.join(output_dir, f'pcca_lstm_lmass{lam}.pth')
        trainer = Trainer(
            model_type='PCCA-LSTM',
            model_params={
                'input_size': 1, 'hidden_size': 128,
                'num_layers': 2, 'output_size': 1, 'dropout': 0.3,
            },
            model_path=model_path,
        )

        model, dataset = trainer.train(
            n_events=150,
            seq_length=288,
            time_step_min=5,
            epochs=epochs,
            lr=0.001,
            max_return_period=10,
            loss_type='physically_consistent',
            lambda_smooth=0.01,
            lambda_peak=0.05,
            lambda_mass=lam,
            node_physics_params=physics,
        )
        results[str(lam)] = 'trained'

    # 汇总
    summary_path = os.path.join(output_dir, 'grid_search_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'node_id': node_id,
            'lambdas_searched': lambdas,
            'results': results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n网格搜索完成! 结果: {output_dir}")
    return results


# ========================== CLI ==========================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PCCA-LSTM + NMCL 训练')
    parser.add_argument('--node', type=str, default='SN_001',
                       help='目标节点ID (SN_001/SN_017/SN_049)')
    parser.add_argument('--lambda_mass', type=float, default=0.01,
                       help='MassConservationLoss 权重')
    parser.add_argument('--lambda_smooth', type=float, default=0.01)
    parser.add_argument('--lambda_peak', type=float, default=0.05)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--n_events', type=int, default=200)
    parser.add_argument('--grid_search', action='store_true',
                       help='运行 λ_mass 网格搜索')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    if args.grid_search:
        grid_search_lambda_mass(node_id=args.node, epochs=args.epochs)
    else:
        train_single_node(
            node_id=args.node,
            lambda_mass=args.lambda_mass,
            epochs=args.epochs,
            n_events=args.n_events,
            lambda_smooth=args.lambda_smooth,
            lambda_peak=args.lambda_peak,
            output_dir=args.output_dir,
        )
