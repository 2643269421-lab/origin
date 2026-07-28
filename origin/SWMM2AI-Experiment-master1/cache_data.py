#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预生成实验数据并缓存，避免每次运行 robust_experiment.py 都重新跑 SWMM 模拟。
只跑一次 200 + 60 = 260 场模拟，保存为 .npz 文件。
"""
import sys, os, io, pickle, time
import numpy as np

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from dataset import SWMMDataset

CACHE_DIR = os.path.join('output', 'robust_final', 'data_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# --- 参数（与 robust_experiment.py 保持一致）---
N_TRAIN = 200
TRAIN_MAX_RP = 10
EXTREME_RPS = [20, 30, 50, 100]
N_TEST_PER_RP = 15
SEQ_LEN = 288
DT = 5

print("=" * 60)
print("  预生成实验数据缓存")
print("=" * 60)

# ---- Step 1: 训练数据 ----
train_cache = os.path.join(CACHE_DIR, 'train_data.npz')
if os.path.exists(train_cache):
    print(f"\n[Step 1] 训练数据缓存已存在，跳过")
    data = np.load(train_cache, allow_pickle=True)
    print(f"  训练集: {len(data['X'])} 个样本")
else:
    print(f"\n[Step 1] 生成训练数据集 ({N_TRAIN} 场)...")
    t0 = time.time()
    train_dataset = SWMMDataset(
        n_events=N_TRAIN,
        seq_length=SEQ_LEN,
        time_step_min=DT,
        max_return_period=TRAIN_MAX_RP,
    )
    print(f"  用时: {time.time() - t0:.1f}s")

    # 保存
    # 只保存原始数据，不要保存 sklearn scaler 对象（避免版本兼容问题）
    np.savez_compressed(train_cache,
        X=train_dataset.X,
        y=train_dataset.y,
        rainfall_events=train_dataset.rainfall_events,
        water_level_events=train_dataset.water_level_events,
    )
    print(f"  已缓存: {train_cache}")

# ---- Step 2: 极端测试数据 ----
extreme_all = {}
for rp in EXTREME_RPS:
    cache_path = os.path.join(CACHE_DIR, f'extreme_T{rp}.npz')
    if os.path.exists(cache_path):
        print(f"\n[Step 2] T={rp}年 测试数据缓存已存在，跳过")
        data = np.load(cache_path, allow_pickle=True)
        extreme_all[rp] = dict(data)
        print(f"  T={rp}年: {len(data['X'])} 个样本")
    else:
        print(f"\n[Step 2] 生成 T={rp}年 测试数据 ({N_TEST_PER_RP} 场)...")
        t0 = time.time()
        ds = SWMMDataset(
            n_events=N_TEST_PER_RP,
            seq_length=SEQ_LEN,
            time_step_min=DT,
            return_period=rp,
        )
        print(f"  用时: {time.time() - t0:.1f}s")

        # 用训练集的 scaler 标准化
        train_data = np.load(train_cache, allow_pickle=True)
        from sklearn.preprocessing import MinMaxScaler
        rain_scaler = MinMaxScaler()
        rain_scaler.min_ = train_data['rain_scaler_min_val']
        rain_scaler.scale_ = train_data['rain_scaler_scale']
        rain_scaler.data_min_ = train_data['rain_scaler_min']
        rain_scaler.data_max_ = train_data['rain_scaler_max']

        water_scaler = MinMaxScaler()
        water_scaler.min_ = train_data['water_scaler_min_val']
        water_scaler.scale_ = train_data['water_scaler_scale']
        water_scaler.data_min_ = train_data['water_scaler_min']
        water_scaler.data_max_ = train_data['water_scaler_max']

        X_scaled = rain_scaler.transform(ds.rainfall_events.reshape(-1, 1)).reshape(ds.rainfall_events.shape)
        y_scaled = water_scaler.transform(ds.water_level_events.reshape(-1, 1)).reshape(ds.water_level_events.shape)

        np.savez_compressed(cache_path,
            X=X_scaled,
            y=y_scaled,
            rainfall_events=ds.rainfall_events,
            water_level_events=ds.water_level_events,
        )
        print(f"  已缓存: {cache_path}")

print(f"\n{'=' * 60}")
print(f"  所有数据缓存完成！")
print(f"  缓存目录: {CACHE_DIR}")
print(f"{'=' * 60}")
