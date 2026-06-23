#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
# Force UTF-8 output for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

"""
物理一致性检验实验 (Physical Consistency Analysis)
====================================================
三个实验：
  实验1 — 水位变化率 (ΔH/Δt) 逐时步对比
  实验2 — 峰值索引误差 (Peak Index Error, min)
  实验3 — 质量守恒近似验证 (Runoff Coefficient)

比较对象：SWMM (Ground Truth)、LSTM、Causal Attention LSTM

使用 extreme_experiment 中已训练的模型（≤10yr 重现期训练）
在 20/30/50/100yr 极端测试集上进行全面评估。
"""

import sys, os, json, time, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from model import Predictor
from swmm.simulator import SWMMSimulator
from swmm.rainfall.generator import RainfallGenerator

# ===================== 全局配置 =====================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_ROOT = os.path.join('output', f'physical_consistency_{TIMESTAMP}')
os.makedirs(OUTPUT_ROOT, exist_ok=True)

TIME_STEP_MIN = 5
SEQ_LENGTH = 288
EXTREME_RETURN_PERIODS = [20, 30, 50, 100]
N_TEST_PER_RP = 15  # per return period for efficiency

# 所用模型路径（来自 extreme_experiment）
MODEL_PATHS = {
    'LSTM': 'output/extreme_experiment_20260610_110041/extreme_simplelstm_model.pth',
    'Causal Attention LSTM': 'output/extreme_experiment_20260610_110041/extreme_causalattentionlstm_model.pth',
}
MODEL_COLORS = {'LSTM': '#f27970', 'Causal Attention LSTM': '#54b345', 'SWMM': '#05b9e2'}
MODEL_LINESTYLE = {'LSTM': '--', 'Causal Attention LSTM': '-.', 'SWMM': '-'}

print(f"输出目录: {OUTPUT_ROOT}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"计算设备: {'GPU' if device == 'cuda' else 'CPU'}")


# ======================== 加载模型 ========================
def load_predictors():
    predictors = {}
    for name, path in MODEL_PATHS.items():
        print(f"加载 {name}: {path}")
        pred = Predictor(model_path=path, output_dir=OUTPUT_ROOT, device=device)
        predictors[name] = pred
    return predictors


# ======================== 生成测试数据 ========================
def generate_test_data():
    """为每个极端重现期生成测试数据（含SWMM真值）"""
    print(f"\n生成测试数据...")
    test_data = {}

    for rp in EXTREME_RETURN_PERIODS:
        print(f"  生成 {rp}年一遇 × {N_TEST_PER_RP} 个事件...")
        t0 = time.time()

        rg = RainfallGenerator(time_step_min=TIME_STEP_MIN)
        sim = SWMMSimulator(template_inp_path='template.inp',
                           output_dir=OUTPUT_ROOT,
                           output_element='SN_001',
                           output_type='node',
                           output_variable='depth')

        rainfall_events = []
        water_targets = []

        for i in range(N_TEST_PER_RP):
            # 生成固定重现期的降雨
            rain = rg.generate_rainfall_event(
                seq_length=SEQ_LENGTH,
                rain_type='chicago',
                duration_hours=np.random.uniform(1, 6),
                return_period=rp,
                peak_position=np.random.uniform(0.3, 0.7),
                start_idx=np.random.randint(0, 36)
            )
            # SWMM模拟
            result = sim.run_swmm_simulation(rainfall_mm_h=rain)
            if result is not None and len(result['values']) == SEQ_LENGTH:
                rainfall_events.append(rain)
                water_targets.append(result['values'])

        rainfall_events = np.array(rainfall_events)
        water_targets = np.array(water_targets)

        elapsed = time.time() - t0
        print(f"    完成: {len(rainfall_events)}/{N_TEST_PER_RP} 个有效样本, 耗时 {elapsed:.1f}s")

        test_data[rp] = {
            'rainfall': rainfall_events,
            'water_swmm': water_targets,
        }

    return test_data


# ======================== 批量预测 ========================
def batch_predict(predictors, test_data):
    """对所有测试数据进行批量预测"""
    predictions = {name: {} for name in predictors}

    for rp in EXTREME_RETURN_PERIODS:
        rain = test_data[rp]['rainfall']
        for name, pred in predictors.items():
            preds = pred.predict_batch(rain)
            predictions[name][rp] = preds

    return predictions


# ====================================================================
#  实验1：水位变化率 ΔH/Δt 分析
# ====================================================================
def experiment1_dhdt_analysis(test_data, predictions):
    """
    计算每个时间步的水位变化率 ΔH/Δt (m/5min)
    比较 SWMM vs LSTM vs CA-LSTM 在上升段和下降段的一致性
    """
    print("\n" + "=" * 60)
    print("  实验1：水位变化率 (ΔH/Δt) 分析")
    print("=" * 60)

    all_results = {}

    for rp in EXTREME_RETURN_PERIODS:
        swmm_water = test_data[rp]['water_swmm']  # (N, 288)
        n_events = swmm_water.shape[0]

        # 计算 SWMM 的 ΔH/Δt
        dh_swmm = np.diff(swmm_water, axis=1)  # (N, 287)

        model_dhs = {}
        for model_name in predictions:
            pred_water = predictions[model_name][rp]
            model_dhs[model_name] = np.diff(pred_water, axis=1)

        # 区分上升段 (dh > 0) 和下降段 (dh < 0)
        rising_mask_swmm = dh_swmm > 0
        falling_mask_swmm = dh_swmm < 0

        metrics_rp = {}

        for model_name, dh_model in model_dhs.items():
            # ---- 上升段一致性 ----
            rising_mask = rising_mask_swmm  # 以SWMM为参考
            dh_swmm_rising = dh_swmm[rising_mask]
            dh_model_rising = dh_model[rising_mask]

            if len(dh_swmm_rising) > 0:
                rmse_rising = np.sqrt(np.mean((dh_model_rising - dh_swmm_rising) ** 2))
                r2_rising_num = np.corrcoef(dh_swmm_rising, dh_model_rising)[0, 1] ** 2
            else:
                rmse_rising = 0
                r2_rising_num = 0

            # ---- 下降段一致性 ----
            falling_mask = falling_mask_swmm
            dh_swmm_falling = dh_swmm[falling_mask]
            dh_model_falling = dh_model[falling_mask]

            if len(dh_swmm_falling) > 0:
                rmse_falling = np.sqrt(np.mean((dh_model_falling - dh_swmm_falling) ** 2))
                r2_falling_num = np.corrcoef(dh_swmm_falling, dh_model_falling)[0, 1] ** 2
            else:
                rmse_falling = 0
                r2_falling_num = 0

            # ---- 整体 ----
            rmse_all = np.sqrt(np.mean((dh_model - dh_swmm) ** 2))
            r2_all = np.corrcoef(dh_swmm.flatten(), dh_model.flatten())[0, 1] ** 2

            # ---- 符号一致率 (是否与SWMM同向变化) ----
            sign_agree = np.mean(np.sign(dh_model) == np.sign(dh_swmm)) * 100

            metrics_rp[model_name] = {
                'RMSE_rising': float(rmse_rising),
                'R2_rising': float(r2_rising_num),
                'RMSE_falling': float(rmse_falling),
                'R2_falling': float(r2_falling_num),
                'RMSE_all': float(rmse_all),
                'R2_all': float(r2_all),
                'SignAgreement_pct': float(sign_agree),
            }

            print(f"  T={rp}yr | {model_name}: 上升RMSE={rmse_rising:.4f} "
                  f"下降RMSE={rmse_falling:.4f} 方向一致率={sign_agree:.1f}%")

        all_results[rp] = metrics_rp

    # ----- 可视化 -----
    _plot_dhdt_scatter(test_data, predictions, all_results)
    _plot_dhdt_timeseries(test_data, predictions)

    return all_results


def _plot_dhdt_scatter(test_data, predictions, results):
    """ΔH/Δt 散点对比图：SWMM vs 模型"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for idx, rp in enumerate(EXTREME_RETURN_PERIODS):
        ax = axes[idx]
        swmm_dh = np.diff(test_data[rp]['water_swmm'], axis=1).flatten()

        for model_name in ['LSTM', 'Causal Attention LSTM']:
            model_dh = np.diff(predictions[model_name][rp], axis=1).flatten()
            # 下采样以减小散点密度
            sample_idx = np.random.choice(len(swmm_dh), min(2000, len(swmm_dh)), replace=False)
            ax.scatter(swmm_dh[sample_idx], model_dh[sample_idx], alpha=0.4, s=8,
                      color=MODEL_COLORS[model_name], label=model_name)

        # 1:1参考线
        lim_max = max(abs(swmm_dh).max(), abs(swmm_dh).max()) * 1.1
        ax.plot([-lim_max, lim_max], [-lim_max, lim_max], 'k-', linewidth=0.8, alpha=0.5, label='1:1线')
        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.axvline(x=0, color='gray', linewidth=0.5)
        ax.set_xlim(-lim_max, lim_max)
        ax.set_ylim(-lim_max, lim_max)
        ax.set_xlabel('SWMM ΔH/Δt (m/5min)', fontsize=11)
        ax.set_ylabel('模型 ΔH/Δt (m/5min)', fontsize=11)
        ax.set_title(f'T={rp}年 — 水位变化率一致性', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('实验1：水位变化率 (ΔH/Δt) SWMM vs 模型对比', fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_ROOT, 'exp1_dhdt_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()


def _plot_dhdt_timeseries(test_data, predictions):
    """ΔH/Δt 时序对比：选取最极端事件"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    for idx, rp in enumerate(EXTREME_RETURN_PERIODS):
        ax = axes[idx]
        # 选取水位峰值最大的事件
        swmm_w = test_data[rp]['water_swmm']
        best_i = np.argmax(swmm_w.max(axis=1))
        swmm_series = swmm_w[best_i]

        time_hours = np.arange(SEQ_LENGTH) * TIME_STEP_MIN / 60
        time_mid = (time_hours[:-1] + time_hours[1:]) / 2

        # SWMM ΔH
        dh_swmm = np.diff(swmm_series)
        ax.plot(time_mid, dh_swmm, color=MODEL_COLORS['SWMM'], linewidth=2,
               label=f'SWMM ΔH/Δt', alpha=0.9)

        for model_name in ['LSTM', 'Causal Attention LSTM']:
            pred_series = predictions[model_name][rp][best_i]
            dh_model = np.diff(pred_series)
            ax.plot(time_mid, dh_model, color=MODEL_COLORS[model_name],
                   linewidth=1.5, linestyle=MODEL_LINESTYLE[model_name],
                   label=f'{model_name} ΔH/Δt', alpha=0.85)

        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle=':')
        ax.set_xlabel('时间 (h)')
        ax.set_ylabel('ΔH/Δt (m/5min)')
        ax.set_title(f'T={rp}年 — 最极端事件ΔH/Δt时序对比', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('实验1：水位变化率时序对比 (选取各重现期最极端事件)', fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_ROOT, 'exp1_dhdt_timeseries.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ====================================================================
#  实验2：峰值索引误差 (Peak Index Error)
# ====================================================================
def experiment2_peak_time_error(test_data, predictions):
    """
    计算预测水位峰值时刻与SWMM真实峰值时刻的差异
    Positive = 预测滞后, Negative = 预测超前
    """
    print("\n" + "=" * 60)
    print("  实验2：峰值索引误差 (Peak Index Error)")
    print("=" * 60)

    all_results = {}
    all_detail = []

    for rp in EXTREME_RETURN_PERIODS:
        swmm_water = test_data[rp]['water_swmm']  # (N, 288)
        n_events = swmm_water.shape[0]

        # SWMM 峰值时刻 (索引)
        swmm_peak_idx = np.argmax(swmm_water, axis=1)  # (N,)

        rp_results = {}
        for model_name in predictions:
            pred_water = predictions[model_name][rp]
            pred_peak_idx = np.argmax(pred_water, axis=1)

            # 时间差 (min), 正数=预测滞后
            time_errors_min = (pred_peak_idx - swmm_peak_idx) * TIME_STEP_MIN

            mean_error = float(np.mean(time_errors_min))
            std_error = float(np.std(time_errors_min))
            mae_error = float(np.mean(np.abs(time_errors_min)))
            max_error = float(np.max(np.abs(time_errors_min)))

            # 滞后/超前统计
            pct_lag = float(np.mean(time_errors_min > 0) * 100)    # 预测滞后占比
            pct_lead = float(np.mean(time_errors_min < 0) * 100)   # 预测超前占比
            pct_exact = float(np.mean(time_errors_min == 0) * 100) # 精确一致占比

            rp_results[model_name] = {
                'MeanError_min': mean_error,
                'StdError_min': std_error,
                'MAE_min': mae_error,
                'MaxError_min': max_error,
                'Lag_pct': pct_lag,
                'Lead_pct': pct_lead,
                'Exact_pct': pct_exact,
            }

            # 存储详细数据
            for i in range(n_events):
                all_detail.append({
                    'ReturnPeriod': rp,
                    'Model': model_name,
                    'EventIdx': i,
                    'SWMM_PeakTime_min': int(swmm_peak_idx[i] * TIME_STEP_MIN),
                    'Pred_PeakTime_min': int(pred_peak_idx[i] * TIME_STEP_MIN),
                    'TimeError_min': int(time_errors_min[i]),
                })

            print(f"  T={rp}yr | {model_name}: 峰值时刻偏差 {mean_error:+.1f}±{std_error:.1f} min "
                  f"| MAE={mae_error:.1f} min | 滞后{pct_lag:.0f}% 超前{pct_lead:.0f}%")

        all_results[rp] = rp_results

    # ----- 可视化 -----
    _plot_peak_time_error(all_results)
    _plot_peak_time_distribution(all_detail)

    # 保存 CSV
    df_detail = pd.DataFrame(all_detail)
    df_detail.to_csv(os.path.join(OUTPUT_ROOT, 'exp2_peak_time_detail.csv'), index=False, encoding='utf-8-sig')

    return all_results


def _plot_peak_time_error(results):
    """峰值索引误差箱线图"""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(EXTREME_RETURN_PERIODS))
    width = 0.3

    for i, model_name in enumerate(['LSTM', 'Causal Attention LSTM']):
        means = [results[rp][model_name]['MeanError_min'] for rp in EXTREME_RETURN_PERIODS]
        stds = [results[rp][model_name]['StdError_min'] for rp in EXTREME_RETURN_PERIODS]
        bars = ax.bar(x + i * width, means, width, yerr=stds,
                     color=MODEL_COLORS[model_name], label=model_name,
                     capsize=4, alpha=0.85)
        # 数值标注
        for j, (m, s) in enumerate(zip(means, stds)):
            ax.text(x[j] + i * width, m + np.sign(m) * max(0.5, abs(m) * 0.3),
                   f'{m:+.1f}', ha='center', va='bottom' if m < 0 else 'top',
                   fontsize=8, fontweight='bold')

    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([f'{rp}年' for rp in EXTREME_RETURN_PERIODS])
    ax.set_xlabel('重现期', fontsize=12)
    ax.set_ylabel('峰值索引误差 (min)', fontsize=12)
    ax.set_title('实验2：峰值索引误差 — 正值=预测滞后, 负值=预测超前', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_ROOT, 'exp2_peak_time_error.png'), dpi=150, bbox_inches='tight')
    plt.close()


def _plot_peak_time_distribution(detail_list):
    """峰值索引误差分布直方图"""
    df = pd.DataFrame(detail_list)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, rp in enumerate(EXTREME_RETURN_PERIODS):
        ax = axes[idx]
        for model_name in ['LSTM', 'Causal Attention LSTM']:
            errors = df[(df['ReturnPeriod'] == rp) & (df['Model'] == model_name)]['TimeError_min']
            ax.hist(errors, bins=np.arange(-30, 35, 5), alpha=0.5, color=MODEL_COLORS[model_name],
                   label=f"{model_name} (μ={errors.mean():+.1f}min)", edgecolor='white')

        ax.axvline(x=0, color='black', linewidth=1.5, linestyle='--')
        ax.set_xlabel('峰值索引误差 (min)')
        ax.set_ylabel('频数')
        ax.set_title(f'T={rp}年 — 峰值索引误差分布', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('实验2：峰值索引误差分布直方图', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_ROOT, 'exp2_peak_time_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ====================================================================
#  实验3：质量守恒近似验证 — 径流系数法
# ====================================================================
def experiment3_mass_conservation(test_data, predictions):
    """
    基于累计降雨量和累计水位响应，计算"伪径流系数" (Runoff Coefficient)
    验证模型预测是否违反基本水文规律：
      - 累计水位 ∝ 累计降雨（正相关）
      - 径流系数应在合理区间 (0 < φ < 1)
      - 不同重现期下的径流系数应遵循物理规律（重现期越大，径流系数越高）

    方法：
      累计降雨量 P_cum (mm)
      累计水位 ΣH (m) — 作为汇流-调蓄综合响应的代理指标
      定义 φ = Δ(ΣH) / Δ(P_cum) 逐时步径流响应比
    """
    print("\n" + "=" * 60)
    print("  实验3：质量守恒近似验证 (Runoff Coefficient)")
    print("=" * 60)

    all_results = {}

    for rp in EXTREME_RETURN_PERIODS:
        rainfall = test_data[rp]['rainfall']  # (N, 288) in mm/h
        swmm_water = test_data[rp]['water_swmm']  # (N, 288) in m
        n_events = rainfall.shape[0]

        # 累计降雨量: 将 (mm/h) 转为每时间步降雨量 (mm/5min)，再累计
        rain_per_step = rainfall * TIME_STEP_MIN / 60  # mm per time step
        rain_cum = np.cumsum(rain_per_step, axis=1)  # (N, 288) cumulative rain

        # 累计水位 (m)
        water_swmm_cum = np.cumsum(swmm_water, axis=1)

        # 逐事件径流响应比 (总累计水位 / 总累计降雨)
        # 相当于 ∫H dt / ∫P dt，量纲为 m/mm
        phi_swmm = np.divide(water_swmm_cum[:, -1],
                            rain_cum[:, -1] + 1e-10,
                            out=np.zeros_like(water_swmm_cum[:, -1]),
                            where=rain_cum[:, -1] > 1e-10)

        rp_results = {}

        for model_name in predictions:
            pred_water = predictions[model_name][rp]
            water_pred_cum = np.cumsum(pred_water, axis=1)

            phi_pred = np.divide(water_pred_cum[:, -1],
                                rain_cum[:, -1] + 1e-10,
                                out=np.zeros_like(water_pred_cum[:, -1]),
                                where=rain_cum[:, -1] > 1e-10)

            # 相关性：模型累计水位 vs SWMM累计水位
            corr_cum = float(np.corrcoef(water_pred_cum[:, -1], water_swmm_cum[:, -1])[0, 1])

            # φ偏差
            phi_error = float(np.mean(np.abs(phi_pred - phi_swmm)))
            phi_bias = float(np.mean(phi_pred - phi_swmm))

            # 时间累积一致性 (R² between cumulative curves at every timestep)
            ss_res = np.sum((water_pred_cum - water_swmm_cum) ** 2)
            ss_tot = np.sum((water_swmm_cum - water_swmm_cum.mean(axis=1, keepdims=True)) ** 2)
            r2_cum = float(1 - ss_res / (ss_tot + 1e-10))

            # 径流系数的物理合理性检验
            # 理论上 φ 应 > 0 (有降雨就有水位响应)
            # 且不同事件间 φ 应保持稳定（同样的管网，同样的响应特性）
            phi_cv = float(np.std(phi_pred) / (np.mean(phi_pred) + 1e-10))  # 变异系数

            rp_results[model_name] = {
                'phi_swmm_mean': float(np.mean(phi_swmm)),
                'phi_pred_mean': float(np.mean(phi_pred)),
                'phi_error_MAE': phi_error,
                'phi_bias': phi_bias,
                'phi_CV': phi_cv,
                'cum_correlation': corr_cum,
                'cum_R2': r2_cum,
                'phi_violation_pct': float(np.mean((phi_pred < 0) | (phi_pred > np.percentile(phi_swmm, 95) * 3)) * 100),
            }

            print(f"  T={rp}yr | {model_name}: phi_SWMM={np.mean(phi_swmm):.4f} "
                  f"phi_pred={np.mean(phi_pred):.4f} cumR2={r2_cum:.4f} "
                  f"phi_MAE={phi_error:.4f}")

        all_results[rp] = rp_results

    # ----- 可视化 -----
    _plot_runoff_coefficient(all_results)
    _plot_cumulative_comparison(test_data, predictions)

    return all_results


def _plot_runoff_coefficient(results):
    """Runoff Coefficient Comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(EXTREME_RETURN_PERIODS))
    width = 0.25

    # SWMM 基准
    swmm_phis = [results[rp]['LSTM']['phi_swmm_mean'] for rp in EXTREME_RETURN_PERIODS]
    ax.bar(x - width, swmm_phis, width, color=MODEL_COLORS['SWMM'], label='SWMM (Ground Truth)', alpha=0.9)

    for i, model_name in enumerate(['LSTM', 'Causal Attention LSTM']):
        pred_phis = [results[rp][model_name]['phi_pred_mean'] for rp in EXTREME_RETURN_PERIODS]
        ax.bar(x + (i) * width, pred_phis, width, color=MODEL_COLORS[model_name],
              label=model_name, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{rp}年' for rp in EXTREME_RETURN_PERIODS])
    ax.set_xlabel('重现期', fontsize=12)
    ax.set_ylabel('Runoff Response Ratio = Sum(H) / Sum(P) (m/mm)', fontsize=12)
    ax.set_title('实验3：径流响应系数对比 — 模型 vs SWMM基准', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_ROOT, 'exp3_runoff_coefficient.png'), dpi=150, bbox_inches='tight')
    plt.close()


def _plot_cumulative_comparison(test_data, predictions):
    """累计降雨-累计水位关系：选取最极端事件"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for idx, rp in enumerate(EXTREME_RETURN_PERIODS):
        ax = axes[idx]
        rain = test_data[rp]['rainfall']
        swmm_w = test_data[rp]['water_swmm']

        # 最极端事件
        best_i = np.argmax(swmm_w.max(axis=1))

        # 累计降雨 (mm)
        rain_cum = np.cumsum(rain[best_i] * TIME_STEP_MIN / 60)
        # 累计水位
        swmm_cum = np.cumsum(swmm_w[best_i])

        # 双轴
        ax2 = ax.twinx()

        # 累计水位-累计降雨散点 (Lagrangian图)
        # SWMM
        ax2.scatter(rain_cum, swmm_cum, s=2, color=MODEL_COLORS['SWMM'], alpha=0.6, label='SWMM')

        for model_name in ['LSTM', 'Causal Attention LSTM']:
            pred_cum = np.cumsum(predictions[model_name][rp][best_i])
            ax2.scatter(rain_cum, pred_cum, s=2, color=MODEL_COLORS[model_name],
                       alpha=0.4, label=model_name)

        ax2.set_ylabel('累计水位 ΣH (m)', fontsize=11)
        ax.set_xlabel('累计降雨量 ΣP (mm)', fontsize=11)
        ax.set_ylabel('累计降雨量 ΣP (mm)', fontsize=11)
        ax.set_title(f'T={rp}年 — 累计降雨-水位响应关系', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 合并图例
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper left')

    fig.suptitle('实验3：累计降雨量-累计水位响应关系 (Lagrangian分析)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_ROOT, 'exp3_cumulative_relation.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ====================================================================
#  综合结果汇总
# ====================================================================
def save_comprehensive_report(exp1, exp2, exp3):
    """汇总三个实验的关键结果到一份结构化报告"""
    print("\n" + "=" * 60)
    print("  生成综合报告")
    print("=" * 60)

    # JSON 报告
    report = {
        'experiment': 'Physical Consistency Analysis',
        'timestamp': TIMESTAMP,
        'device': device,
        'return_periods': EXTREME_RETURN_PERIODS,
        'experiment1_dhdt': {str(rp): exp1[rp] for rp in EXTREME_RETURN_PERIODS},
        'experiment2_peak_time': {str(rp): exp2[rp] for rp in EXTREME_RETURN_PERIODS},
        'experiment3_mass_conservation': {str(rp): exp3[rp] for rp in EXTREME_RETURN_PERIODS},
    }

    json_path = os.path.join(OUTPUT_ROOT, 'physical_consistency_report.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告已保存: {json_path}")

    # ===== 打印控制台摘要 =====
    print("\n" + "=" * 70)
    print("  物理一致性检验 — 综合结果摘要")
    print("=" * 70)

    # 实验1摘要
    print(f"\n{'─' * 50}")
    print("  实验1：ΔH/Δt 符号方向一致率 (%)")
    print(f"{'─' * 50}")
    header = f"{'模型':<22}"
    for rp in EXTREME_RETURN_PERIODS:
        header += f"T={rp}yr".rjust(12)
    print(header)
    for model_name in ['LSTM', 'Causal Attention LSTM']:
        row = f"{model_name:<22}"
        for rp in EXTREME_RETURN_PERIODS:
            val = exp1[rp][model_name]['SignAgreement_pct']
            row += f"{val:>8.1f}%  "
        print(row)

    # 实验2摘要
    print(f"\n{'─' * 50}")
    print("  实验2：峰值索引误差 MAE (min)")
    print(f"{'─' * 50}")
    print(header)
    for model_name in ['LSTM', 'Causal Attention LSTM']:
        row = f"{model_name:<22}"
        for rp in EXTREME_RETURN_PERIODS:
            val = exp2[rp][model_name]['MAE_min']
            row += f"{val:>8.1f}  "
        print(row)

    # 实验3摘要
    print(f"\n{'─' * 50}")
    print("  实验3：径流响应比 φ (m/mm)")
    print(f"{'─' * 50}")
    phi_header = f"{'模型':<22}"
    for rp in EXTREME_RETURN_PERIODS:
        phi_header += f"T={rp}yr".rjust(12)
    print(phi_header)
    # SWMM
    row = f"{'SWMM (Ground Truth)':<22}"
    for rp in EXTREME_RETURN_PERIODS:
        val = exp3[rp]['LSTM']['phi_swmm_mean']
        row += f"{val:>8.4f}  "
    print(row)
    for model_name in ['LSTM', 'Causal Attention LSTM']:
        row = f"{model_name:<22}"
        for rp in EXTREME_RETURN_PERIODS:
            val = exp3[rp][model_name]['phi_pred_mean']
            row += f"{val:>8.4f}  "
        print(row)

    return json_path


# ====================================================================
#  主函数
# ====================================================================
def main():
    print("=" * 70)
    print("  物理一致性检验实验 (Physical Consistency Analysis)")
    print("=" * 70)
    print(f"\n  时间: {TIMESTAMP}")
    print(f"  测试重现期: {EXTREME_RETURN_PERIODS}")
    print(f"  每重现期样本数: {N_TEST_PER_RP}")
    print(f"  对比模型: LSTM, Causal Attention LSTM")
    print(f"  基准: SWMM 水力模拟")

    # 1. 加载已训练模型
    predictors = load_predictors()

    # 2. 生成测试数据
    test_data = generate_test_data()

    # 3. 批量预测
    print(f"\n批量预测...")
    predictions = batch_predict(predictors, test_data)
    print("  预测完成")

    # 4. 三个实验
    exp1 = experiment1_dhdt_analysis(test_data, predictions)
    exp2 = experiment2_peak_time_error(test_data, predictions)
    exp3 = experiment3_mass_conservation(test_data, predictions)

    # 5. 综合报告
    save_comprehensive_report(exp1, exp2, exp3)

    print(f"\n{'=' * 60}")
    print(f"  物理一致性检验完成！结果: {OUTPUT_ROOT}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
