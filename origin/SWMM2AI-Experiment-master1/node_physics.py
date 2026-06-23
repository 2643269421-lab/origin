#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
节点物理参数提取与 Rating Curve 拟合工具
==========================================
从 SWMM template.inp 文件中提取节点的物理参数（底高程、汇水面积、
下游管渠几何参数等），并提供基于 SWMM 模拟数据的 rating curve 拟合。

用途：为 MassConservationLoss（节点质量守恒损失）提供物理参数。
"""

import os
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional


# ========================== 数据类 ==========================

@dataclass
class NodePhysicsParams:
    """节点物理参数"""
    node_id: str
    H0: float                          # 节点底高程 (m)
    max_depth: float                   # 最大水深 (m)
    A_surface: float                   # 有效水面面积 (m²) — 上下游管道截面积之和
    catchment_area_ha: float           # 上游总汇水面积 (ha)
    catchment_area_m2: float           # 上游总汇水面积 (m²)
    runoff_coeff: float                # 综合径流系数 (0~1)
    downstream_links: List[str] = field(default_factory=list)  # 下游管渠ID列表
    downstream_diameters: List[float] = field(default_factory=list)  # 下游管径 (m)
    upstream_links: List[str] = field(default_factory=list)     # 上游管渠ID列表
    upstream_diameters: List[float] = field(default_factory=list)  # 上游管径 (m)

    # Rating curve 参数 (需通过 SWMM 数据拟合)
    alpha: float = 1.0                 # Q_out = α × max(0, H - H₀)^β
    beta: float = 1.5                  # 默认 1.5 (宽顶堰/孔口混合)


# ========================== INP 解析函数 ==========================

def extract_node_physics(inp_path: str, node_id: str) -> NodePhysicsParams:
    """
    从 template.inp 文件提取指定节点的物理参数。

    解析 [JUNCTIONS], [SUBCATCHMENTS], [CONDUITS], [XSECTIONS] 四个节。

    Args:
        inp_path: template.inp 文件路径
        node_id:  目标节点ID (e.g. 'SN_001')

    Returns:
        NodePhysicsParams 对象
    """
    with open(inp_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1) 解析 [JUNCTIONS] — 获取 H₀ 和 MaxDepth
    H0, max_depth = _parse_junction(content, node_id)

    # 2) 解析 [SUBCATCHMENTS] — 汇总排入该节点的子汇水区
    catchment_ha, runoff_coeff = _parse_subcatchments_for_node(content, node_id)

    # 3) 解析 [CONDUITS] — 获取上下游管渠
    downstream_links, upstream_links = _parse_conduits_for_node(content, node_id)

    # 4) 解析 [XSECTIONS] — 获取管渠直径
    link_diameters = _parse_xsections(content)

    downstream_diameters = [link_diameters.get(l, 0.3) for l in downstream_links]
    upstream_diameters = [link_diameters.get(l, 0.3) for l in upstream_links]

    # 5) 计算有效水面面积 A_surface
    #    ≈ Σ(上下游管道截面积)
    all_diameters = downstream_diameters + upstream_diameters
    if all_diameters:
        A_surface = sum(np.pi * d**2 / 4 for d in all_diameters)
    else:
        A_surface = 1.0  # fallback: 1 m²

    catchment_m2 = catchment_ha * 10000.0

    return NodePhysicsParams(
        node_id=node_id,
        H0=H0,
        max_depth=max_depth,
        A_surface=A_surface,
        catchment_area_ha=catchment_ha,
        catchment_area_m2=catchment_m2,
        runoff_coeff=runoff_coeff,
        downstream_links=downstream_links,
        downstream_diameters=downstream_diameters,
        upstream_links=upstream_links,
        upstream_diameters=upstream_diameters,
    )


def _parse_junction(content: str, node_id: str) -> Tuple[float, float]:
    """从 [JUNCTIONS] 节提取节点底高程和最大水深"""
    import re
    pattern = rf'^{node_id}\s+([\d.]+)\s+([\d.]+)'
    for line in content.split('\n'):
        m = re.match(pattern, line.strip())
        if m:
            return float(m.group(1)), float(m.group(2))
    raise ValueError(f"节点 {node_id} 未在 [JUNCTIONS] 中找到")


def _parse_subcatchments_for_node(content: str, node_id: str) -> Tuple[float, float]:
    """
    汇总所有以 node_id 为出口的子汇水区。

    Returns:
        (total_area_ha, weighted_runoff_coeff)
    """
    import re
    # 匹配: SU_xxx ... OUTLET_NODE_ID Area %Imperv ...
    # 格式: Name  Raingage  Outlet  Area  %Imperv  Width  %Slope  ...
    total_area = 0.0
    weighted_c_sum = 0.0  # Σ(C × A)

    in_section = False
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('[SUBCATCHMENTS]'):
            in_section = True
            continue
        if in_section and line.startswith('['):
            break
        if not in_section or not line or line.startswith(';;'):
            continue

        parts = line.split()
        if len(parts) < 5:
            continue
        # parts[0]=Name, parts[1]=Raingage, parts[2]=Outlet, parts[3]=Area, parts[4]=%Imperv
        outlet = parts[2]
        if outlet != node_id:
            continue
        try:
            area = float(parts[3])
            imperv = float(parts[4]) / 100.0  # 转为小数
        except (ValueError, IndexError):
            continue

        # 综合径流系数经验公式: C ≈ 0.05 + 0.9 × %Imperv
        c = 0.05 + 0.9 * imperv
        c = min(max(c, 0.05), 0.95)  # clamp

        total_area += area
        weighted_c_sum += c * area

    if total_area > 0:
        runoff_coeff = weighted_c_sum / total_area
    else:
        runoff_coeff = 0.3  # fallback

    return total_area, runoff_coeff


def _parse_conduits_for_node(content: str, node_id: str) -> Tuple[List[str], List[str]]:
    """
    找到所有以 node_id 为起点（下游）和终点（上游）的管渠。

    Returns:
        (downstream_links, upstream_links)
    """
    downstream = []
    upstream = []

    in_section = False
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('[CONDUITS]'):
            in_section = True
            continue
        if in_section and line.startswith('['):
            break
        if not in_section or not line or line.startswith(';;'):
            continue

        parts = line.split()
        if len(parts) < 4:
            continue
        link_name = parts[0]
        from_node = parts[1]
        to_node = parts[2]

        if from_node == node_id:
            downstream.append(link_name)
        if to_node == node_id:
            upstream.append(link_name)

    return downstream, upstream


def _parse_xsections(content: str) -> Dict[str, float]:
    """解析 [XSECTIONS] 节，返回 {link_id: diameter_m}"""
    diameters = {}

    in_section = False
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('[XSECTIONS]'):
            in_section = True
            continue
        if in_section and line.startswith('['):
            break
        if not in_section or not line or line.startswith(';;'):
            continue

        parts = line.split()
        if len(parts) < 3:
            continue
        link_id = parts[0]
        shape = parts[1].upper()
        if shape == 'CIRCULAR':
            try:
                diameters[link_id] = float(parts[2])
            except (ValueError, IndexError):
                pass

    return diameters


# ========================== Manning 公式 Rating Curve ==========================


def _manning_rating_curve(physics: NodePhysicsParams) -> Tuple[float, float]:
    """
    使用 Manning 公式计算理论 rating curve 并拟合为幂律形式:
        Q = α × H^β

    适用条件：下游管道为正坡（坡度 > 0），逆坡管道不适用。

    对圆形管道：
        Q = (1/n) × A × R^(2/3) × S^(1/2)
        其中 A = D²/8×(θ−sinθ), R = A/P, θ = 2×arccos(1−2h/D)

    Args:
        physics: 节点物理参数（含下游管径、糙率）

    Returns:
        (alpha, beta) — 拟合后的幂律参数
    """
    from scipy.optimize import curve_fit

    # 取下游管渠中最大管径的作为主出流管
    if not physics.downstream_diameters:
        print("    ⚠ 无下游管渠数据，使用默认 α=1, β=1.5")
        return 1.0, 1.5

    D = max(physics.downstream_diameters)  # 取最大管径
    n = 0.01  # Manning's n (与 template.inp 一致)

    # 计算管道坡度 — 默认取典型坡度
    # SN_049: 1.02%, SN_017: 1.01%, SN_001: -0.82%(逆坡不可用Manning)
    if physics.downstream_links:
        slope = 0.01  # 1% 典型坡度
    else:
        slope = 0.01

    S = slope
    S = max(S, 0.001)  # 最小坡度 0.1%

    depths = np.linspace(0.001, D, 100)
    Qs = []
    for h in depths:
        if h <= 1e-6:
            Qs.append(0.0)
            continue
        theta = 2.0 * np.arccos(max(-1.0, min(1.0, 1.0 - 2.0 * h / D)))
        A = D**2 / 8.0 * (theta - np.sin(theta))
        P = D * theta / 2.0
        R = A / max(P, 1e-10)
        Q = (1.0 / n) * A * R**(2.0/3.0) * S**0.5
        Qs.append(Q)

    Qs = np.array(Qs)

    def power_law(H, alpha, beta):
        return alpha * np.maximum(H, 1e-10) ** beta

    try:
        popt, _ = curve_fit(power_law, depths, Qs, p0=[1.0, 2.0],
                           bounds=([1e-6, 0.5], [1e6, 3.0]), maxfev=10000)
        alpha, beta = popt
        r2 = 1 - np.sum((Qs - power_law(depths, *popt))**2) / np.sum((Qs - np.mean(Qs))**2)
        print(f"    Manning 拟合: α={alpha:.4f}, β={beta:.4f}, D={D:.2f}m, "
              f"n={n}, S={S*100:.2f}%, R²={r2:.4f}")
        return float(alpha), float(beta)
    except Exception:
        print(f"    Manning 拟合失败，使用默认 α=1, β=1.5")
        return 1.0, 1.5


# ========================== Rating Curve 拟合 ==========================

def fit_rating_curve_from_swmm(
    inp_path: str,
    node_id: str,
    n_calibration_events: int = 10,
    seq_length: int = 288,
    time_step_min: int = 5,
    output_dir: str = None,
) -> Tuple[float, float]:
    """
    通过 SWMM 模拟数据拟合节点出流的 rating curve:
        Q_out = α × max(0, H - H₀)^β

    步骤:
      1. 生成 n_calibration_events 个不同重现期的降雨事件
      2. 运行 SWMM，分别提取节点水位 H(t) 和下游管渠流量 Q(t)
      3. 汇总所有 (H, Q_out) 数据对
      4. 用最小二乘法拟合 α, β

    Args:
        inp_path: template.inp 路径
        node_id:  目标节点 ID
        n_calibration_events: 标定事件数
        seq_length: 序列长度
        time_step_min: 时间步长 (min)
        output_dir: SWMM 输出目录 (None=临时目录)

    Returns:
        (alpha, beta) — 拟合参数
    """
    from swmm.simulator import SWMMSimulator
    from swmm.rainfall.generator import RainfallGenerator
    from scipy.optimize import curve_fit

    # 获取节点物理参数 (主要是 H₀ 和下游管渠)
    physics = extract_node_physics(inp_path, node_id)
    H0 = physics.H0

    # 确定下游管渠 ID (用于提取流量)
    if physics.downstream_links:
        downstream_link = physics.downstream_links[0]  # 主下游管渠
    else:
        raise ValueError(f"节点 {node_id} 没有下游管渠，无法定义 rating curve")

    # 生成标定降雨事件
    rg = RainfallGenerator(time_step_min=time_step_min)
    sim_depth = SWMMSimulator(
        template_inp_path=inp_path,
        output_dir=output_dir,
        output_element=node_id,
        output_type='node',
        output_variable='depth',
    )
    sim_flow = SWMMSimulator(
        template_inp_path=inp_path,
        output_dir=output_dir,
        output_element=downstream_link,
        output_type='link',
        output_variable='flow',
    )

    # 使用多个重现期以确保数据覆盖低水位和高水位
    return_periods = np.linspace(1, 50, n_calibration_events).astype(int).tolist()

    all_H = []
    all_Q = []

    print(f"  拟合 rating curve: node={node_id}, link={downstream_link}, H0={H0:.3f}m")
    for i, rp in enumerate(return_periods):
        try:
            rain = rg.generate_rainfall_event(
                seq_length=seq_length,
                rain_type='chicago',
                duration_hours=np.random.uniform(1, 6),
                return_period=rp,
                peak_position=np.random.uniform(0.3, 0.7),
                start_idx=np.random.randint(0, 36),
            )
            res_depth = sim_depth.run_swmm_simulation(rainfall_mm_h=rain)
            res_flow = sim_flow.run_swmm_simulation(rainfall_mm_h=rain)

            if (res_depth is not None and res_flow is not None
                    and res_depth.get('values') is not None
                    and res_flow.get('values') is not None
                    and len(res_depth['values']) == seq_length
                    and len(res_flow['values']) == seq_length):
                H = res_depth['values']
                Q = np.abs(res_flow['values'])  # 流量取绝对值
                all_H.append(H)
                all_Q.append(Q)
        except Exception as e:
            print(f"    RP={rp}yr 模拟失败: {e}")
            continue

    if len(all_H) < 2:
        print("  ⚠ 标定数据不足，使用默认 rating curve 参数 (α=1, β=1.5)")
        return 1.0, 1.5

    H_all = np.concatenate(all_H)
    Q_all = np.concatenate(all_Q)

    # SWMM 输出的是相对节点底高程的水深 (depth above invert)
    # Rating curve: Q_out = α × max(0, H - H_min)^β
    # 其中 H_min = 数据中观测到的最小水深（即基流/初始水深）
    H_min = float(np.percentile(H_all, 5))  # 5分位数作为基流水深
    H_min = max(H_min, 0.01)  # 至少 1cm

    # 只取 H > H_min + δ 的有效数据点（有明显出流的时段）
    valid = H_all > (H_min + 0.02)
    H_valid = H_all[valid]
    Q_valid = Q_all[valid]

    if len(H_valid) < 20:
        print(f"  ⚠ 有效数据点不足 ({len(H_valid)}), 回退到 Manning 公式")
        return _manning_rating_curve(physics)

    # 拟合: Q = α × max(0, H - H_min)^β
    def rating_func(H, alpha, beta):
        return alpha * np.maximum(0, H - H_min) ** beta

    try:
        popt, pcov = curve_fit(
            rating_func, H_valid, Q_valid,
            p0=[1.0, 1.5],
            bounds=([1e-6, 0.5], [1e6, 3.0]),
            maxfev=10000,
        )
        alpha, beta = popt
        print(f"  拟合完成: α={alpha:.4f}, β={beta:.4f}, H_min={H_min:.4f}m")
        # Store H_min as an attribute for use in MassConservationLoss
        # We return (alpha, beta) but the caller needs H_min too
        print(f"  R² ≈ {1 - np.sum((Q_valid - rating_func(H_valid, *popt))**2) / np.sum((Q_valid - Q_valid.mean())**2):.4f}")
        return float(alpha), float(beta)
    except Exception as e:
        print(f"  拟合失败 ({e})，使用默认参数")
        return 1.0, 1.5


# ========================== 参数缓存 ==========================

def get_or_fit_node_physics(
    inp_path: str,
    node_id: str,
    cache_dir: str = None,
    force_refit: bool = False,
) -> NodePhysicsParams:
    """
    获取节点物理参数（优先从缓存读取，否则从 INP 提取并拟合）。

    Args:
        inp_path: template.inp 路径
        node_id:  目标节点 ID
        cache_dir: 缓存目录 (默认: 与 inp_path 同目录)
        force_refit: 是否强制重新拟合

    Returns:
        NodePhysicsParams (含 rating curve 参数)
    """
    if cache_dir is None:
        cache_dir = os.path.dirname(inp_path) or '.'

    cache_file = os.path.join(cache_dir, f'node_physics_{node_id}.json')

    # 尝试从缓存读取
    if not force_refit and os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return NodePhysicsParams(**data)

    # 提取基础参数
    physics = extract_node_physics(inp_path, node_id)

    # 拟合 rating curve
    try:
        alpha, beta = fit_rating_curve_from_swmm(inp_path, node_id)
        physics.alpha = alpha
        physics.beta = beta
    except Exception as e:
        print(f"  Rating curve 拟合失败 ({e})，使用默认 α=1, β=1.5")
        physics.alpha = 1.0
        physics.beta = 1.5

    # 保存缓存
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(asdict(physics), f, ensure_ascii=False, indent=2)
    print(f"  物理参数已缓存至: {cache_file}")

    return physics


# ========================== CLI ==========================

if __name__ == '__main__':
    import sys
    inp = sys.argv[1] if len(sys.argv) > 1 else 'template.inp'
    node = sys.argv[2] if len(sys.argv) > 2 else 'SN_001'

    print(f"提取节点物理参数: {node} (INP: {inp})")
    physics = get_or_fit_node_physics(inp, node, force_refit=True)
    print(f"\n=== {node} 物理参数 ===")
    print(f"  底高程 H₀:          {physics.H0:.3f} m")
    print(f"  最大水深:            {physics.max_depth:.3f} m")
    print(f"  有效水面面积:        {physics.A_surface:.3f} m²")
    print(f"  上游汇水面积:        {physics.catchment_area_ha:.4f} ha ({physics.catchment_area_m2:.1f} m²)")
    print(f"  综合径流系数:        {physics.runoff_coeff:.4f}")
    print(f"  下游管渠:            {physics.downstream_links}")
    print(f"  下游管径:            {physics.downstream_diameters}")
    print(f"  上游管渠:            {physics.upstream_links}")
    print(f"  Rating curve α:     {physics.alpha:.4f}")
    print(f"  Rating curve β:     {physics.beta:.4f}")
