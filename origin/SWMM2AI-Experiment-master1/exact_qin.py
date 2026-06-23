#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
精确 Q_in 提取器
=================
通过直接读取 SWMM 输出文件中的上游管渠流量并求和，
获得精确的节点入流 Q_in（而非降雨-径流系数估算）。

原理：
  对节点 N，Q_in(N, t) = Σ flow(link_i, t)
  其中 link_i 是以 N 为终点的所有管渠（即上游管渠）。

优势：
  - 精确：直接从 SWMM 动力学波模拟结果中提取
  - 不需要径流系数经验公式
  - 自动包含管网汇流的非线性延迟效应
  - 论文中可以声明"Q_in 由 SWMM 输出精确求和得到"

使用方式：
  >>> extractor = ExactQInExtractor('template.inp')
  >>> result = extractor.extract('SN_001', rainfall_mmh)
  >>> print(result['q_in'])  # (288,) numpy array of exact Q_in
"""

import os, sys, tempfile
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional


class ExactQInExtractor:
    """
    精确 Q_in 提取器。

    在一次 SWMM 运行中同时提取：
      - 节点水位 H(t)   (node depth)
      - 各上游管渠流量   (link flow)
      - 精确 Q_in = Σ(上游管渠流量)
      - 下游管渠流量     (link flow, 用于 rating curve)
    """

    def __init__(self, inp_path: str, node_id: str,
                 output_dir: str = None):
        """
        Args:
            inp_path: template.inp 路径
            node_id:  目标节点 ID
            output_dir: 临时文件输出目录 (None=系统临时目录)
        """
        self.inp_path = inp_path
        self.node_id = node_id
        self.output_dir = output_dir

        # 解析上游/下游管渠
        from node_physics import extract_node_physics
        physics = extract_node_physics(inp_path, node_id)
        self.upstream_links = physics.upstream_links
        self.downstream_links = physics.downstream_links
        self.H0 = physics.H0

    def extract(self, rainfall_mmh: np.ndarray,
                start_datetime: datetime = None,
                time_step_min: int = 5) -> Dict:
        """
        运行 SWMM 并提取深度 + 所有上下游管渠流量。

        Args:
            rainfall_mmh: 降雨强度序列 (mm/h), shape (seq_len,)
            start_datetime: 模拟开始时间
            time_step_min: 时间步长 (min)

        Returns:
            {
                'depth':      np.array (seq_len,) — 节点水位深度 (m)
                'q_in':       np.array (seq_len,) — 精确 Q_in = Σ(上游管渠流量)
                'q_out':      np.array (seq_len,) — 下游管渠流量 (若存在)
                'link_flows': {link_id: np.array} — 各管渠流量明细
                'upstream_links': [...],
                'downstream_links': [...],
            }
        """
        from swmm.simulator import SWMMSimulator
        from swmm_api import read_out_file

        if start_datetime is None:
            start_datetime = datetime(2026, 1, 1, 0, 0, 0)

        # 1) 准备临时目录
        if self.output_dir is None:
            output_dir = tempfile.mkdtemp()
        else:
            output_dir = self.output_dir
            os.makedirs(output_dir, exist_ok=True)

        # 2) 生成 SWMM 输入文件 (复用 SWMMSimulator 的逻辑)
        sim = SWMMSimulator(
            template_inp_path=self.inp_path,
            output_dir=output_dir,
            output_element=self.node_id,
            output_type='node',
            output_variable='depth',
        )

        inp_path = os.path.join(output_dir, 'swmm_model.inp')
        duration_hours = len(rainfall_mmh) * time_step_min / 60
        end_datetime = start_datetime + timedelta(hours=duration_hours)
        sim._create_swmm_input_file(
            inp_path, rainfall_mmh, start_datetime, end_datetime, time_step_min
        )

        # 3) 运行 SWMM
        rpt_path = os.path.join(output_dir, 'swmm_report.rpt')
        out_path = os.path.join(output_dir, 'swmm_output.out')

        import subprocess
        # Project root = directory containing this file AND runswmm.exe
        project_root = os.path.abspath(os.path.dirname(__file__))
        swmm_exe = None
        for path in [
            os.path.join(project_root, 'runswmm.exe'),
            os.path.join(project_root, 'swmm5.exe'),
        ]:
            if os.path.exists(path):
                swmm_exe = path
                break

        if swmm_exe is None:
            # fallback to finding runswmm
            for p in ['runswmm.exe', 'swmm5.exe']:
                import shutil
                if shutil.which(p):
                    swmm_exe = p
                    break

        if swmm_exe is None:
            raise RuntimeError("未找到 SWMM 可执行文件 (runswmm.exe)")

        subprocess.run(
            [swmm_exe, inp_path, rpt_path, out_path],
            capture_output=True, text=True, timeout=60,
            check=True,
        )

        # 4) 读取 SWMM 输出 — 所有需要的变量
        out = read_out_file(out_path)

        # 4a) 节点深度
        depth_data = out.get_part('node', self.node_id, 'depth')
        depth = depth_data.values.flatten()

        # 4b) 上游管渠流量 → Q_in
        link_flows = {}
        q_in = np.zeros(len(depth_data.index))
        for link_id in self.upstream_links:
            try:
                flow_data = out.get_part('link', link_id, 'flow')
                flow = np.abs(flow_data.values.flatten())  # 取绝对值
                link_flows[link_id] = flow
                q_in += flow
            except Exception as e:
                print(f"  ⚠ 上游管渠 {link_id} 流量提取失败: {e}")

        # 4c) 下游管渠流量 → Q_out (取第一根下游管渠)
        q_out = None
        if self.downstream_links:
            try:
                dlink = self.downstream_links[0]
                flow_data = out.get_part('link', dlink, 'flow')
                q_out = np.abs(flow_data.values.flatten())
                link_flows[dlink] = q_out
            except Exception as e:
                print(f"  ⚠ 下游管渠 {self.downstream_links[0]} 流量提取失败: {e}")

        # 5) 清理临时文件
        if self.output_dir is None:
            import shutil
            try:
                shutil.rmtree(output_dir)
            except Exception:
                pass

        return {
            'depth': depth,
            'q_in': q_in,
            'q_out': q_out,
            'link_flows': link_flows,
            'upstream_links': self.upstream_links,
            'downstream_links': self.downstream_links,
        }


def generate_nmcl_dataset(
    inp_path: str,
    node_id: str,
    n_events: int = 200,
    seq_length: int = 288,
    time_step_min: int = 5,
    max_return_period: int = 10,
    output_dir: str = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成 NMCL 训练数据集：同时包含降雨、水位和精确 Q_in。

    Args:
        inp_path: template.inp 路径
        node_id:  目标节点 ID
        n_events: 事件数量
        seq_length: 序列长度
        time_step_min: 时间步长
        max_return_period: 最大重现期
        output_dir: 工作目录
        verbose: 是否打印进度

    Returns:
        (rainfall_events, depth_events, q_in_events)
        三个 numpy 数组，shape 均为 (n_valid_events, seq_length)
    """
    from swmm.rainfall.generator import RainfallGenerator

    extractor = ExactQInExtractor(inp_path, node_id, output_dir=output_dir)
    rg = RainfallGenerator(time_step_min=time_step_min)

    rainfall_list = []
    depth_list = []
    q_in_list = []

    if verbose:
        print(f"生成 NMCL 数据集: node={node_id}, n={n_events}, RP≤{max_return_period}")

    for i in range(n_events):
        # 生成降雨
        rain = rg.generate_rainfall_event(
            seq_length=seq_length,
            rain_type='chicago',
            duration_hours=np.random.uniform(1, 6),
            max_return_period=max_return_period,
            peak_position=np.random.uniform(0.3, 0.7),
            start_idx=np.random.randint(0, 36),
        )

        try:
            result = extractor.extract(rainfall_mmh=rain)
        except Exception as e:
            if verbose:
                print(f"  事件 {i}: 提取失败 ({e})")
            continue

        depth = result['depth']
        q_in = result['q_in']

        if len(depth) != seq_length or len(q_in) != seq_length:
            if verbose:
                print(f"  事件 {i}: 长度不匹配 (depth={len(depth)}, q_in={len(q_in)})")
            continue

        rainfall_list.append(rain)
        depth_list.append(depth)
        q_in_list.append(q_in)

        if verbose and (i + 1) % 20 == 0:
            print(f"  已生成 {i+1}/{n_events}, 有效 {len(depth_list)}")

    if verbose:
        print(f"  完成: {len(depth_list)}/{n_events} 个有效样本")
        if depth_list:
            print(f"  深度范围: {np.min(depth_list):.4f} ~ {np.max(depth_list):.4f} m")
            print(f"  Q_in 范围: {np.min(q_in_list):.6f} ~ {np.max(q_in_list):.6f} m³/s")

    return (
        np.array(rainfall_list),
        np.array(depth_list),
        np.array(q_in_list),
    )


# ========================== CLI 测试 ==========================

if __name__ == '__main__':
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    import numpy as np
    from swmm.rainfall.generator import RainfallGenerator

    node = sys.argv[1] if len(sys.argv) > 1 else 'SN_001'
    n_test = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    print(f"测试精确 Q_in 提取: node={node}, n={n_test}")
    print(f"上游管渠: {ExactQInExtractor('template.inp', node).upstream_links}")
    print(f"下游管渠: {ExactQInExtractor('template.inp', node).downstream_links}")

    rg = RainfallGenerator(time_step_min=5)
    for i in range(n_test):
        n_steps = 100  # use 100 steps for quick test
        rain = rg.generate_rainfall_event(
            seq_length=288, rain_type='chicago',
            return_period=(i + 1) * 10,
        )
        extractor = ExactQInExtractor('template.inp', node)
        result = extractor.extract(rainfall_mmh=rain)

        q_in_exact = result['q_in'][:n_steps]
        q_out_exact = result['q_out'][:n_steps] if result['q_out'] is not None else None
        rain_short = rain[:n_steps]

        print(f"\n事件 {i+1} (RP={(i+1)*10}yr):")
        print(f"  深度范围: {result['depth'][:n_steps].min():.4f} ~ {result['depth'][:n_steps].max():.4f} m")
        print(f"  Q_in (精确): {q_in_exact.min():.6f} ~ {q_in_exact.max():.6f} m³/s")
        if q_out_exact is not None:
            print(f"  Q_out (精确): {q_out_exact.min():.6f} ~ {q_out_exact.max():.6f} m³/s")
            # Check mass balance: Q_in - Q_out - A*dH/dt should be small
            from node_physics import extract_node_physics
            p = extract_node_physics('template.inp', node)
            dH = np.diff(result['depth'][:n_steps])
            storage = p.A_surface * dH / (5 * 60)
            R = q_in_exact[:-1] - q_out_exact[:-1] - storage
            print(f"  质量守恒残差: mean={np.mean(np.abs(R)):.6f}, max={np.max(np.abs(R)):.6f} m³/s")
            print(f"  残差/平均Q_in: {np.mean(np.abs(R)) / (np.mean(q_in_exact) + 1e-10):.2%}")

        # 比较精确 Q_in 和简化估算
        q_in_approx = p.runoff_coeff * p.catchment_area_m2 * rain_short / (1000.0 * 3600.0)
        corr = np.corrcoef(q_in_exact, q_in_approx)[0, 1]
        ratio = np.sum(q_in_exact) / (np.sum(q_in_approx) + 1e-10)
        print(f"  精确Q_in vs 简化估算: 相关系数={corr:.4f}, 总流量比={ratio:.4f}")
