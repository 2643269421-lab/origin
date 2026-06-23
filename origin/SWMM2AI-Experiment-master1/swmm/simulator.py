import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import tempfile
from typing import List, Dict, Tuple, Optional
from swmm_api import SwmmInput, SwmmOutput, SwmmReport, read_inp_file, read_out_file
from swmm_api.input_file.sections import RainGage, Timeseries, TimeseriesData
# from .rainfall.generator import RainfallGenerator
# 修复：添加项目根目录到Python路径，兼容所有运行方式
from .rainfall.generator import RainfallGenerator
class SWMMSimulator:
    """
    SWMM模拟器类，用于调用SWMM进行管网模拟
    """
    
    def __init__(self, template_inp_path: str = None, output_dir: str = None,
                 output_element: str = 'J1', output_type: str = 'node', output_variable: str = 'depth'):
        """
        初始化SWMM模拟器
        
        参数:
            template_inp_path: SWMM模板文件路径
            output_element: 输出要素ID
            output_type: 输出要素类型, 'node','link','subcatchment'
            output_variable: 输出变量类型，可选值：'depth'(水位), 'flow'(流量)
        """
        self.template_inp_path = template_inp_path
        self.output_dir = output_dir
        self.output_element = output_element
        self.output_type = output_type
        self.output_variable = output_variable

        if output_dir is not None and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
    
    def run_swmm_simulation(self, rainfall_mm_h: np.ndarray, 
                           start_datetime: datetime = None,
                           time_step_min: int = 5) -> Dict:
        """
        运行SWMM模拟
        
        参数:
            rainfall_mm_h: 降雨强度序列 (mm/h)
            start_datetime: 模拟开始时间
            time_step_min: 时间步长 (分钟)
            
        返回:
            模拟结果字典
        """
        # 设置开始时间
        if start_datetime is None:
            start_datetime = datetime(2026, 1, 1, 0, 0, 0)
        
        # 计算模拟时长
        duration_hours = len(rainfall_mm_h) * time_step_min / 60
        end_datetime = start_datetime + timedelta(hours=duration_hours)
                
        try:
            # 创建临时目录
            if self.output_dir is None:
                output_dir = tempfile.mkdtemp()
            else:
                output_dir = self.output_dir
            # 生成SWMM输入文件
            inp_path = os.path.join(output_dir, 'swmm_model.inp')
            self._create_swmm_input_file(inp_path, rainfall_mm_h, 
                                        start_datetime, end_datetime, time_step_min)
            
            # 运行SWMM模拟
            result = self._run_swmm_executable(inp_path, output_dir)
            
            # 清理临时文件
            if self.output_dir is None:
                self._cleanup_temp_files(output_dir)
            
            return result
            
        except Exception as e:
            print(f"SWMM模拟失败: {e}")
    
    def _create_swmm_input_file(self, inp_path: str, rainfall_mm_h: np.ndarray,
                               start_datetime: datetime, end_datetime: datetime,
                               time_step_min: int):
        """创建SWMM输入文件"""
        
        # 读取模板文件或使用默认模板
        with open(self.template_inp_path, 'r', encoding='utf-8') as f:
            inp_content = f.read()
        
        # 生成降雨时间序列数据
        timeseries_data = self._generate_timeseries_data(rainfall_mm_h, 
                                                         start_datetime, 
                                                         time_step_min)
        
        # 替换时间序列部分
        ts_start = inp_content.find('[TIMESERIES]')
        if ts_start != -1:
            ts_end = inp_content.find('\n[', ts_start + 1)
            if ts_end == -1:
                ts_end = len(inp_content)
            
            new_ts_section = f"[TIMESERIES]\n{timeseries_data}\n"
            inp_content = inp_content[:ts_start] + new_ts_section + inp_content[ts_end:]
        
        # 更新模拟时间
        # inp_content = inp_content.replace('2026-01-01', start_datetime.strftime('%Y-%m-%d'))
        # inp_content = inp_content.replace('2026-01-02', end_datetime.strftime('%Y-%m-%d'))
        # inp_content = inp_content.replace('00:00:00', start_datetime.strftime('%H:%M:%S'))
        
        # 写入文件
        with open(inp_path, 'w', encoding='utf-8') as f:
            f.write(inp_content)
    
    def _generate_timeseries_data(self, rainfall_mm_h: np.ndarray,
                                 start_datetime: datetime,
                                 time_step_min: int) -> str:
        """生成降雨时间序列数据"""
        ts_lines = []
        
        # 添加表头注释
        ts_lines.append(";;名称       日期       时间       值")
        
        # 生成时间序列
        current_time = start_datetime
        
        for i, intensity in enumerate(rainfall_mm_h):
            # SWMM降雨格式：时间 强度(mm/h)
            time_str = current_time.strftime('%H:%M:%S')
            date_str = current_time.strftime('%m/%d/%Y')
            
            # 只添加有降雨的时间点（优化文件大小）
            if intensity > 0.01:  # 阈值设为0.01mm/h
                ts_lines.append(f"TS1        {date_str}  {time_str}  {intensity:.2f}")
            
            # 增加时间
            current_time += timedelta(minutes=time_step_min)
        
        return '\n'.join(ts_lines)
    
    def _run_swmm_executable(self, inp_path: str, output_dir: str) -> Dict:
        """
        运行SWMM可执行文件
        
        注意：需要预先安装SWMM 5.1+，并设置环境变量
        """
        import subprocess
        import sys
        
        # 设置输出文件路径
        rpt_path = os.path.join(output_dir, 'swmm_report.rpt')
        out_path = os.path.join(output_dir, 'swmm_output.out')
        
        # 构建SWMM命令
        # 在Windows上，SWMM可执行文件通常是swmm5.exe
        # 在Linux/macOS上，可能是swmm5或runswmm
        
        swmm_executable = None
        
        # 尝试查找SWMM可执行文件
        # 获取项目根目录（相对当前文件的路径）
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        possible_paths = [
            os.path.join(project_root, 'runswmm.exe'),  # 项目根目录
            'swmm5.exe',  # Windows
            'swmm5',      # Linux/macOS
            'runswmm.exe',    # 其他
            'C:\\Program Files\\EPA SWMM 5.1\\swmm5.exe',  # Windows默认路径
            '/usr/local/bin/swmm5',  # Linux常见路径
        ]
        
        for path in possible_paths:
            try:
                subprocess.run([path, '--version'], capture_output=True, check=False)
                swmm_executable = path
                break
            except:
                continue
        
        if swmm_executable is None:
            raise Exception("未找到SWMM可执行文件，请确保SWMM已安装并添加到系统路径")
        
        # 运行SWMM
        cmd = [swmm_executable, inp_path, rpt_path, out_path]
        
        print(f"运行SWMM命令: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                print(f"SWMM运行错误: {result.stderr}")
                raise Exception(f"SWMM模拟失败，返回码: {result.returncode}")
            
            # 读取输出文件
            return self._read_swmm_output(out_path)
            
        except subprocess.TimeoutExpired:
            raise Exception("SWMM模拟超时")
        except Exception as e:
            raise Exception(f"SWMM模拟错误: {e}")
    
    def _read_swmm_output(self, out_path: str) -> Dict:
        """读取SWMM输出文件"""
        
        # 注意：SWMM输出文件是二进制格式，需要使用专门的库读取
        # 这里我们使用swmm-api库来读取
        
        try:
            # 使用swmm-api读取输出文件
            out = read_out_file(out_path)
            
            # 获取节点结果
            results = {}
            data = out.get_part(self.output_type, self.output_element, self.output_variable)
            
            # 转换为numpy数组
            timestamps = data.index
            values = data.values.flatten()
            
            # 转换为字典格式
            results = {
                'timestamps': timestamps,
                'values': values,
                'type': self.output_type,
                'id': self.output_element,
                'variable': self.output_variable
            }
            
            return results
            
        except Exception as e:
            print(f"读取SWMM输出文件失败: {e}")
    
    
    def _cleanup_temp_files(self, temp_dir: str):
        """清理临时文件"""
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
    
    def plot_simulation_results(self, rainfall_mm_h: np.ndarray, 
                               swmm_results: Dict, time_step_min: int = 5):
        """绘制模拟结果"""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 绘制降雨
        time_hours = np.arange(len(rainfall_mm_h)) * time_step_min / 60
        ax1.bar(time_hours, rainfall_mm_h, width=time_step_min/60/1.5, 
                alpha=0.7, color='blue', edgecolor='darkblue')
        ax1.set_xlabel('时间 (小时)')
        ax1.set_ylabel('降雨强度 (mm/h)', color='blue')
        ax1.set_title('降雨过程线')
        ax1.grid(True, alpha=0.3)
        
        # 绘制水位
        ax2_twin = ax1.twinx()
        water_level = swmm_results['values']
        ax2_twin.plot(time_hours, water_level, 'r-', linewidth=2, label='水位')
        ax2_twin.set_ylabel('水位 (m)', color='red')
        ax2_twin.tick_params(axis='y', labelcolor='red')
        ax2_twin.legend(loc='upper right')
        
        # 绘制详细水位图
        ax2.plot(time_hours, water_level, 'g-', linewidth=2)
        ax2.fill_between(time_hours, 0, water_level, alpha=0.3, color='green')
        ax2.set_xlabel('时间 (小时)')
        ax2.set_ylabel('水位 (m)')
        ax2.set_title(f'{swmm_results["type"]} {swmm_results["id"]} {swmm_results["variable"]}响应过程线')
        ax2.grid(True, alpha=0.3)
        
        # 添加统计信息
        stats_text = (f"最大水位: {water_level.max():.2f} m\n"
                     f"平均水位: {water_level.mean():.2f} m\n"
                     f"响应延迟: {self._calculate_response_delay(rainfall_mm_h, water_level, time_step_min):.1f} 小时")
        
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if swmm_results.get('simulated', False):
            fig.suptitle('简化模拟结果', fontsize=14, fontweight='bold')
        else:
            fig.suptitle('SWMM模拟结果', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def _calculate_response_delay(self, rainfall: np.ndarray, 
                                 water_level: np.ndarray, 
                                 time_step_min: int) -> float:
        """计算水位响应延迟"""
        
        # 找到降雨峰值位置
        rainfall_peak_idx = np.argmax(rainfall)
        
        # 找到水位峰值位置
        water_level_peak_idx = np.argmax(water_level)
        
        # 计算时间差（小时）
        delay_hours = (water_level_peak_idx - rainfall_peak_idx) * time_step_min / 60
        
        return max(0, delay_hours)








