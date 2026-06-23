import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

class RainfallGenerator:
    """
    暴雨生成器类

    功能：生成不同雨型、不同历时的暴雨过程线
    支持的雨型：三角形雨型、芝加哥雨型
    """

    def __init__(self, time_step_min: int = 5):
        """
        初始化暴雨生成器

        参数:
            time_step_min: 时间步长（分钟），默认为5分钟
        """
        self.time_step_min = time_step_min

        # 默认暴雨参数
        self.default_params = {
            'A': 23.0904,      # 暴雨公式参数A
            'C': 1.0116,      # 暴雨公式参数C
            'b': 18.8619,      # 暴雨公式参数b
            'n': 0.8418,      # 暴雨公式参数n    
        }

        # 降雨事件统计信息
        self.generated_events = []

    def set_time_step(self, time_step_min: int) -> None:
        """
        设置时间步长

        参数:
            time_step_min: 新的时间步长（分钟）
        """
        self.time_step_min = time_step_min
        print(f"时间步长已设置为: {time_step_min} 分钟")

    def get_default_params(self) -> Dict:
        """获取默认暴雨参数"""
        return self.default_params.copy()

    def set_default_params(self, A: float = None, b: float = None,
                           n: float = None, C: float = None) -> None:
        """
        设置默认暴雨参数

        参数:
            A, b, n, C: 暴雨公式参数
        """
        if A is not None:
            self.default_params['A'] = A
        if b is not None:
            self.default_params['b'] = b
        if n is not None:
            self.default_params['n'] = n
        if C is not None:
            self.default_params['C'] = C

        print("默认暴雨参数已更新:")
        print(f"  A = {self.default_params['A']}")
        print(f"  b = {self.default_params['b']}")
        print(f"  n = {self.default_params['n']}")
        print(f"  C = {self.default_params['C']}")

    def generate_triangle_rainfall(self, duration_hours: float, return_period: int = 10,
                                   peak_position: float = 0.5, **kwargs) -> np.ndarray:
        """
        生成三角形雨型暴雨过程线

        参数:
            duration_hours: 降雨历时（小时）
            return_period: 重现期（年）
            peak_position: 峰值位置（0-1之间，0.5表示在中点）
            **kwargs: 可覆盖默认暴雨参数 A, b, n

        返回:
            降雨强度序列（mm/h）
        """
        # 获取参数，使用kwargs覆盖默认值
        params = {**self.default_params, **kwargs}
        A = params['A']
        C = params['C']
        b = params['b']
        n = params['n']

        # 计算总时间步数
        total_steps = int(duration_hours * 60 / self.time_step_min)

        # 初始化降雨数组
        rainfall = np.zeros(total_steps)

        # 计算峰值时刻（小时）
        peak_time_hours = duration_hours * peak_position

        # 计算平均强度
        a = A * (1 + C * np.log10(return_period)) 
        i = a / (duration_hours * 60 + b) ** n
        H = i * duration_hours * 60
        peak_intensity = 2 * H / (duration_hours * 60)

        # 生成三角形降雨过程
        for i in range(total_steps):
            current_time_hours = i * self.time_step_min / 60

            if current_time_hours <= peak_time_hours:
                # 峰前上升段
                intensity = peak_intensity * current_time_hours / peak_time_hours
            else:
                # 峰后下降段
                intensity = peak_intensity * (duration_hours - current_time_hours) / (duration_hours - peak_time_hours)
            # 转换为mm/h并确保非负
            rainfall[i] = round(max(0, intensity * 60), 2)  # mm/min 转换为 mm/h

        # 记录生成信息
        self._record_event('triangle', duration_hours,
                           return_period, peak_position, rainfall)

        return rainfall

    def generate_chicago_rainfall(self, duration_hours: float, return_period: int = 10,
                                  peak_position: float = 0.4,
                                  **kwargs) -> np.ndarray:
        """
        生成芝加哥雨型暴雨过程线

        参数:
            duration_hours: 降雨历时（小时）
            return_period: 重现期（年）
            peak_intensity_factor: 峰值强度调整因子
            **kwargs: 可覆盖暴雨参数 A, b, n, r

        返回:
            降雨强度序列（mm/h）
        """
        params = {**self.default_params, **kwargs}

        A = params['A']
        C = params['C']
        b = params['b']
        n = params['n']
        r = peak_position    # 雨峰系数，默认0.4

        # 转换为分钟单位
        duration_min = duration_hours * 60
        time_steps = int(duration_min / self.time_step_min)

        # 初始化降雨强度数组
        rainfall = np.zeros(time_steps)

        # 计算峰值时刻（分钟）
        peak_time_min = r * duration_min

        a = A * (1 + C * np.log10(return_period)) 
        # 生成每个时间步的降雨强度
        for i in range(time_steps):
            current_time_min = (i + 0.5) * self.time_step_min  # 取时间步中点

            # 计算距离峰值的绝对值时间
            if current_time_min <= peak_time_min:
                # 峰前部分
                ta = peak_time_min - current_time_min
                intensity = a * ((1 - n) * ta / r + b ) / ((ta / r + b) ** (n + 1))
            else:
                # 峰后部分
                tb = current_time_min - peak_time_min
                intensity = a * ((1 - n) * tb / (1 - r) + b ) / ((tb / (1 - r) + b) ** (n + 1))


            # 转换为mm/h并确保非负
            rainfall[i] = round(max(0, intensity * 60), 2)  # mm/min 转换为 mm/h

        # 记录生成信息
        self._record_event('chicago', duration_hours,
                           return_period, peak_position, rainfall)

        return rainfall

    def generate_rainfall_event(self, seq_length: int = 288, rain_type: str = 'chicago',
                                duration_hours: float = 2, return_period: int = 10,
                                  peak_position: float = 0.4, start_idx: int = 12,
                                       **kwargs) -> np.ndarray:
        """
        生成随机降雨事件

        参数:
            seq_length: 序列总长度（默认288=24小时*12个5分钟）
            min_duration: 最小降雨历时（小时）
            max_duration: 最大降雨历时（小时）
            rain_type: 雨型类型 'triangle' 或 'chicago'
            **kwargs: 传递给具体雨型生成函数的参数

        返回:
            降雨强度序列（mm/h）
        """

        # 根据雨型生成降雨过程
        if rain_type == 'triangle':
            rainfall_event = self.generate_triangle_rainfall(
                duration_hours=duration_hours,
                return_period=return_period,
                peak_position=peak_position,
                **kwargs
            )
        elif rain_type == 'chicago':
            rainfall_event = self.generate_chicago_rainfall(
                duration_hours=duration_hours,
                return_period=return_period,
                peak_position=peak_position,
                **kwargs
            )
        else:
            raise ValueError(f"不支持的雨型: {rain_type}。请选择 'triangle' 或 'chicago'")

        # 补零到指定长度
        padded_rainfall = np.zeros(seq_length)

        # 降雨开始时间
        end_idx = start_idx + len(rainfall_event)
        padded_rainfall[start_idx:end_idx] = rainfall_event

        return padded_rainfall

    def generate_random_rainfall_event(self, seq_length: int = 288, min_duration: float = 1,
                                       max_duration: float = 6, rain_type: str = 'chicago',
                                       max_return_period: int = None,
                                       return_period: int = None,
                                       **kwargs) -> np.ndarray:
        """
        生成随机降雨事件

        参数:
            seq_length: 序列总长度（默认288=24小时*12个5分钟）
            min_duration: 最小降雨历时（小时）
            max_duration: 最大降雨历时（小时）
            rain_type: 雨型类型 'triangle' 或 'chicago'
            max_return_period: 最大重现期（用于过滤训练数据），若指定则仅选 ≤ 此值的重现期
            return_period: 指定固定重现期（用于测试），若指定则忽略 max_return_period
            **kwargs: 传递给具体雨型生成函数的参数

        返回:
            降雨强度序列（mm/h）
        """
        # 随机选择降雨历时
        duration = np.random.uniform(min_duration, max_duration)

        # 确定重现期选择池
        all_return_periods = [1, 2, 3, 5, 10, 20, 30, 50]
        all_weights = [0.40, 0.25, 0.15, 0.08, 0.05, 0.03, 0.02, 0.02]

        if return_period is not None:
            # 固定重现期（用于测试极端事件）
            chosen_return_period = return_period
        elif max_return_period is not None:
            # 只使用不超过 max_return_period 的重现期
            filtered = [(rp, w) for rp, w in zip(all_return_periods, all_weights) if rp <= max_return_period]
            rps, wts = zip(*filtered)
            wts = np.array(wts) / np.sum(wts)
            chosen_return_period = np.random.choice(rps, p=wts)
        else:
            chosen_return_period = np.random.choice(all_return_periods, p=all_weights)

        # 根据雨型生成降雨过程
        if rain_type == 'triangle':
            # 随机选择峰值位置
            peak_position = np.random.uniform(0.3, 0.7)
            rainfall_event = self.generate_triangle_rainfall(
                duration_hours=duration,
                return_period=chosen_return_period,
                peak_position=peak_position,
                **kwargs
            )
        elif rain_type == 'chicago':
            # 随机选择暴雨类型
            peak_position = np.random.uniform(0.3, 0.7)
            rainfall_event = self.generate_chicago_rainfall(
                duration_hours=duration,
                return_period=chosen_return_period,
                peak_position=peak_position,
                **kwargs
            )
        else:
            raise ValueError(f"不支持的雨型: {rain_type}。请选择 'triangle' 或 'chicago'")

        # 补零到指定长度
        padded_rainfall = np.zeros(seq_length)

        # 随机选择降雨开始时间，留足1小时用于排水峰延
        start_idx = np.random.randint(0, seq_length - len(rainfall_event) - 12)
        end_idx = start_idx + len(rainfall_event)
        padded_rainfall[start_idx:end_idx] = rainfall_event

        return padded_rainfall

    def generate_multiple_events(self, n_events: int, seq_length: int = 288, min_duration: float = 1,
                                 max_duration: float = 6, rain_type: str = 'chicago',
                                 max_return_period: int = None,
                                 return_period: int = None,
                                 **kwargs) -> List[np.ndarray]:
        """
        生成多个降雨事件

        参数:
            n_events: 降雨事件数量
            seq_length: 每个事件的序列长度
            max_return_period: 最大重现期（用于训练数据过滤）
            return_period: 固定重现期（用于测试极端事件）
            **kwargs: 传递给generate_random_rainfall_event的参数

        返回:
            降雨事件列表
        """
        events = []
        for i in range(n_events):
            if (i + 1) % 100 == 0:
                print(f"  已生成 {i+1}/{n_events} 场降雨事件")

            event = self.generate_random_rainfall_event(
                seq_length=seq_length, min_duration=min_duration,
                max_duration=max_duration, rain_type=rain_type,
                max_return_period=max_return_period,
                return_period=return_period,
                **kwargs)
            events.append(event)

        return events

    def _record_event(self, rain_type: str, duration: float, return_period: int, peak_position: float,
                      rainfall: np.ndarray) -> None:
        """
        记录生成的降雨事件信息

        参数:
            rain_type: 雨型
            duration: 降雨历时
            return_period: 重现期
            rainfall: 降雨序列
        """
        event_info = {
            'rain_type': rain_type,
            'duration_hours': duration,
            'return_period': return_period,
            'peak_position': peak_position,
            'max_intensity': rainfall.max(),
            'total_rainfall': rainfall.sum() * self.time_step_min / 60,  # mm
            'data': rainfall.copy()  # 保存副本
        }

        self.generated_events.append(event_info)

    def get_event_statistics(self, event_idx: int = -1) -> Dict:
        """
        获取降雨事件统计信息

        参数:
            event_idx: 事件索引，-1表示最新生成的事件

        返回:
            统计信息字典
        """
        if not self.generated_events:
            return {}

        event = self.generated_events[event_idx]
        rainfall = event['data']

        # 计算更多统计信息
        non_zero_mask = rainfall > 0
        non_zero_data = rainfall[non_zero_mask]

        stats = {
            '雨型': event['rain_type'],
            '历时(小时)': event['duration_hours'],
            '重现期(年)': event['return_period'],
            '雨峰系数': event['peak_position'],
            '最大强度(mm/h)': event['max_intensity'],
            '总降雨量(mm)': event['total_rainfall'],
            '平均强度(mm/h)': non_zero_data.mean() if len(non_zero_data) > 0 else 0,
            '降雨时间占比(%)': (non_zero_mask.sum() / len(rainfall)) * 100
        }

        return stats

    def plot_rainfall(self, rainfall: np.ndarray = None, event_idx: int = -1,
                      title: str = None, show_cumulative: bool = True) -> plt.Figure:
        """
        绘制降雨过程线

        参数:
            rainfall: 降雨序列，如果为None则使用event_idx指定的历史事件
            event_idx: 历史事件索引
            title: 图表标题
            show_cumulative: 是否显示累积降雨量

        返回:
            matplotlib图表对象
        """
        if rainfall is None:
            if not self.generated_events:
                raise ValueError("没有历史降雨事件，请提供rainfall参数")
            rainfall = self.generated_events[event_idx]['data']

        # 创建时间轴
        time_hours = np.arange(len(rainfall)) * self.time_step_min / 60

        # 创建图表
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # 绘制降雨强度柱状图
        bars = ax1.bar(time_hours, rainfall, width=self.time_step_min/60/1.5,
                       alpha=0.7, color='blue', edgecolor='darkblue')
        ax1.set_xlabel('时间 (小时)', fontsize=12)
        ax1.set_ylabel('降雨强度 (mm/h)', color='blue', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, alpha=0.3)

        # 添加标题
        if title is None:
            stats = self.get_event_statistics(
                event_idx) if rainfall is None else {}
            title = f'降雨过程线 - 最大强度: {rainfall.max():.1f} mm/h, 总降雨量: {rainfall.sum() * self.time_step_min/60:.1f} mm'
        ax1.set_title(title, fontsize=14, fontweight='bold')

        # 显示累积降雨量
        if show_cumulative:
            ax2 = ax1.twinx()
            cumulative = np.cumsum(rainfall) * \
                self.time_step_min / 60  # 累积降雨量 (mm)
            ax2.plot(time_hours, cumulative, 'r-', linewidth=2, label='累积降雨量')
            ax2.set_ylabel('累积降雨量 (mm)', color='red', fontsize=12)
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.set_ylim(bottom=0)

            # 添加图例
            ax2.legend(loc='upper left')

        plt.tight_layout()
        return fig

    def export_to_csv(self, rainfall: np.ndarray, filename: str = 'rainfall_data.csv') -> None:
        """
        将降雨数据导出为CSV文件

        参数:
            rainfall: 降雨序列
            filename: 输出文件名
        """
        # 创建时间列
        time_minutes = np.arange(len(rainfall)) * self.time_step_min

        # 创建数据
        data = np.column_stack([time_minutes, rainfall])

        # 保存为CSV
        np.savetxt(filename, data, delimiter=',',
                   header='Time(minutes),Intensity(mm/h)',
                   fmt=['%.1f', '%.2f'])

        print(f"降雨数据已保存到: {filename}")

    def export_to_swmm(self, rainfall: np.ndarray, filename: str = 'swmm_rainfall.txt') -> None:
        """
        将降雨数据导出为SWMM格式

        参数:
            rainfall: 降雨序列
            filename: 输出文件名
        """
        with open(filename, 'w') as f:
            f.write("; Rainfall data generated by RainfallGenerator\n")
            f.write("; Time(min) Intensity(mm/h)\n")

            for i, intensity in enumerate(rainfall):
                if intensity > 0:  # SWMM通常只记录有降雨的时间
                    time_min = i * self.time_step_min
                    f.write(f"{time_min} {intensity}\n")

        print(f"SWMM格式降雨数据已保存到: {filename}")

    def clear_history(self) -> None:
        """清除历史降雨事件记录"""
        self.generated_events.clear()
        print("历史降雨事件记录已清除")

    def get_summary(self) -> Dict:
        """
        获取暴雨生成器使用统计

        返回:
            统计信息字典
        """
        if not self.generated_events:
            return {"message": "尚未生成任何降雨事件"}

        total_events = len(self.generated_events)
        triangle_count = sum(
            1 for e in self.generated_events if e['rain_type'] == 'triangle')
        chicago_count = total_events - triangle_count

        # 计算平均统计
        avg_duration = np.mean([e['duration_hours']
                               for e in self.generated_events])
        avg_max_intensity = np.mean([e['max_intensity']
                                    for e in self.generated_events])
        avg_total_rainfall = np.mean(
            [e['total_rainfall'] for e in self.generated_events])

        summary = {
            "总事件数": total_events,
            "三角形雨型事件数": triangle_count,
            "芝加哥雨型事件数": chicago_count,
            "平均历时(小时)": avg_duration,
            "平均最大强度(mm/h)": avg_max_intensity,
            "平均总降雨量(mm)": avg_total_rainfall,
            "时间步长(分钟)": self.time_step_min,
        }

        return summary
