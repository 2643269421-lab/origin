from generator import RainfallGenerator
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']

def example_usage():
    """暴雨生成器使用示例"""
    
    print("=== 暴雨生成器使用示例 ===")
    
    # 0. 创建暴雨生成器实例
    print("\n0. 创建暴雨生成器...")
    rg = RainfallGenerator(time_step_min=5)
    print(f"时间步长: {rg.time_step_min} 分钟")
    
    # 1. 修改默认参数
    print("\n1. 修改默认暴雨参数...")
    rg.set_default_params(A=23.0904, C=1.0116, n=0.8418, b=18.8619)

    # 2. 生成三角形雨型
    print("\n2. 生成三角形雨型...")
    triangle_rain = rg.generate_triangle_rainfall(
        duration_hours=2,
        return_period=10,
        peak_position=0.4  # 峰值在40%时间处
    )
    print(f"三角形雨型长度: {len(triangle_rain)} 个时间步")
    rg.plot_rainfall(rainfall=triangle_rain, title="三角形雨型降雨事件")
    plt.show()

    # 3. 生成芝加哥雨型
    print("\n3. 生成芝加哥雨型...")
    chicago_rain = rg.generate_chicago_rainfall(
        duration_hours=2,
        return_period=10,
        peak_position=0.4
    )
    print(f"芝加哥雨型长度: {len(chicago_rain)} 个时间步")
    rg.plot_rainfall(rainfall=chicago_rain, title="芝加哥雨型降雨事件")
    plt.show()
    
    # 4. 获取统计信息
    print("\n4. 最新降雨事件统计:")
    stats = rg.get_event_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 5. 生成随机降雨事件
    print("\n5. 生成随机降雨事件...")
    random_rain = rg.generate_random_rainfall_event(
        seq_length=288,  # 24小时
        min_duration=2,
        max_duration=6,
        rain_type='chicago'
    )
    print(f"随机降雨事件长度: {len(random_rain)} 个时间步")
    rg.plot_rainfall(rainfall=random_rain, title="随机生成的24小时降雨事件")
    plt.show()
    
    # 6. 生成多个事件
    print("\n6. 生成多个降雨事件...")
    events = rg.generate_multiple_events(
        n_events=4,
        seq_length=120,
        min_duration=2,
        max_duration=6,
        rain_type='chicago'
    )
    plot_4events(events[0],events[1],events[2],events[3])
    plt.show()
    print(f"生成了 {len(events)} 个降雨事件")
    
    # 7. 获取生成器使用统计
    print("\n7. 生成器使用统计:")
    summary = rg.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # 8. 导出数据
    print("\n8. 导出数据...")
    rg.export_to_csv(chicago_rain, "example_rainfall.csv")
    rg.export_to_swmm(chicago_rain, "example_swmm_rainfall.txt")
    
    return

def plot_4events(event1, event2, event3, event4):
    seq_length = len(event1.data)
    time_step_min = 5
    time_hours = np.arange(seq_length) * time_step_min / 60
    fig, axes = plt.subplots(2,2,figsize=(12, 6))
   
    ax1 = axes[0,0]
    # 绘制降雨强度柱状图
    ax1.bar(time_hours, event1.data, width=time_step_min/60/1.5,
                    alpha=0.7, color='#05b9e2', edgecolor='darkblue')
    ax1.set_xlabel('Time (h)', fontsize=12)
    ax1.set_ylabel('Intensity (mm/h)', color='blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.3)
    # 添加标题
    ax1.set_title("Rainfall Event 1", fontsize=14, fontweight='bold')

    ax2 = axes[0,1]
    # 绘制降雨强度柱状图
    ax2.bar(time_hours, event2.data, width=time_step_min/60/1.5,
                    alpha=0.7, color='#05b9e2', edgecolor='darkblue')
    ax2.set_xlabel('Time (h)', fontsize=12)
    ax2.set_ylabel('Intensity (mm/h)', color='blue', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.grid(True, alpha=0.3)
    # 添加标题
    ax2.set_title("Rainfall Event 2", fontsize=14, fontweight='bold')

    ax3 = axes[1,0]
    # 绘制降雨强度柱状图
    ax3.bar(time_hours, event3.data, width=time_step_min/60/1.5,
                    alpha=0.7, color='#05b9e2', edgecolor='darkblue')
    ax3.set_xlabel('Time (h)', fontsize=12)
    ax3.set_ylabel('Intensity (mm/h)', color='blue', fontsize=12)
    ax3.tick_params(axis='y', labelcolor='blue')
    ax3.grid(True, alpha=0.3)
    # 添加标题
    ax3.set_title("Rainfall Event 3", fontsize=14, fontweight='bold')

    ax4 = axes[1,1]
    # 绘制降雨强度柱状图
    ax4.bar(time_hours, event4.data, width=time_step_min/60/1.5,
                    alpha=0.7, color='#05b9e2', edgecolor='darkblue')
    ax4.set_xlabel('Time (h)', fontsize=12)
    ax4.set_ylabel('Intensity (mm/h)', color='blue', fontsize=12)
    ax4.tick_params(axis='y', labelcolor='blue')
    ax4.grid(True, alpha=0.3)
    # 添加标题
    ax4.set_title("Rainfall Event 4", fontsize=14, fontweight='bold')
    plt.tight_layout()


if __name__ == "__main__":
    # 运行示例
    print("暴雨生成器类演示")
    print("=" * 50)
    
    # 基础示例
    example_usage()
    
