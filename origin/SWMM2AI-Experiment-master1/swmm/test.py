from rainfall.generator import RainfallGenerator
from simulator import SWMMSimulator
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']


if __name__ == "__main__":
    # 运行示例
    print("SWMM水位模拟演示")
    print("=" * 50)

    # 创建暴雨生成器
    rg = RainfallGenerator(time_step_min=5)
    
    # 生成一场降雨
    rainfall = rg.generate_rainfall_event(
        rain_type = 'chicago',
        duration_hours=2,
        return_period=10,
        peak_position=0.4
    )
    
    print(f"生成降雨数据: {len(rainfall)} 个时间点")
    print(f"最大降雨强度: {rainfall.max():.2f} mm/h")
    print(f"总降雨量: {rainfall.sum() * 5/60:.2f} mm")
    rg.plot_rainfall(rainfall=rainfall, title="芝加哥雨型降雨事件")
    plt.show()

    # 创建SWMM模拟器
    simulator = SWMMSimulator(template_inp_path='template.inp', output_element='SN_001', output_type='node', output_variable='depth')

    # 运行模拟
    print("\n运行SWMM模拟...")
    start_time = datetime.now()
    
    results = simulator.run_swmm_simulation(
        rainfall_mm_h=rainfall
    )
    # 新增：判断模拟是否失败，失败就停止运行
    if results is None:
        print("模拟失败，无法输出结果！")
        exit()
    
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    print(f"模拟完成，耗时: {elapsed:.2f} 秒")
    print(f"要素类型: {results['type']}")
    print(f"要素编号: {results['id']}")
    print(f"结果变量: {results['variable']}")
    print(f"数据点数: {len(results['values'])}")
    print(f"最大水位: {results['values'].max():.2f} m")
    print(f"模拟方式: {'简化模拟' if results.get('simulated', False) else 'SWMM模拟'}")
    
    # 绘制结果
    fig = simulator.plot_simulation_results(rainfall, results, time_step_min=5)
    plt.show()
    
    
    # 批量生成数据集示例
    # print("\n" + "=" * 50)
    # print("批量生成SWMM数据集...")
    
    # dataset = SWMMRainfallWaterLevelDataset(
    #     n_events=20,  # 为了演示，使用较小数据集
    #     seq_length=144,  # 12小时
    #     predict_length=72,  # 6小时
    #     time_step_min=5
    # )
    
    # # 获取训练数据
    # X_train, y_train, X_val, y_val, X_test, y_test, scalers = dataset.get_training_data()
    
    # print(f"训练数据形状: X_train={X_train.shape}, y_train={y_train.shape}")
    # print(f"验证数据形状: X_val={X_val.shape}, y_val={y_val.shape}")
    # print(f"测试数据形状: X_test={X_test.shape}, y_test={y_test.shape}")
    
    # # 显示一个样本
    # sample_idx = 0
    # print(f"\n样本 {sample_idx} 统计:")
    # print(f"  输入降雨: {X_train[sample_idx].shape}")
    # print(f"  输出水位: {y_train[sample_idx].shape}")
    # print(f"  降雨范围: {X_train[sample_idx].min():.3f} - {X_train[sample_idx].max():.3f}")
    # print(f"  水位范围: {y_train[sample_idx].min():.3f} - {y_train[sample_idx].max():.3f}")