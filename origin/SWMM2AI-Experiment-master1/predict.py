import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
# 强制使用独立窗口后端
plt.switch_backend('TkAgg')  # Windows系统首选
# 如果上面不行，试试这个：plt.switch_backend('Qt5Agg')
from mlp import SimpleMLP
from lstm import SimpleLSTM, MultiScaleLSTM
from gru import SimpleGRU
from attention import AttentionLSTM, CausalAttentionLSTM
from residual import ResidualLSTM
from seq2seq import Seq2SeqLSTM
# from transformer import Transformer

from model import Predictor
from swmm.rainfall.generator import RainfallGenerator  
from swmm.simulator import SWMMSimulator

def test_single_prediction():
    """单个序列预测示例"""
    print("=== 单个序列预测示例 ===")
    # 0. 创建输出目录
    output_dir = create_next_folder(base_path='output')

    # 1. 加载预测器
    predictor = Predictor(model_path='simple_mlp_model.pth', output_dir=output_dir)
    
    # 2. 生成测试降雨序列
    # rg = RainfallGenerator(time_step_min=5)
    
    # test_rainfall = rg.generate_random_rainfall_event(
    #     seq_length=288,  # 必须与训练时相同的长度
    #     min_duration=2,
    #     max_duration=4,
    #     rain_type='chicago'
    # )

    rainfall_event = [2.4000000000000004
        ,2.4000000000000004
        ,0
        ,2.4000000000000004
        ,0
        ,0
        ,0
        ,0
        ,0
        ,0
        ,0
        ,0
        ,2.4000000000000004
        ,0
        ,2.4000000000000004
        ,0
        ,2.4000000000000004
        ,0
        ,2.4000000000000004
        ,0
        ,7.199999999999999
        ,4.800000000000001
        ,9.600000000000001
        ,2.4000000000000004
        ,2.4000000000000004
        ,4.800000000000001
        ,4.800000000000001
        ,4.800000000000001
        ,2.4000000000000004
        ,0
        ,0
        ,2.4000000000000004
        ,0
        ,0
        ,0
        ,0
        ,0
        ,7.199999999999999
        ,2.4000000000000004
        ,0
        ,0
        ,0
        ,0
        ,4.800000000000001
        ,9.600000000000001
        ,4.800000000000001
        ,4.800000000000001
        ,24
        ,0
        ,7.199999999999999
        ,7.199999999999999
        ,4.800000000000001
        ,2.4000000000000004
        ,0
        ,0
        ,0
        ,2.4000000000000004
        ,0
        ,0
        ,0
        ,0
        ,4.800000000000001
        ,9.600000000000001
        ,2.4000000000000004
        ,7.199999999999999
    ]

    test_rainfall = np.zeros(288)
    # 随机选择降雨开始时间，留足1小时用于排水峰延
    start_idx = np.random.randint(0, 288 - len(rainfall_event) - 12)
    end_idx = start_idx + len(rainfall_event)
    test_rainfall[start_idx:end_idx] = rainfall_event

    
    print(f"测试降雨序列长度: {len(test_rainfall)}")
    print(f"最大降雨强度: {test_rainfall.max():.1f} mm/h")
    print(f"总降雨量: {test_rainfall.sum() * 5/60:.1f} mm")
    
    # 3. 创建SWMM模拟
    simulator = SWMMSimulator(template_inp_path='template.inp', 
                              output_dir=output_dir,
                              output_element='SN_001', output_type='node', output_variable='depth')

    # 4. 预测水位
    predicted_water = predictor.predict(test_rainfall)
    
    print(f"预测水位序列长度: {len(predicted_water)}")
    print(f"预测最大水位: {predicted_water.max():.3f} m")
    print(f"预测平均水位: {predicted_water.mean():.3f} m")
    
    # 5. 可视化
    fig = predictor.visualize_prediction_with_swmm(
        rainfall_sequence=test_rainfall,
        water_predicted=predicted_water,
        time_step_min=5,
        swmm_simulator=simulator
    )

    save_path = os.path.join(output_dir, 'single_prediction_plot.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"预测图已保存至: {save_path}")
    # plt.show()

    return

def compare_multiple_prediction():
    """单个序列预测示例"""
    print("=== 单个序列预测示例 ===")
    # 0. 创建输出目录
    output_dir = create_next_folder(base_path='output')

    # 1. 加载预测器
    #models = ['simple_lstm_model.pth', 'simple_gru_model.pth',
           #   'simple_attention_lstm_model.pth', 'causal_attention_lstm_model.pth']
    # 修改后（按你实际训练的模型填写）
    models = [ #后改！
        'simple_lstm_model.pth',
        'simple_gru_model.pth',
        'attention_lstm_model.pth',
        'causal_attention_lstm_model.pth'
    ]
    predictors = []
    for model_path in models:
        predictors.append(Predictor(model_path=model_path, output_dir=output_dir))

    # 2. 生成测试降雨序列
    # rg = RainfallGenerator(time_step_min=5)
    
    # test_rainfall = rg.generate_random_rainfall_event(
    #     seq_length=288,  # 必须与训练时相同的长度
    #     min_duration=2,
    #     max_duration=4,
    #     rain_type='chicago'
    # )
    rainfall_event = [2.4000000000000004
        ,2.4000000000000004
        ,0
        ,2.4000000000000004
        ,0
        ,0
        ,0
        ,0
        ,0
        ,0
        ,0
        ,0
        ,2.4000000000000004
        ,0
        ,2.4000000000000004
        ,0
        ,2.4000000000000004
        ,0
        ,2.4000000000000004
        ,0
        ,7.199999999999999
        ,4.800000000000001
        ,9.600000000000001
        ,2.4000000000000004
        ,2.4000000000000004
        ,4.800000000000001
        ,4.800000000000001
        ,4.800000000000001
        ,2.4000000000000004
        ,0
        ,0
        ,2.4000000000000004
        ,0
        ,0
        ,0
        ,0
        ,0
        ,7.199999999999999
        ,2.4000000000000004
        ,0
        ,0
        ,0
        ,0
        ,4.800000000000001
        ,9.600000000000001
        ,4.800000000000001
        ,4.800000000000001
        ,24
        ,0
        ,7.199999999999999
        ,7.199999999999999
        ,4.800000000000001
        ,2.4000000000000004
        ,0
        ,0
        ,0
        ,2.4000000000000004
        ,0
        ,0
        ,0
        ,0
        ,4.800000000000001
        ,9.600000000000001
        ,2.4000000000000004
        ,7.199999999999999
    ]

    test_rainfall = np.zeros(288)
    # 随机选择降雨开始时间，留足1小时用于排水峰延
    start_idx = np.random.randint(0, 288 - len(rainfall_event) - 12)
    end_idx = start_idx + len(rainfall_event)
    test_rainfall[start_idx:end_idx] = rainfall_event

    
    print(f"测试降雨序列长度: {len(test_rainfall)}")
    print(f"最大降雨强度: {test_rainfall.max():.1f} mm/h")
    print(f"总降雨量: {test_rainfall.sum() * 5/60:.1f} mm")
    
    # 3. 创建SWMM模拟
    simulator = SWMMSimulator(template_inp_path='template.inp', 
                              output_dir=output_dir,
                              output_element='SN_001', output_type='node', output_variable='depth')
    swmm_results = simulator.run_swmm_simulation(rainfall_mm_h=test_rainfall)
    swmm_water_sequence = swmm_results['values']

    # 4. 预测水位
    predicted_water_dict = {}
    predicted_stats_dict = {}
    for predictor in predictors:
        water_predicted = predictor.predict(test_rainfall)
        predicted_water_dict[predictor.model_type] = water_predicted

        # 计算绝对误差
        absolute_error = np.abs(water_predicted - swmm_water_sequence)
        relative_error = np.abs(water_predicted - swmm_water_sequence) / (swmm_water_sequence + 1e-10) * 100
        # 计算并显示误差统计
        mse = np.mean((water_predicted - swmm_water_sequence) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(absolute_error)
        mape = np.mean(relative_error[swmm_water_sequence > 0])  # 避免除以0
        max_abs_error = np.max(absolute_error)
        max_rel_error = np.max(relative_error[swmm_water_sequence > 0])
        
        # R²计算
        ss_res = np.sum((swmm_water_sequence - water_predicted) ** 2)
        ss_tot = np.sum((swmm_water_sequence - np.mean(swmm_water_sequence)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        
        predicted_stats_dict[predictor.model_type] = (f"{predictor.model_type}\n"
            f"均方误差 (MSE): {mse:.4f}\n"
            f"均方根误差 (RMSE): {rmse:.4f}\n"
            f"平均绝对误差 (MAE): {mae:.4f}\n"
            f"平均绝对百分比误差 (MAPE): {mape:.2f}%\n"
            f"最大绝对误差: {max_abs_error:.4f}\n"
            f"最大相对误差: {max_rel_error:.2f}%\n"
            f"R²分数: {r2:.4f}")

    # 5. 可视化对比
    fig = compare(
        rainfall_sequence=test_rainfall,
        swmm_water_sequence=swmm_water_sequence,
        predicted_water_dict=predicted_water_dict,
        predicted_stats_dict=predicted_stats_dict
    )

    save_path = os.path.join(output_dir, 'comparison_plot.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"对比图已保存至: {save_path}")
    # plt.show()
    return

def create_next_folder(base_path=".", prefix="scenario"):
    """
    最简单的创建下一个编号文件夹
    """
    i = 1
    while os.path.exists(os.path.join(base_path, f"{prefix}_{i}")):
        i += 1
    os.makedirs(os.path.join(base_path, f"{prefix}_{i}"))
    return os.path.join(base_path, f"{prefix}_{i}")

def compare(rainfall_sequence, swmm_water_sequence, predicted_water_dict, predicted_stats_dict):
    seq_length = 288
    time_step_min = 5
    time_hours = np.arange(seq_length) * time_step_min / 60
    fig, ax = plt.subplots(figsize=(16, 10))
    # 绘制降雨（次坐标轴）
    ax_rain = ax.twinx()
    ax_rain.bar(time_hours, rainfall_sequence, width=time_step_min/60/1.5,
                      alpha=0.3, color='blue', label='降雨强度')
    ax_rain.set_ylabel('降雨强度 (mm/h)', color='blue')
    ax_rain.tick_params(axis='y', labelcolor='blue')
    
    ax.set_prop_cycle(color=['#f27970', '#54b345', '#8983bf', '#c76da2'])
    # 绘制水位
    for key, value in predicted_water_dict.items():
        ax.plot(time_hours, value, linewidth=2, label=key+'预测水位')

    for index, (key, value) in enumerate(predicted_stats_dict.items()):
        ax.text(0.02 + 0.15*index, 0.98, value, transform=ax.transAxes,
                    verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    ax.plot(time_hours, swmm_water_sequence, color='#05b9e2', linestyle='--', linewidth=2, label='SWMM模拟水位')

    ax.set_xlabel('时间 (小时)')
    ax.set_ylabel('水位 (m)', color='red')
    ax.tick_params(axis='y', labelcolor='red')
    ax.set_title('降雨与水位响应关系')
    ax.grid(True, alpha=0.3)


    # 合并图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_rain.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    fig.suptitle('多模型水位预测与SWMM模拟对比分析', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    return fig
# 定义要一起对比的模型文件（和你文件夹里的文件名完全一致）   #后修改的！
models = [
    'simple_lstm_model.pth',          # 普通 LSTM
    'simple_gru_model.pth',           # 普通 GRU
    'simple_attention_lstm_model.pth', # ✅ 你已经有的注意力 LSTM
    'causal_attention_lstm_model.pth'  # 因果注意力 LSTM（如果没训练就先删掉这行）
]
if __name__ == "__main__":
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    """主函数：运行所有示例"""
    print("水位预测模型 - 使用示例")
    print("=" * 50)
    
    try:
        # 示例1：单个序列预测  #修改后   多模型对比预测（把 models 传进去）
        compare_multiple_prediction()
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请先训练模型或确保模型文件存在。")
        print("运行 `python train.py` 训练模型。")
    except Exception as e:
        print(f"运行时错误: {e}")
        import traceback
        traceback.print_exc()