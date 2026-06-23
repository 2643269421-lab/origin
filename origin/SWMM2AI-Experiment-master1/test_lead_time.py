# 实验1：超前预报测试
import torch
import numpy as np
import pandas as pd
import os
from dataset import *  # 复用原有数据集加载
from lstm import SimpleLSTM
from gru import SimpleGRU
from attention import AttentionLSTM, CausalAttentionLSTM
import matplotlib.pyplot as plt

# ===================== 基础配置（与原项目完全一致，关键！）=====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 1  # 单输入：降雨强度
HIDDEN_SIZE = 128  # 与原训练一致
NUM_LAYERS = 2  # 与原训练一致
OUTPUT_SIZE = 1  # 单输出：节点水位
BATCH_SIZE = 32  # 与原训练一致
# 超前预报时长配置：key=时长名称，value=对应时间步长（5分钟/步）
LEAD_TIME_CONFIG = {
    "1h": 12,
    "3h": 36,
    "6h": 72,
    "12h": 144
}
# 模型与权重文件映射（严格对应仓库中的pth文件）
MODEL_MAP = {
    "SimpleLSTM": (SimpleLSTM, "simple_lstm_model.pth"),
    "SimpleGRU": (SimpleGRU, "simple_gru_model.pth"),
    "AttentionLSTM": (AttentionLSTM, "simple_attention_lstm_model.pth"),
    "CausalAttentionLSTM": (CausalAttentionLSTM, "causal_attention_lstm_model.pth")
}
# 数据路径：与原dataset.py中的测试集路径一致，若原项目有修改请同步
TEST_DATA_PATH = "./data/test/"  # 替换为你实际的测试集路径
# 结果保存路径
SAVE_PATH = "./results/lead_time_results.csv"


# ===================== 指标计算函数（与原论文完全一致，关键！）=====================
def calculate_metrics(y_true, y_pred):
    """
    计算RMSE/NSE/PAE/PRE
    :param y_true: 真实值，np.array
    :param y_pred: 预测值，np.array
    :return: 各指标字典
    """
    # 去除NaN（若有）
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    # NSE
    nse = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    # 峰值误差PAE/PRE
    y_true_peak = np.max(y_true)
    y_pred_peak = np.max(y_pred)
    pae = np.abs(y_true_peak - y_pred_peak)
    pre = (pae / y_true_peak) * 100 if y_true_peak != 0 else 0.0

    return {
        "RMSE": round(rmse, 6),
        "NSE": round(nse, 6),
        "PAE": round(pae, 6),
        "PRE(%)": round(pre, 2)
    }


# ===================== 加载测试集并调整超前预报输入输出窗口 =====================
def load_lead_time_data(lead_step):
    """
    加载测试集并按超前步长调整输入输出：输入t-k~t的降雨，输出t+lead_step的水位
    :param lead_step: 超前时间步长
    :return: test_loader (DataLoader)
    """
    # 复用原dataset.py的数据集读取逻辑，此处以原项目的SWMMDataset为例，若原类名不同请同步
    # 原dataset.py中若有归一化，此处完全复用，保证与训练一致
    test_dataset = SWMMDataset(
        data_path=TEST_DATA_PATH,
        seq_len=None,  # 自定义序列长度
        lead_step=lead_step  # 超前步长
    )
    # 生成超前预报的输入输出对
    rain_data, water_data = test_dataset.get_data()  # 原数据集的降雨和水位数组，shape=(N, )
    X, y = [], []
    # 输入序列长度=超前步长（与实验设计一致：t-lead_step+1 ~ t的降雨）
    seq_len = lead_step
    for i in range(seq_len, len(rain_data) - lead_step):
        X.append(rain_data[i - seq_len:i].reshape(-1, 1))  # 输入：降雨序列，shape=(seq_len, 1)
        y.append(water_data[i + lead_step])  # 输出：超前lead_step的水位
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)

    # 转换为TensorDataset
    X_tensor = torch.from_numpy(X).to(DEVICE)
    y_tensor = torch.from_numpy(y).to(DEVICE)
    test_dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return test_loader


# ===================== 模型测试主函数 =====================
def test_lead_time():
    # 初始化结果字典
    final_results = []
    # 遍历每个超前预报时长
    for lead_name, lead_step in LEAD_TIME_CONFIG.items():
        print(f"===== 测试超前预报时长：{lead_name}（{lead_step}步） =====")
        # 加载对应步长的测试集
        test_loader = load_lead_time_data(lead_step)
        # 遍历4种模型
        for model_name, (model_cls, model_path) in MODEL_MAP.items():
            print(f"--- 测试模型：{model_name} ---")
            # 1. 初始化模型（与原训练一致）
            model = model_cls(
                input_size=INPUT_SIZE,
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                output_size=OUTPUT_SIZE
            ).to(DEVICE)
            # 2. 加载预训练权重（关键：不重新训练，直接加载）
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            else:
                raise FileNotFoundError(f"模型权重文件{model_path}不存在，请检查仓库文件！")
            # 3. 模型评估模式
            model.eval()
            y_true_all = []
            y_pred_all = []
            # 4. 前向推理（无梯度，加快速度）
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    y_pred = model(X_batch)
                    # 保存真实值和预测值
                    y_true_all.append(y_batch.cpu().numpy())
                    y_pred_all.append(y_pred.cpu().numpy())
            # 5. 合并结果并计算指标
            y_true = np.concatenate(y_true_all, axis=0)
            y_pred = np.concatenate(y_pred_all, axis=0)
            metrics = calculate_metrics(y_true, y_pred)
            # 6. 保存结果
            result = {
                "Model": model_name,
                "Lead_Time": lead_name,
                "Lead_Step": lead_step,
                **metrics
            }
            final_results.append(result)
            print(f"{model_name} - {lead_name}：{metrics}")
    # 7. 保存结果到CSV
    df = pd.DataFrame(final_results)
    df.to_csv(SAVE_PATH, index=False, encoding="utf-8-sig")
    print(f"\n实验1结果已保存至：{SAVE_PATH}")
    return df


# ===================== 运行测试 =====================
if __name__ == "__main__":
    # 创建results目录（若不存在）
    if not os.path.exists("./results"):
        os.makedirs("./results")
    # 运行测试
    test_lead_time()