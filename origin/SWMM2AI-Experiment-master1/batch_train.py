# -*- coding: utf-8 -*-
"""
批量训练所有模型脚本
用于SWMM2AI毕设多模型对比实验
一键自动训练所有模型，自动保存不同pth文件
作者：豆包（专为你定制）
"""
import torch
from model import Trainer

def batch_train_all_models():
    """
    批量训练所有对比模型
    格式: (模型类型, 模型参数, 保存路径)
    """
    # 定义所有需要训练的模型列表（毕设常用对比模型）
    model_configs = [
        # 1. 基础MLP模型（对照组）
        {
            "model_type": "SimpleMLP",
            "model_params": {
                "input_size": 1,
                "num_layers": 2,
                "output_size": 1
            },
            "model_path": "simple_mlp_model.pth"
        },
        # 2. 基础LSTM模型（核心时序模型）
        {
            "model_type": "SimpleLSTM",
            "model_params": {
                "input_size": 1,
                "hidden_size": 128,
                "num_layers": 2,
                "output_size": 1
            },
            "model_path": "simple_lstm_model.pth"
        },
        # 3. 基础GRU模型（轻量级时序模型）
        {
            "model_type": "SimpleGRU",
            "model_params": {
                "input_size": 1,
                "hidden_size": 128,
                "num_layers": 2,
                "output_size": 1
            },
            "model_path": "simple_gru_model.pth"
        },
        # 4. 残差LSTM（深层高精度模型）
        {
            "model_type": "ResidualLSTM",
            "model_params": {
                "input_size": 1,
                "hidden_size": 128,
                "num_layers": 3,
                "output_size": 1,
                "dropout": 0.3
            },
            "model_path": "residual_lstm_model.pth"
        },
        # 5. 注意力LSTM（聚焦关键降雨时段）
        {
            "model_type": "AttentionLSTM",
            "model_params": {
                "input_size": 1,
                "hidden_size": 128,
                "num_layers": 2,
                "output_size": 1
            },
            "model_path": "attention_lstm_model.pth"
        }
    ]

    # 开始批量训练
    total_models = len(model_configs)
    for idx, config in enumerate(model_configs):
        print(f"\n==================================================")
        print(f"开始训练第 {idx+1}/{total_models} 个模型: {config['model_type']}")
        print(f"模型保存路径: {config['model_path']}")
        print(f"==================================================\n")

        try:
            # 初始化训练器
            trainer = Trainer(
                model_type=config["model_type"],
                model_params=config["model_params"],
                model_path=config["model_path"]
            )

            # 开始训练（使用默认参数：100个样本，200轮训练）
            model, dataset = trainer.train()

            print(f"\n✅ 模型 {config['model_type']} 训练完成并保存成功！")

        except Exception as e:
            print(f"\n❌ 模型 {config['model_type']} 训练失败: {str(e)}")
            continue

    print("\n==================================================")
    print("🎉 所有模型批量训练任务全部执行完毕！")
    print("📦 模型文件已全部保存在项目根目录")
    print("🔍 接下来可运行 predict.py 进行多模型对比实验")
    print("==================================================")

if __name__ == "__main__":
    # 设置PyTorch警告等级
    import warnings
    warnings.filterwarnings("ignore")

    # 启动批量训练
    batch_train_all_models()