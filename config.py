# src/config.py

import os
from huggingface_hub import login
from termcolor import cprint

def login_huggingface():
    """
    从环境变量 HUGGING_FACE_TOKEN 登录到 Hugging Face Hub。
    为了安全，强烈建议使用环境变量来管理你的Token。
    在你的终端中设置: export HUGGING_FACE_TOKEN='hf_YOUR_TOKEN_HERE'
    """
    TOKEN = os.getenv("HUGGING_FACE_TOKEN")
    
    if TOKEN:
        try:
            login(token=TOKEN)
            cprint("✅ Hugging Face Login Successful!", "green")
            return True
        except Exception as e:
            cprint(f"❌ Hugging Face Login Failed: {e}", "red")
            return False
    else:
        cprint("WARNING: HUGGING_FACE_TOKEN not found in environment variables.", "yellow")
        cprint("Please set it by running: export HUGGING_FACE_TOKEN='your_token'", "yellow")
        cprint("Attempting to run without explicit login...", "yellow")
        return False

class Config:
    """
    配置所有实验参数。
    在这里修改所有配置，无需改动其他代码。
    """
    # --- 实验组合 ---
    # 要进行实验的基础模型列表 (使用Hugging Face Hub ID)
    MODELS_TO_RUN = [
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct"
    ]
    
    # LOCAL_MODEL_PATH = None
    LOCAL_MODEL_PATH = [
        "../../models/Qwen2-7B-Instruct",
        "../../models/Llama-3.1-8B-hf",
    ]
    
    # 要进行实验的Prompt类型列表 ("base", "example", "chain")
    PROMPTS_TO_RUN = ["base", "example", "chain"]
    
    # --- 路径配置 ---
    # 训练数据集路径
    TRAIN_DATASET_PATH = "../data/train.xml"
    # 测试数据集路径
    TEST_DATASET_PATH = "../data/test-ground-truth.xml"
    
    # 保存所有实验结果的根目录
    OUTPUT_DIR_BASE = "../results"

    # --- 训练参数 ---
    EPOCHS = 3
    BATCH_SIZE = 2      # 每个设备的训练批次大小
    LEARNING_RATE = 2e-4
    
    # --- 评估文件配置 ---
    # 汇总所有模型预测结果的文件名
    PREDICTIONS_FILENAME = "all_model_predictions.csv"