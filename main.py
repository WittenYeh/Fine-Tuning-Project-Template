# main.py

import argparse
import os
from termcolor import cprint

# 在导入其他模块之前，先进行一些基本的路径和环境设置
# 这样可以确保其他模块能正确找到彼此
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import login_huggingface, Config
from training import start_training
from evaluate import start_evaluation

def main():
    parser = argparse.ArgumentParser(
        description="使用LoRA微调大模型进行吸烟状态分类的项目",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "action",
        choices=["train", "evaluate"],
        help="选择要执行的操作:\n"
             "  train     - 根据config.py中的设置，开始训练所有模型。\n"
             "  evaluate  - 对所有已训练的模型进行结果汇总、分析和可视化。"
    )
    args = parser.parse_args()

    # 确保输出目录存在，避免在训练或评估过程中因目录不存在而出错
    config = Config()
    os.makedirs(config.OUTPUT_DIR_BASE, exist_ok=True)
    
    if args.action == "train":
        cprint("\n" + "="*50, "green", attrs=['bold'])
        cprint("               启动训练流程", "green", attrs=['bold'])
        cprint("="*50 + "\n", "green", attrs=['bold'])
        start_training()
    elif args.action == "evaluate":
        cprint("\n" + "="*50, "yellow", attrs=['bold'])
        cprint("               启动评估流程", "yellow", attrs=['bold'])
        cprint("="*50 + "\n", "yellow", attrs=['bold'])
        start_evaluation()

if __name__ == "__main__":
    main()