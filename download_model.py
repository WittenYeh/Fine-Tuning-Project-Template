import argparse
import os
from modelscope.hub.snapshot_download import snapshot_download
from termcolor import cprint
from modelscope.hub.api import HubApi

def download_model_from_modelscope(model_id: str, output_dir: str):
    """
    从 ModelScope 下载指定的模型到本地目录。

    参数:
    - model_id (str): 要下载的模型的ID，例如 'qwen/Qwen2-7B-Instruct'。
    - output_dir (str): 用于保存模型的本地目录。
    """
    cprint(f"--- 开始下载模型: {model_id} ---", "cyan", attrs=['bold'])
    cprint(f"目标保存目录: {os.path.abspath(output_dir)}", "cyan")

    try:
        # snapshot_download 函数会下载整个模型仓库
        # 它会将模型文件保存在 output_dir 下的一个与 model_id 相关的子目录中
        # ignore_file_pattern 参数可以用来排除不需要的文件，例如排除 .safetensors 文件以节省空间
        local_model_path = snapshot_download(
            model_id,
            cache_dir=output_dir,
            # revision='master' # 可以指定模型版本，默认为 'master'
        )
        
        cprint("\n--- ✅ 下载完成 ---", "green", attrs=['bold'])
        cprint(f"模型文件已成功保存到以下路径:", "green")
        print(local_model_path)

    except Exception as e:
        cprint(f"\n--- ❌ 下载失败 ---", "red", attrs=['bold'])
        cprint(f"错误原因: {e}", "red")
        cprint("请检查以下几点:", "yellow")
        print("1. 模型ID (model_id) 是否正确？")
        print("2. 网络连接是否正常？")
        print("3. 是否需要登录 ModelScope？(部分模型需要登录才能下载)")
        print("   - 登录方法: 在终端运行 'pip install modelscope' 后，执行 'modelscope login'")


if __name__ == "__main__":
    # 创建API实例
    api = HubApi()

    # 使用您的Token登录
    api.login('ms-bef1a8e2-a1df-4347-9af4-f24f806f5cab')

    print("成功登录到 ModelScope！")
    
    parser = argparse.ArgumentParser(description="从 ModelScope 下载模型到本地。")
    
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="要下载的 ModelScope 模型ID。例如: 'qwen/Qwen2-7B-Instruct' 或 'modelscope/Meta-Llama-3-8B-Instruct'。"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="~/models",
        help="保存模型的本地目录。默认为当前目录下的 '~/models' 文件夹。"
    )

    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    download_model_from_modelscope(args.model_id, args.output_dir)