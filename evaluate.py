# src/evaluate.py

import os
import glob
import gc
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from termcolor import cprint
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, classification_report
from transformers import AutoModelForCausalLM, AutoTokenizer

# 从项目其他模块导入
from config import Config
from utils import load_xml_data, clean_prediction, LABEL_MAP
from prompts import PROMPT_DICT


# --- 结果汇总与可视化 ---
def aggregate_results(base_dir, prompts_list):
    """遍历所有实验目录，汇总 training_history.csv 文件。"""
    all_history_files = glob.glob(os.path.join(base_dir, "*", "training_history.csv"))
    
    if not all_history_files:
        cprint(f"错误：在 '{base_dir}' 目录下未找到任何 'training_history.csv' 文件。", "red")
        cprint("请确认实验是否已成功运行并生成了结果文件。", "yellow")
        return pd.DataFrame()
    
    all_dfs = []
    for f_path in all_history_files:
        try:
            experiment_name = os.path.basename(os.path.dirname(f_path))
            prompt_type = None
            model_name_safe = None
            
            # 从后向前匹配，确保能正确处理包含prompt名的模型名
            for p in sorted(prompts_list, key=len, reverse=True):
                if experiment_name.endswith(f"_{p}"):
                    prompt_type = p
                    model_name_safe = experiment_name[:-len(f"_{p}")]
                    break
            
            if prompt_type is None:
                cprint(f"警告：无法从目录 '{experiment_name}' 中解析出Prompt类型。跳过此文件。", "yellow")
                continue
            
            # 替换回原始的'/'以匹配显示名称
            readable_model_name = model_name_safe.replace("_", "/")

            df = pd.read_csv(f_path)
            df['model_id'] = readable_model_name
            df['prompt_type'] = prompt_type
            all_dfs.append(df)
        except Exception as e:
            cprint(f"处理文件 {f_path} 时出错: {e}", "red")

    if not all_dfs:
        cprint("未能成功加载任何实验数据。", "red")
        return pd.DataFrame()
        
    return pd.concat(all_dfs, ignore_index=True)


def plot_metric_curves(results_df, base_dir):
    """绘制所有关键指标在训练过程中的变化曲线。"""
    cprint("--> 正在生成训练过程指标变化曲线图...", "blue")
    
    def plot_single_metric(metric_name, title, y_label):
        if metric_name not in results_df.columns:
            cprint(f"指标 '{metric_name}' 未在结果中找到，跳过绘制 '{title}' 图表。", "yellow")
            return
        
        sns.set_theme(style="whitegrid", palette="viridis")
        g = sns.FacetGrid(results_df, col="model_id", col_wrap=2, height=5, aspect=1.5, hue="prompt_type")
        g.map(sns.lineplot, "epoch", metric_name, marker='o', sort=False)
        g.add_legend(title='Prompt 类型')
        g.fig.suptitle(title, fontsize=18, y=1.03)
        g.set_axis_labels("训练轮次 (Epoch)", y_label)
        g.set_titles("模型: {col_name}")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # 保存图表
        filename = f"training_curve_{metric_name}.png"
        filepath = os.path.join(base_dir, filename)
        plt.savefig(filepath, dpi=300)
        cprint(f"已保存图表到: {filepath}", "green")
        plt.show()

    plot_single_metric('eval_accuracy', '各模型在不同Prompt下的测试集准确率变化曲线', '测试集准确率 (Test Accuracy)')
    plot_single_metric('eval_f1_score', '各模型在不同Prompt下的测试集 F1 Score 变化曲线', '测试集 F1 Score (Weighted)')
    plot_single_metric('eval_recall', '各模型在不同Prompt下的测试集召回率变化曲线', '测试集召回率 (Weighted Recall)')


def plot_final_performance(results_df, base_dir):
    """使用 catplot 同时展示 Accuracy, F1 Score, 和 Recall 的最终性能对比。"""
    cprint("--> 正在生成最终模型性能对比图...", "blue")
    
    final_performance_df = results_df.loc[results_df.groupby(['model_id', 'prompt_type'])['epoch'].idxmax()].copy()
    final_metrics_melted_df = final_performance_df.melt(
        id_vars=['model_id', 'prompt_type'], 
        value_vars=['eval_accuracy', 'eval_f1_score', 'eval_recall'],
        var_name='metric', value_name='score'
    )
    
    metric_labels = {
        'eval_accuracy': 'Accuracy', 'eval_f1_score': 'F1 Score (Weighted)', 'eval_recall': 'Recall (Weighted)'
    }
    final_metrics_melted_df['metric'] = final_metrics_melted_df['metric'].map(metric_labels)

    g = sns.catplot(
        data=final_metrics_melted_df, kind='bar', x='model_id', y='score', 
        hue='prompt_type', col='metric', palette='plasma', height=6, aspect=1.2
    )
    
    g.fig.suptitle('不同模型与Prompt组合的最终测试集性能对比', fontsize=20, y=1.03)
    g.set_axis_labels("模型", "分数")
    g.set_titles("指标: {col_name}")
    g.set_xticklabels(rotation=15, ha='right')
    g.set(ylim=(0, max(1.0, final_metrics_melted_df['score'].max() * 1.15)))
    g.legend.set_title("Prompt 类型")

    for ax in g.axes.flat:
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    filepath = os.path.join(base_dir, "final_performance_comparison.png")
    plt.savefig(filepath, dpi=300)
    cprint(f"已保存图表到: {filepath}", "green")
    plt.show()

# --- 生成预测文件 ---
def generate_predictions_for_all(config):
    """加载所有训练好的模型，在同一个完整的测试集上生成预测并保存。"""
    cprint("--- 准备固定的测试集 ---", "cyan")
    test_df = load_xml_data(config.TEST_DATASET_PATH)
    if test_df.empty:
        cprint("测试数据文件为空，无法生成预测。", "red")
        return pd.DataFrame()
        
    test_df.dropna(subset=['text', 'status'], inplace=True)
    test_df['corrected_status'] = test_df['status'].str.strip().str.upper().apply(lambda x: LABEL_MAP.get(x, "UNKNOWN"))
    cprint(f"已加载完整的测试集，包含 {len(test_df)} 条样本。", "green")

    all_predictions = []
    experiment_dirs = [d for d in glob.glob(os.path.join(config.OUTPUT_DIR_BASE, "*")) if os.path.isdir(d)]
    
    if not experiment_dirs:
        cprint("未找到任何实验结果目录，跳过预测生成。", "red")
        return pd.DataFrame()

    for exp_dir in tqdm(experiment_dirs, desc="处理所有实验"):
        model_path = os.path.join(exp_dir, "final_model")
        if not os.path.exists(model_path):
            continue
            
        exp_name = os.path.basename(exp_dir)
        prompt_type = next((p for p in sorted(config.PROMPTS_TO_RUN, key=len, reverse=True) if exp_name.endswith(f"_{p}")), None)
        if not prompt_type: continue
        
        model_name_safe = exp_name[:-len(f"_{prompt_type}")]
        readable_model_name = model_name_safe.replace("_", "/")
        
        cprint(f"\n--- 正在为模型 '{readable_model_name}' (Prompt: {prompt_type}) 在测试集上生成预测 ---\n", "blue")
        
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
            prompt_function = PROMPT_DICT[prompt_type]

            for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"推理 {readable_model_name}-{prompt_type}", leave=False):
                prompt = prompt_function(row['text'])
                
                inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, max_new_tokens=10, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id
                    )
                
                output_ids = outputs[0][inputs.input_ids.shape[1]:]
                generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
                
                all_predictions.append({
                    'id': row['ID'], 'model_id': readable_model_name, 'prompt_type': prompt_type,
                    'true_label': row['corrected_status'], 'predicted_label': clean_prediction(generated_text)
                })
        except Exception as e:
            cprint(f"处理 {exp_name} 时发生错误: {e}", "red")
        finally:
            # 清理内存
            if 'model' in locals(): del model
            if 'tokenizer' in locals(): del tokenizer
            gc.collect()
            torch.cuda.empty_cache()

    predictions_df = pd.DataFrame(all_predictions)
    if not predictions_df.empty:
        output_path = os.path.join(config.OUTPUT_DIR_BASE, config.PREDICTIONS_FILENAME)
        predictions_df.to_csv(output_path, index=False)
        cprint(f"\n所有预测结果已成功保存到: {output_path}", "green", attrs=['bold'])
    
    return predictions_df


def analyze_predictions(predictions_df, base_dir):
    """对预测结果进行详细分析，包括Kappa和分类报告，并绘图。"""
    if 'predictions_df' in locals() and not predictions_df.empty:
        # --- Cohen's Kappa ---
        cprint("--> 正在计算并可视化 Cohen's Kappa 系数...", "blue")
        kappa_results = []
        for name, group in predictions_df.groupby(['model_id', 'prompt_type']):
            kappa = cohen_kappa_score(group['true_label'], group['predicted_label'])
            kappa_results.append({'model_id': name[0], 'prompt_type': name[1], 'cohen_kappa': kappa})
        
        kappa_df = pd.DataFrame(kappa_results)
        cprint("--- Cohen's Kappa 系数计算结果 (基于完整测试集) ---", "cyan", attrs=['bold'])
        print(kappa_df.sort_values(by='cohen_kappa', ascending=False))

        plt.figure(figsize=(12, 7))
        sns.barplot(data=kappa_df, x='model_id', y='cohen_kappa', hue='prompt_type', palette='viridis')
        plt.title("各实验组合的 Cohen's Kappa 系数 (测试集)", fontsize=16)
        plt.xlabel("模型", fontsize=12); plt.ylabel("Cohen's Kappa", fontsize=12)
        plt.xticks(rotation=15, ha='right'); plt.tight_layout()
        filepath = os.path.join(base_dir, "cohen_kappa_comparison.png")
        plt.savefig(filepath, dpi=300); cprint(f"已保存图表到: {filepath}", "green"); plt.show()

        # --- 分类报告 ---
        cprint("\n--> 正在计算并可视化详细分类指标...", "blue")
        metrics = []
        for name, group in predictions_df.groupby(['model_id', 'prompt_type']):
            report = classification_report(group['true_label'], group['predicted_label'], output_dict=True, zero_division=0)
            avg = report['weighted avg']
            metrics.append({
                'model_id': name[0], 'prompt_type': name[1], 'precision': avg['precision'],
                'recall': avg['recall'], 'f1-score': avg['f1-score']
            })
        
        metrics_df = pd.DataFrame(metrics)
        cprint("--- 各实验组合的分类性能指标 (基于完整测试集, 加权平均) ---", "cyan", attrs=['bold'])
        print(metrics_df.sort_values(by='f1-score', ascending=False))
        
        metrics_melted_df = metrics_df.melt(id_vars=['model_id', 'prompt_type'], value_vars=['precision', 'recall', 'f1-score'], var_name='metric', value_name='score')
        g = sns.catplot(
            data=metrics_melted_df, kind='bar', x='model_id', y='score', hue='prompt_type',
            col='metric', palette='magma', height=6, aspect=1.2
        )
        g.fig.suptitle('各模型与Prompt组合的最终分类性能指标对比 (测试集)', fontsize=20, y=1.03)
        g.set_axis_labels("模型", "分数"); g.set_titles("指标: {col_name}")
        g.set_xticklabels(rotation=15, ha='right'); g.set(ylim=(0, 1))
        g.legend.set_title("Prompt 类型")
        for ax in g.axes.flat:
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontsize=9)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        filepath = os.path.join(base_dir, "classification_metrics_comparison.png")
        plt.savefig(filepath, dpi=300); cprint(f"已保存图表到: {filepath}", "green"); plt.show()
    else:
        cprint("预测数据框未找到或为空，无法进行分析。", "yellow")


def start_evaluation():
    """
    启动完整的评估流程：汇总结果、绘图、生成预测、深入分析。
    """
    config = Config()
    cprint("--- 开始评估流程 ---", "cyan", attrs=['bold'])

    # 1. 汇总训练历史并绘图
    cprint("\n--- 步骤 1: 汇总训练历史并绘制曲线图 ---", "blue")
    results_df = aggregate_results(config.OUTPUT_DIR_BASE, config.PROMPTS_TO_RUN)
    if not results_df.empty:
        cprint("\n成功汇总所有实验结果！", "green", attrs=['bold'])
        print("汇总数据预览 (指标基于测试集):"); print(results_df.head())
        plot_metric_curves(results_df, config.OUTPUT_DIR_BASE)
        plot_final_performance(results_df, config.OUTPUT_DIR_BASE)
    else:
        cprint("未能加载任何训练历史数据，跳过绘图。", "yellow")

    # 2. 为所有最终模型生成统一的预测文件
    cprint("\n--- 步骤 2: 为所有最终模型生成统一的预测文件 ---", "blue")
    predictions_path = os.path.join(config.OUTPUT_DIR_BASE, config.PREDICTIONS_FILENAME)
    if os.path.exists(predictions_path):
        cprint(f"检测到已存在的预测文件: {predictions_path}。将直接加载用于分析。", "yellow")
        predictions_df = pd.read_csv(predictions_path)
    else:
        predictions_df = generate_predictions_for_all(config)
    
    # 3. 对预测结果进行详细分析
    if not predictions_df.empty:
        cprint("\n--- 步骤 3: 对最终预测结果进行详细分析 ---", "blue")
        analyze_predictions(predictions_df, config.OUTPUT_DIR_BASE)
    else:
        cprint("未能生成或加载预测数据，无法进行详细分析。", "red")

    cprint("\n🎉🎉🎉 评估流程已完成！ 🎉🎉🎉", "green", attrs=['bold'])