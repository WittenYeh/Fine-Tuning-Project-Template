# src/training.py

import os
import gc
import math
import torch
import pandas as pd
from termcolor import cprint
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, classification_report

# 从项目其他模块导入
from prompts import PROMPT_DICT
from utils import LABEL_MAP, clean_prediction, load_xml_data
from config import Config

# --- 数据准备 ---

def prepare_train_dataset(df, tokenizer, prompt_func):
    """仅准备和处理训练数据集。"""
    cprint("--> 准备训练数据集...", "blue")
    df['corrected_status'] = df['status'].str.strip().str.upper().apply(lambda x: LABEL_MAP.get(x, "UNKNOWN"))
    df['prompt'] = df['text'].apply(prompt_func)
    df['full_text'] = df['prompt'] + "\n" + df['corrected_status']
    train_dataset = Dataset.from_pandas(df)
    def tokenize_train(examples):
        return tokenizer(examples["full_text"], truncation=True, padding="max_length", max_length=512)
    tokenized_train_dataset = train_dataset.map(tokenize_train, batched=True, remove_columns=train_dataset.column_names)
    cprint(f"训练集准备完成: {len(tokenized_train_dataset)} 条记录", "green")
    return tokenized_train_dataset

def prepare_eval_dataset(df, prompt_func):
    """仅准备和处理评估（测试）数据集。"""
    cprint("--> 准备评估(测试)数据集...", "blue")
    df['corrected_status'] = df['status'].str.strip().str.upper().apply(lambda x: LABEL_MAP.get(x, "UNKNOWN"))
    df['prompt'] = df['text'].apply(prompt_func)
    eval_dataset = Dataset.from_pandas(df)
    cprint(f"评估(测试)集准备完成: {len(eval_dataset)} 条记录", "green")
    return eval_dataset


# --- 自定义回调 ---

class EvaluateAndLogCallback(TrainerCallback):
    """在训练开始前和每个epoch结束时评估并记录指标的自定义回调"""
    def __init__(self, eval_dataset, tokenizer):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.history = []

    def _do_evaluate(self, model, epoch, state=None):
        """执行评估的核心逻辑"""
        cprint(f"\n--- 开始在测试集上评估 Epoch {epoch}... ---", "yellow", attrs=['bold'])
        predictions, true_labels = [], []
        model.eval()
        with torch.no_grad():
            for example in tqdm(self.eval_dataset, desc=f"Evaluating Epoch {epoch} on Test Set"):
                prompt = example['prompt']
                true_label = example['corrected_status']
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    temperature=0.2,
                    top_p=0.8,
                    do_sample=True
                )
                output_ids = outputs[0][inputs.input_ids.shape[1]:]
                generated_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                predicted_label = clean_prediction(generated_text)
                predictions.append(predicted_label)
                true_labels.append(true_label)
        
        accuracy = accuracy_score(true_labels, predictions)
        unique_labels = sorted(list(set(true_labels)))
        report = classification_report(
            true_labels, 
            predictions, 
            output_dict=True, 
            labels=unique_labels, 
            zero_division=0
        )
        weighted_avg = report.get('weighted avg', {})
        f1 = weighted_avg.get('f1-score', 0.0)
        recall = weighted_avg.get('recall', 0.0)

        train_loss, eval_loss = None, None
        if state and state.log_history:
            train_loss = next((log['loss'] for log in reversed(state.log_history) if 'loss' in log), None)
            eval_loss = next((log['eval_loss'] for log in reversed(state.log_history) if 'eval_loss' in log), None)
        
        cprint(f"Epoch {epoch} 评估结果 (测试集):", "green", attrs=['bold'])
        cprint(f"  - 测试集准确率 (Test Accuracy): {accuracy:.4f}", "green")
        cprint(f"  - 测试集 F1 Score (Weighted): {f1:.4f}", "green")
        cprint(f"  - 测试集召回率 (Weighted Recall): {recall:.4f}", "green")
        if eval_loss is not None:
            cprint(f"  - 测试集损失 (Test Loss): {eval_loss:.4f}", "green")

        self.history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "eval_accuracy": accuracy,
            "eval_f1_score": f1,
            "eval_recall": recall
        })
        model.train()

    def on_train_begin(self, args, state, control, model, **kwargs):
        cprint("--- 在正式训练开始前，执行基线性能评估 (Epoch 0 on Test Set) ---", "magenta", attrs=['bold'])
        self._do_evaluate(model, epoch=0)

    def on_epoch_end(self, args, state, control, model, **kwargs):
        self._do_evaluate(model, epoch=round(state.epoch), state=state)

# --- 主训练函数 ---

def run_experiment(model_id, prompt_type, config):
    """执行单次完整的训练实验"""

    # --- 步骤 1: 加载数据 ---
    cprint("--- 步骤 1: 加载数据 ---", "cyan", attrs=['bold'])
    train_df = load_xml_data(config.TRAIN_DATASET_PATH)
    test_df = load_xml_data(config.TEST_DATASET_PATH)
    if train_df.empty or test_df.empty:
        cprint("训练数据或测试数据为空，跳过此实验。", "red")
        return
    train_df.dropna(subset=['text', 'status'], inplace=True)
    test_df.dropna(subset=['text', 'status'], inplace=True)
    
    # --- 步骤 2: 加载模型和分词器 ---
    cprint(f"--- 步骤 2: 加载模型 '{model_id}' ---", "cyan", attrs=['bold'])
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config, 
        device_map="auto", 
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    prompt_function = PROMPT_DICT[prompt_type]
    train_dataset = prepare_train_dataset(train_df, tokenizer, prompt_function)
    test_dataset = prepare_eval_dataset(test_df, prompt_function)
    
    # --- 步骤 3: 配置 LoRA ---
    cprint("--- 步骤 3: 配置 LoRA ---", "cyan", attrs=['bold'])
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'])
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # --- 步骤 4: 设置并开始训练 ---
    cprint("--- 步骤 4: 设置并开始训练 ---", "cyan", attrs=['bold'])
    evaluation_callback = EvaluateAndLogCallback(eval_dataset=test_dataset, tokenizer=tokenizer)
    model_name_safe = model_id.replace("/", "_")
    output_dir_experiment = os.path.join(config.OUTPUT_DIR_BASE, f"{model_name_safe}_{prompt_type}")

    num_training_samples = len(train_dataset)
    # gradient_accumulation_steps = 4
    effective_batch_size = config.BATCH_SIZE * 4 
    steps_per_epoch = math.ceil(num_training_samples / effective_batch_size)
    cprint(f"计算得出每个 Epoch 的步数: {steps_per_epoch}", "blue")

    training_args = TrainingArguments(
        output_dir=output_dir_experiment,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=4,
        num_train_epochs=config.EPOCHS,
        learning_rate=config.LEARNING_RATE,
        optim="paged_adamw_8bit",
        logging_steps=steps_per_epoch,
        # eval_strategy and save_strategy are set to "epoch" by default with save_strategy="epoch"
        save_strategy="epoch",
        fp16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[evaluation_callback],
    )
    
    cprint(f"模型设备分布情况: {model.hf_device_map}", "yellow")
    cprint(f"Trainer将使用的设备: {trainer.args.device}", "yellow")
    
    cprint("即将开始训练...", "green", attrs=['bold'])
    trainer.train()
    cprint("训练完成！", "green", attrs=['bold'])
    
    # --- 步骤 5: 保存最终结果 ---
    cprint(f"--- 步骤 5: 保存最终结果 ---", "cyan", attrs=['bold'])
    final_output_dir = os.path.join(output_dir_experiment, "final_model")
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    cprint(f"模型和分词器已成功保存到 {final_output_dir}。", "green")

    history_df = pd.DataFrame(evaluation_callback.history)
    history_path = os.path.join(output_dir_experiment, "training_history.csv")
    history_df.to_csv(history_path, index=False)
    cprint(f"训练历史已保存到 {history_path}。", "green")
    cprint("训练历史记录 (在测试集上评估):", "yellow")
    print(history_df)


def start_training():
    """
    启动所有在Config中定义的训练实验。
    """
    config = Config()
    
    models_to_run = config.LOCAL_MODEL_PATH if config.LOCAL_MODEL_PATH else config.MODELS_TO_RUN
    
    for model_id in models_to_run:
        for prompt_type in config.PROMPTS_TO_RUN:
            model_name_safe = model_id.replace("/", "_")
            final_model_path = os.path.join(config.OUTPUT_DIR_BASE, f"{model_name_safe}_{prompt_type}", "final_model")

            if os.path.exists(final_model_path):
                cprint(f"\n{'='*80}", "yellow", attrs=['bold'])
                cprint(f"⏭️  检测到已存在的模型目录: {final_model_path}", "yellow")
                cprint(f"跳过训练: 模型 = {model_id}, Prompt类型 = {prompt_type}", "yellow")
                cprint(f"{'='*80}\n", "yellow", attrs=['bold'])
                continue

            cprint(f"\n{'='*80}", "blue", attrs=['bold'])
            cprint(f"🚀 开始新实验: 模型 = {model_id}, Prompt类型 = {prompt_type}", "blue", attrs=['bold'])
            cprint(f"{'='*80}", "blue", attrs=['bold'])
            
            run_experiment(model_id, prompt_type, config)
            
            cprint(f"\n{'~'*80}", "magenta")
            cprint(f"✅ 实验结束: 模型 = {model_id}, Prompt类型 = {prompt_type}", "magenta", attrs=['bold'])
            cprint("清理内存...", "magenta")
            gc.collect()
            torch.cuda.empty_cache()
            cprint(f"{'~'*80}\n", "magenta")

    cprint("🎉🎉🎉 所有训练实验均已完成！ 🎉🎉🎉", "green", attrs=['bold'])
    