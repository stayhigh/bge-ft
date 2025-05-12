from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset
import torch.distributed as dist
import os
import numpy as np
from datetime import datetime
import wandb


# 设置 CUDA 架构列表
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

def print_dataset(dataset):
    print(f"The number of training samples: {len(dataset['train'])}")
    print(f"The number of validation samples: {len(dataset['validation'])}")
    print(f"The number of test samples: {len(dataset['test'])}")

def load_data_from_json():
    dataset_name = "clue/afqmc"
    dataset = load_dataset("json", data_files={"train": "./data/afqmc/afqmc_train.json", "validation": "./data/afqmc/afqmc_validation.json"})
    cache_dir = "./cached_data"
    os.makedirs(cache_dir, exist_ok=True)
    print(f"downloading {dataset_name}...")
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    return dataset

def main():
    # 加载预训练的 bge-base 模型和分词器
    model_name = "BAAI/bge-base-zh"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # 加载 AFQMC 数据集
    cache_dir = "./cached_data"
    os.makedirs(cache_dir, exist_ok=True)
    data_path = "clue"
    data_name = "afqmc"
    dataset_name = f"{data_path}/{data_name}"
    dataset =  load_dataset(data_path, data_name, cache_dir=cache_dir)
    print_dataset(dataset)



    # 定义数据预处理函数
    def preprocess_function(examples):
        return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True)

    # 对数据集进行预处理
    encoded_dataset = dataset.map(preprocess_function, batched=True)

    # 定义数据整理器
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # Replace slashes in model and dataset names for consistent logging
    safe_model_name = model_name.replace('/', '_')
    safe_dataset_name = dataset_name.replace('/', '_')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{safe_model_name}-{safe_dataset_name}-train-{timestamp}"
    
    # wandb 初始化
    wandb.init(
        project='bge-ft-afqm',
        name="finetune bge model with afqmc dataset",
        id = run_name,
        config={
            "model_name": model_name,
            "dataset_name": dataset_name, 
            "timestamp": timestamp
        }
    )

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir="./finetuned_models/finetuned_models_with_afqmc_" + timestamp,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        logging_dir="./logs",
        metric_for_best_model="eval_accuracy",
        load_best_model_at_end=True, # NOTE: --load_best_model_at_end requires the save and eval strategy to match
        # deepspeed="ds_config.json",  # NOTE: 可以指定 DeepSpeed 配置文件，但建議使用deepspeed命令參數 --deepspeed_config
        bf16=True,  # 启用混合精度训练
        ddp_find_unused_parameters=False,
        learning_rate=2e-5,
        run_name=run_name,
        report_to="all" # 使用默認選項，all選項可以觸發wandb
    )
    wandb.run.tags = ["finetuning", "afqmc", "bert-base"]  # 添加实验标签

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = (predictions == labels).mean()
        wandb.log({"accuracy": accuracy, "epoch": eval_pred[0].shape[0]})  # 添加 epoch 参数
        return {"accuracy": accuracy}
    
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=10,
        early_stopping_threshold=0.01
    )

    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[early_stopping_callback],
        processing_class=tokenizer  # 使用新的参数名
    )

    # 开始训练
    try:
        trainer.train()
    finally:
        cleanup()
        wandb.finish()  # 确保wandb清理

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
