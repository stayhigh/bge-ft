from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, EarlyStoppingCallback, Trainer
from datasets import load_dataset
import torch.distributed as dist
import os
import numpy as np
from datetime import datetime


# 设置 CUDA 架构列表
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
os.environ["XFORMERS_MORE_DETAILS"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# setting ss_proxy
SS_PROXY = "http://127.0.0.1:2001"
os.environ["http_proxy"] = SS_PROXY
os.environ["https_proxy"] = SS_PROXY
os.environ["HTTP_RPOXY"] = SS_PROXY
os.environ["HTTPS_PROXY"] = SS_PROXY
os.environ["NO_PROXY"] = "localhost,127.0.0.1"


def print_dataset(dataset):
    print(f"The number of training samples: {len(dataset['train'])}")
    print(f"The number of validation samples: {len(dataset['validation'])}")
    print(f"The number of test samples: {len(dataset['test'])}")

def main():
    # 加载预训练的 bge-base 模型和分词器
    model_name = "BAAI/bge-base-zh"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 获取标签类别数量
    # 假设数据集有6个情感类别（根据num_labels=6配置）
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)
    
    # 加载 AFQMC 数据集    
    dataset_name = "dair-ai/emotion"
    cache_dir = "./cached_data"
    os.makedirs(cache_dir, exist_ok=True)
    print(f"downloading {dataset_name}...")
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)

    # 打印數據量
    print_dataset(dataset)

    def preprocess_function_for_dairai_emotion(examples):
        # 处理文本和标签
        tokenized_inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
        # 添加标签处理（假设数据集中的标签字段为"label"）
        tokenized_inputs["labels"] = examples["label"]
        return tokenized_inputs

    # 对数据集进行预处理
    encoded_dataset = dataset.map(preprocess_function_for_dairai_emotion, batched=True)

    # 定义数据整理器
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    timestamp =  str(datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir="./finetuned_models/finetuned_models_with_emotion_" + timestamp,
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
        # deepspeed="ds_configs/ds_config.json",  # NOTE: 可以指定 DeepSpeed 配置文件，但建議使用deepspeed命令參數 --deepspeed_config
        bf16=True,  # 启用混合精度训练
        ddp_find_unused_parameters=True,
        run_name=f"{model_name}-{dataset_name}-train-" + timestamp,
        report_to="all" # 使用默認選項，all選項可以觸發wandb
    )

    # 打印training_args每項配置
    print(f"Training arguments: {training_args}")

    from sklearn.metrics import classification_report

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # 计算多分类指标
        report = classification_report(labels, predictions, output_dict=True)
        
        # 提取每个类别的指标
        class_reports = {}
        for label in range(6):  # 假设共有6个情感类别
            class_name = f"class_{label}"
            class_reports[f"{class_name}_precision"] = report[str(label)]["precision"]
            class_reports[f"{class_name}_recall"] = report[str(label)]["recall"]
            class_reports[f"{class_name}_f1-score"] = report[str(label)]["f1-score"]
        
        # 计算整体指标
        overall_metrics = {
            "accuracy": (predictions == labels).mean(),
            **class_reports
        }
        
        print(f"Accuracy in this epoch: {overall_metrics['accuracy']}")
        return overall_metrics
    
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=10,
        early_stopping_threshold=0.01
    )
    
    # 初始化DeepSpeed引擎
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)
    
    # 初始化Trainer with callbacks
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
    

    # 启动训练过程
    try:
        print("starting training...")
        trainer.train()
    finally:
        cleanup()

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
