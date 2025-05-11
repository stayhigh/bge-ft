"""
情感分类评估脚本
"""

import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def load_model(model_path):
    """加载预训练模型和分词器"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

def predict_emotion(text, tokenizer, model):
    """预测文本情感分类"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    return predicted_class, probabilities.tolist()

def main():
    """命令行接口"""
    parser = argparse.ArgumentParser()
    # parser.add_argument("--text", type=str, required=True, help="输入要评估的文本")
    parser.add_argument("--model_path", type=str, default="finetuned_models/finetuned_models_with_emotion_20250511_155747/checkpoint-375", 
                        help="模型路径 (默认: finetuned_models/finetuned_models_with_emotion_20250511_155747/checkpoint-375)")
    args = parser.parse_args()

    # 加载模型
    tokenizer, model = load_model(args.model_path)
    
    # 预测情感
    while True:
        text = input("\n请输入要评估的文本 (输入 'exit' 结束): ")
        if text.lower() == 'exit':
            break                
        if text is not None:
            class_id, probabilities = predict_emotion(text, tokenizer, model)
        
            # 输出结果
            labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        
            # 获取预测标签名称
            predicted_label = labels[class_id]
        
            # 输出结果
            print(f"文本: {text}")
            print(f"预测类别: {class_id} ({predicted_label})")
            print("概率分布:")
            for i, prob in enumerate(probabilities[0]):
                print(f"  类别 {i}: {prob:.4f}")

if __name__ == "__main__":
    main()
