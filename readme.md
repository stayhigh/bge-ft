# BGE 模型微调 README

## 项目结构
```
├── src/                  # 核心训练代码
│   ├── train_bge_base_with_afqmc.py       # AFQMC数据集训练脚本
│   └── train_bge_base_with_emotion.py     # 情感分类训练脚本
│
├── ds_configs/         # DeepSpeed配置文件
│   ├── ds_config afqmc.json        # AFQMC任务配置
│   └── ds_config_with_emotion.json # 情感分类配置
│
└── scripts/            # 执行脚本
    ├── finetune_with_afqmc.sh      # AFQMC微调脚本
    └── finetune_with_emotion.sh    # 情感分类微调脚本
```

## 环境依赖安装
```bash
# 安装基础依赖
pip install transformers datasets torch deepspeed

# 安装MPI（根据系统需求）
sudo apt-get install libopenmpi-dev
```

## 数据集配置
- **AFQMC数据集**: 默认路径 `data/afqmc/`
- **情感数据集**: 默认路径 `cached_data/dair-ai/emotion/`
- 可通过修改脚本中的 `load_dataset` 参数自定义路径

## DeepSpeed 配置说明
### `ds_config_with_emotion.json` 关键参数：
```json
{
  "bf16": {"enabled": true},          # 启用混合精度训练
  "zero_optimization": {"stage": 1},  # ZeRO优化级别
  "gradient_accumulation_steps": 4    # 梯度累积步数
}
```

## 训练脚本运行方式
```bash
# 单机单卡训练示例
cd $project_dir
bash -x ./scripts/finetune_with_emotion.sh

# 参数说明
--deepspeed_config ds_configs/ds_config_with_emotion.json \n
--num_nodes=1 --num_gpus=1  # 单机单卡训练
```

## 常见问题解答
**Q: 如何修改训练轮数？**  
A: 在 `train_bge_base_with_emotion.py` 中调整 `num_train_epochs` 参数

**Q: 如何使用多GPU训练？**  
A: 修改脚本中的 `--num_gpus` 参数，并确保已安装 NCCL 支持

**Q: 遇到 CUDA 内存不足怎么办？**  
A: 减小 `per_device_train_batch_size` 或增加 `gradient_accumulation_steps`
