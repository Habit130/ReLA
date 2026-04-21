# Linux Server Runbook

本仓库已按单卡 RTX 4090、CUDA 11.8、Python 3.10、Miniconda 目标完成 `plantseg` 适配。默认数据位置固定为仓库同级 `../plantseg`，文本固定使用 `caption[3]`，训练总轮数固定为 50 epoch。

## 目录与产物

- 环境定义：`environment/server-linux-cu118-py310.yml`
- 默认训练配置：`configs/plantseg_swin_base_4090.yaml`
- 资产准备脚本：`tools/prepare_server_assets.py`
- 环境检查脚本：`tools/check_server_environment.py`
- 默认输出目录：`output/plantseg_swin_base_4090`
- 默认测试 mask 导出目录：`output/plantseg_swin_base_4090/test_masks`
- 默认本地 BERT 目录：`models/plantseg/bert-base-uncased`

## 数据约定

- 数据根目录默认是 `../plantseg`
- 注释文件固定为 `../plantseg/main.json`
- 使用 `split` 字段注册 `plantseg_train`、`plantseg_val`、`plantseg_test`
- 使用 `caption[3]` 作为唯一文本描述
- 使用 `mask` 作为唯一监督 mask
- 忽略所有带 `false_healthy_ann` 的样本
- 文本编码器优先读取 `models/plantseg/bert-base-uncased`，找不到时才回退到 `bert-base-uncased`

## 服务器端流程

1. 创建并激活 Conda 环境  
使用 `environment/server-linux-cu118-py310.yml`

2. 准备模型与 CUDA 扩展  
运行 `python tools/prepare_server_assets.py --build-op`

3. 检查环境与数据注册  
运行 `python tools/check_server_environment.py --config-file configs/plantseg_swin_base_4090.yaml`

4. 启动训练  
运行 `python train_net.py --config-file configs/plantseg_swin_base_4090.yaml --num-gpus 1`

5. 使用最佳验证集 IoU checkpoint 做最终测试  
运行 `python train_net.py --config-file configs/plantseg_swin_base_4090.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS output/plantseg_swin_base_4090/model_best.pth DATASETS.TEST '("plantseg_test",)'`

## 输出说明

- 训练期每个 epoch 会在 `plantseg_val` 上评估
- `model_best.pth` 依据验证集 `IoU` 更新
- `plantseg_test_summary.json` 中包含 `IoU`、`Dice`、`Recall`、`mIoU`、`mACC`
- `test_masks` 下按 `main.json` 中 `mask` 的相对路径镜像导出预测 PNG，像素值为 `0/255`
