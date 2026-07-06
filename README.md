# mmseg_dino_infer — DINOv3 + Mask2Former 统一推理

极简推理框架, 支持所有 **dinov3 + mask2former (openmmlab/mmdet 训练)** 的网络。
所有网络共享相同的 Mask2Former head; backbone 仅 **时序(temporal)** 与 **单帧(single)** 之分。

## 用法

```bash
# 在 configs.py 注册的网络中选择:
python infer.py --net pastis              # PASTIS 时序 (12帧×10波段, 19类)
python infer.py --net agri                # 农业单帧 RGB (2类)
python infer.py --net pastis --limit 40   # 只跑 40 样本 (快速验证)
python infer.py --net pastis --gpu 1      # 指定 GPU
python infer.py --net pastis --ckpt /path/to/other.pth --force  # 用其他权重
# 或用脚本:
bash infer.sh pastis
bash infer.sh agri --limit 40
```

## 结构

```
mmseg_dino_infer/
├── infer.py              # 统一推理入口
├── infer.sh
├── configs.py            # 网络注册表 (在此添加新网络)
├── convert_checkpoint.py # mmdet 格式 -> 官方 head 格式 key 转换器
├── models/
│   ├── segmentor.py      # backbone + head 组装
│   ├── backbone.py       # 统一 backbone (temporal/single)
│   └── head.py           # 官方 Mask2Former head 子类
└── datasets/
    └── loaders.py        # PASTIS + agri 数据加载
```

## 新增一个网络

在 `configs.py` 的 `NETWORKS` 加一个条目即可, 无需改其他代码:

```python
'my_net': dict(
    backbone='single',          # 'single' 或 'temporal'
    arch='vit_large', patch_size=16,
    in_bands=3,                 # 输入波段数 (≠3 会自动膨胀 patch_embed)
    img_size=512,               # 训练时的输入尺寸
    num_classes=2, ignore_index=255,
    mean=[...], std=[...],      # 训练时的归一化统计量
    ckpt='/path/to/mmdet.pth',  # openmmlab 训练的权重 (mmdet 格式)
    dataset='agri',             # 数据集类型 (需在 loaders.py 注册)
),
```

若 `num_queries` / `dec_layers` 等非标准, 可在条目里覆盖 (默认 100/9)。
新数据集需在 `datasets/loaders.py` 加一个 Dataset 类并注册到 `_DATASETS`。

## 依赖

- `dinov3` 包 (自动从 mmseg_dino_v2 项目加载, 不复制，可以在infer.py中改变加载路径)
- torch + numpy + cv2 + scipy
- 推理前向**不需要**编译 MSDA (使用纯 PyTorch 回退); 仅训练反向才需要
