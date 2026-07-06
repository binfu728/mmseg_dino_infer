# 统一 segmentor: DINOv3 backbone + 官方 Mask2Former head。
import torch.nn as nn
from .backbone import DINOv3Backbone
from .head import OfficialHead


class Segmentor(nn.Module):
    def __init__(self, net_cfg, dino_ckpt, head_defaults):
        super().__init__()
        # head 参数: 默认值, 允许 net_cfg 覆盖 (如 num_queries)
        head_cfg = {**head_defaults, **{k: net_cfg[k] for k in head_defaults if k in net_cfg}}
        self.backbone = DINOv3Backbone(net_cfg, dino_ckpt)
        embed = self.backbone.embed_dim
        s = net_cfg['img_size']
        input_shape = {
            "1": [embed, s // 4, s // 4, 4],
            "2": [embed, s // 8, s // 8, 8],
            "3": [embed, s // 16, s // 16, 16],
            "4": [embed, s // 32, s // 32, 32],
        }
        self.head = OfficialHead(
            input_shape=input_shape, num_classes=net_cfg['num_classes'], **head_cfg)

    def forward(self, x):
        feats = self.backbone(x)
        features = {"1": feats[0], "2": feats[1], "3": feats[2], "4": feats[3]}
        return self.head.layers(features)
