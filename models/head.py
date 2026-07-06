# 官方 dinov3 Mask2FormerHead 子类: 不改官方源码, 用 openmmlab 一致的超参构造。
import torch.nn as nn
from dinov3.eval.segmentation.models.heads.mask2former_head import Mask2FormerHead
from dinov3.eval.segmentation.models.heads.pixel_decoder import MSDeformAttnPixelDecoder
from dinov3.eval.segmentation.models.heads.mask2former_transformer_decoder import (
    MultiScaleMaskedTransformerDecoder, SelfAttentionLayer, CrossAttentionLayer, FFNLayer,
)
from dinov3.eval.segmentation.models.heads.pixel_decoder import MSDeformAttnTransformerEncoderLayer


class OfficialHead(Mask2FormerHead):
    """官方 head 子类, 用标准 m2f 超参 (256/8, ffn 1024/2048)。"""
    def __init__(self, input_shape, num_classes, hidden_dim=256, nheads=8,
                 pixel_ffn=1024, dec_ffn=2048, enc_layers=6, dec_layers=9, num_queries=100):
        nn.Module.__init__(self)  # 跳过父类硬编码 (nheads=16/ffn=4096)
        orig = input_shape
        input_shape = sorted(input_shape.items(), key=lambda x: x[1][-1])
        self.in_features = [k for k, _ in input_shape]
        self.common_stride = 4
        self.transformer_in_feature = "multi_scale_pixel_decoder"
        self.num_classes = num_classes
        self.pixel_decoder = MSDeformAttnPixelDecoder(
            input_shape=orig, transformer_dropout=0.0, transformer_nheads=nheads,
            transformer_dim_feedforward=pixel_ffn, transformer_enc_layers=enc_layers,
            conv_dim=hidden_dim, mask_dim=hidden_dim, norm="GN",
            transformer_in_features=["1", "2", "3", "4"], common_stride=4)
        self.predictor = MultiScaleMaskedTransformerDecoder(
            in_channels=hidden_dim, mask_classification=True, num_classes=num_classes,
            hidden_dim=hidden_dim, num_queries=num_queries, nheads=nheads,
            dim_feedforward=dec_ffn, dec_layers=dec_layers, pre_norm=False,
            mask_dim=hidden_dim, enforce_input_project=False)

    def layers(self, features, mask=None):
        mf, _, msf = self.pixel_decoder.forward_features(features)
        return self.predictor(msf, mf, mask)
