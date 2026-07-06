# 统一 DINOv3 backbone: 支持 时序(temporal) 和 单帧(single) 两种模式。
# 两者共用 DINOv3_Adapter (deform_ratio=1.0 与 openmmlab 训练一致); temporal 额外加 tpool。
import torch
import torch.nn as nn
from omegaconf import OmegaConf


_BAND_OF_RGB = [2, 1, 0]  # 遥感10波段中 R,G,B 对应的波段索引

def _inflate_conv(conv, target_in):
    """3 通道 conv -> target_in 通道: RGB 权重均值复制 + 精准放入 RGB 波段。"""
    new = nn.Conv2d(target_in, conv.out_channels, kernel_size=conv.kernel_size,
                    stride=conv.stride, padding=conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        w = conv.weight
        new.weight.copy_(w.mean(dim=1, keepdim=True).repeat(1, target_in, 1, 1) * (3.0 / target_in))
        for rgb_idx, band_idx in enumerate(_BAND_OF_RGB):
            new.weight[:, band_idx] = w[:, rgb_idx]
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


class TemporalAttnPool(nn.Module):
    """时序注意力池化: T 帧特征 -> 1 帧 (与 v4 训练时一致)。"""
    def __init__(self, dim, n_frames, n_heads=8):
        super().__init__()
        self.month_embed = nn.Parameter(torch.zeros(n_frames, dim))
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.month_embed, std=0.02)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, x):
        B, T, D, H, W = x.shape
        feat = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, D)
        feat = self.norm(feat + self.month_embed)
        q = self.query.expand(B * H * W, -1, -1)
        out, _ = self.attn(q, feat, feat)
        return out.reshape(B, H, W, D).permute(0, 3, 1, 2).contiguous()


class DINOv3Backbone(nn.Module):
    """统一 backbone。

    forward:
      temporal: x=[B, T*C, H, W] -> 4 features
      single:   x=[B, C, H, W]   -> 4 features
    """
    _DEFAULT_INTER = {"vit_large": [5, 11, 17, 23], "vit_base": [2, 5, 8, 11]}

    def __init__(self, cfg, dino_ckpt):
        super().__init__()
        from dinov3.models import build_model_for_eval
        from dinov3.eval.segmentation.models.backbone.dinov3_adapter import DINOv3_Adapter

        self.temporal = (cfg['backbone'] == 'temporal')
        self.in_bands = cfg['in_bands']
        self.n_frames = cfg.get('n_frames', 1)

        vit_cfg = OmegaConf.create({
            "student": dict(arch=cfg['arch'], patch_size=cfg['patch_size'], drop_path_rate=0.0,
                pos_embed_rope_base=None, pos_embed_rope_min_period=4, pos_embed_rope_max_period=50,
                pos_embed_rope_normalize_coords="separate", pos_embed_rope_shift_coords=None,
                pos_embed_rope_jitter_coords=None, pos_embed_rope_rescale_coords=None,
                qkv_bias=True, layerscale=1e-5, norm_layer="layernorm", ffn_layer="mlp",
                ffn_bias=True, proj_bias=True, n_storage_tokens=0, mask_k_bias=False,
                untie_cls_and_patch_norms=False, untie_global_and_local_cls_norm=False, fp8_enabled=False),
            "crops": {"global_crops_size": 224}})
        vit = build_model_for_eval(vit_cfg, pretrained_weights=dino_ckpt)

        self.adapter = DINOv3_Adapter(
            vit, interaction_indexes=self._DEFAULT_INTER.get(cfg['arch'], [5, 11, 17, 23]),
            with_cp=False, deform_ratio=1.0)

        if cfg['in_bands'] != 3:
            bb = self.adapter.backbone
            bb.patch_embed.proj = _inflate_conv(bb.patch_embed.proj, cfg['in_bands'])
            bb.patch_embed.in_chans = cfg['in_bands']
            self.adapter.spm.stem[0] = _inflate_conv(self.adapter.spm.stem[0], cfg['in_bands'])

        self.embed_dim = vit.embed_dim
        if self.temporal:
            self.tpool = nn.ModuleDict()
            for key in ("2", "3", "4"):
                self.tpool[key] = TemporalAttnPool(self.embed_dim, self.n_frames)

    def _pool(self, f, key):
        return self.tpool[key](f) if (self.temporal and key in self.tpool) else f.mean(1)

    def forward(self, x):
        B = x.shape[0]
        if self.temporal:
            out = self.adapter(x.view(B * self.n_frames, self.in_bands, *x.shape[2:]))
        else:
            out = self.adapter(x)
        feats = []
        for key in ("1", "2", "3", "4"):
            f = out[key]
            if self.temporal:
                f = f.view(B, self.n_frames, *f.shape[1:])
                f = self._pool(f, key)
            feats.append(f)
        return tuple(feats)
