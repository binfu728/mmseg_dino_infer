# 文件路径: convert_checkpoint.py
# 将 openmmlab (mmseg) 训练得到的 mmdet 格式权重 转换为 本项目官方 head 的 state_dict 格式。
# 用法:
#   python convert_checkpoint.py --src <mmseg.pth> --dst <converted.pth>

import argparse
import torch


def remap_ffn_suffix(rest_after_ffn):
    """rest_after_ffn 形如 'layers.0.0.weight' / 'layers.1.bias' -> 'linear1.weight' / 'linear2.bias'"""
    if rest_after_ffn.startswith("layers.0.0."):
        return "linear1." + rest_after_ffn[len("layers.0.0."):]
    if rest_after_ffn.startswith("layers.1."):
        return "linear2." + rest_after_ffn[len("layers.1."):]
    return rest_after_ffn


def convert_head_key(b):
    """b: 去掉 'decode_head.' 前缀后的 mmdet 内部键 -> 官方 head 键。"""
    # ---- pixel decoder ----
    if b.startswith("pixel_decoder.encoder.layers."):
        inner = b[len("pixel_decoder.encoder.layers."):]  # '{i}.X'
        head, tail = inner.split(".", 1)  # head='{i}', tail='X'
        if tail.startswith("self_attn."):
            new_tail = tail  # self_attn 子模块名一致
        elif tail.startswith("ffn.layers.0.0."):
            new_tail = "linear1." + tail[len("ffn.layers.0.0."):]
        elif tail.startswith("ffn.layers.1."):
            new_tail = "linear2." + tail[len("ffn.layers.1."):]
        elif tail.startswith("norms."):
            ni = tail[len("norms.")].split(".")[0]
            attr = tail[tail.index(".", len("norms.")) + 1:]
            new_tail = f"norm{int(ni) + 1}.{attr}"
        else:
            new_tail = tail
        return f"head.pixel_decoder.encoder.encoder.layers.{head}.{new_tail}"

    if b == "pixel_decoder.level_encoding.weight":
        return "head.pixel_decoder.encoder.level_encoding"

    if b.startswith("pixel_decoder.input_convs."):
        return ("head." + b).replace(".conv.", ".0.").replace(".gn.", ".1.")

    if b.startswith("pixel_decoder.lateral_convs.") or b.startswith("pixel_decoder.output_convs."):
        return ("head." + b).replace(".conv.weight", ".weight").replace(".gn.", ".norm.")

    if b.startswith("pixel_decoder.mask_feature."):
        return "head." + b

    # ---- transformer decoder (predictor): layers / post_norm 在 transformer_decoder 下 ----
    if b.startswith("transformer_decoder.layers."):
        inner = b[len("transformer_decoder.layers."):]  # '{i}.X'
        head, tail = inner.split(".", 1)
        if tail.startswith("self_attn.attn."):
            return f"head.predictor.transformer_self_attention_layers.{head}.self_attn." + tail[len("self_attn.attn."):]
        if tail.startswith("cross_attn.attn."):
            return f"head.predictor.transformer_cross_attention_layers.{head}.multihead_attn." + tail[len("cross_attn.attn."):]
        if tail.startswith("ffn.layers.0.0."):
            return f"head.predictor.transformer_ffn_layers.{head}.linear1." + tail[len("ffn.layers.0.0."):]
        if tail.startswith("ffn.layers.1."):
            return f"head.predictor.transformer_ffn_layers.{head}.linear2." + tail[len("ffn.layers.1."):]
        if tail.startswith("norms."):
            ni = tail[len("norms.")].split(".")[0]
            attr = tail[tail.index(".", len("norms.")) + 1:]
            target = {0: "transformer_cross_attention_layers", 1: "transformer_self_attention_layers", 2: "transformer_ffn_layers"}[int(ni)]
            return f"head.predictor.{target}.{head}.norm.{attr}"

    if b == "transformer_decoder.post_norm.weight":
        return "head.predictor.post_norm.weight"
    if b == "transformer_decoder.post_norm.bias":
        return "head.predictor.post_norm.bias"
    if b == "transformer_decoder.query_embed.weight":
        return "head.predictor.query_embed.weight"
    if b == "transformer_decoder.query_feat.weight":
        return "head.predictor.query_feat.weight"
    if b == "transformer_decoder.level_embed.weight":
        return "head.predictor.level_embed.weight"
    if b.startswith("transformer_decoder.cls_embed."):
        return "head.predictor.class_embed." + b[len("transformer_decoder.cls_embed."):]
    if b.startswith("transformer_decoder.mask_embed."):
        return "head.predictor.mask_embed.layers." + b[len("transformer_decoder.mask_embed."):]

    # ---- head 直接挂载的标量/嵌入参数 (mmseg 中不在 transformer_decoder 下) ----
    if b.startswith("cls_embed."):
        return "head.predictor.class_embed." + b[len("cls_embed."):]
    if b.startswith("mask_embed."):
        # mmseg: mask_embed.{0,2,4}.X (Sequential, ReLU 在奇数位) -> official MLP layers.{0,1,2}.X
        idx = int(b.split(".")[1])
        attr = b[b.index(".", len("mask_embed.")) + 1:]
        return f"head.predictor.mask_embed.layers.{idx // 2}.{attr}"
    if b == "query_embed.weight":
        return "head.predictor.query_embed.weight"
    if b == "query_feat.weight":
        return "head.predictor.query_feat.weight"
    if b == "level_embed.weight":
        return "head.predictor.level_embed.weight"

    return None


def convert_key(k):
    """mmseg 全键 -> 本项目模型键。"""
    if k.startswith("backbone."):
        # 唯一差异: adapter 内 mmcv 包了一层 .attn (extractor.attn.attn.* -> extractor.attn.*)
        return k.replace(".attn.attn.", ".attn.")
    if k.startswith("decode_head."):
        return convert_head_key(k[len("decode_head."):])
    return None


def convert(src_path, dst_path):
    ck = torch.load(src_path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck

    new_sd = {}
    skipped = []
    for k, v in sd.items():
        nk = convert_key(k)
        if nk is None:
            skipped.append(k)
            continue
        new_sd[nk] = v
    torch.save(new_sd, dst_path)
    print(f"converted {len(new_sd)} tensors -> {dst_path}")
    if skipped:
        print(f"skipped {len(skipped)} non-param keys, e.g. {skipped[:3]}")
    return new_sd


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    convert(a.src, a.dst)
