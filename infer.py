#!/usr/bin/env python
# 统一推理: 加载任意 dinov3+m2f (openmmlab 训练) 网络, 在 val 集上计算 mIoU。
#
# 用法:
#   python infer.py --net pastis              # PASTIS 时序网络
#   python infer.py --net agri                # 农业单帧网络
#   python infer.py --net pastis --gpu 1
#   python infer.py --net pastis --ckpt /path/to/other.pth   # 指定其他权重
#
# 原理: 所有网络共享相同 Mask2Former head; backbone 仅 时序/单帧 之分。
#       mmdet 格式 ckpt 经 convert_checkpoint 转换后 strict 加载到统一模型。
import os, sys, argparse
# dinov3 依赖: 优先用 v2 项目里的 dinov3 (已验证可用)
for _p in ['/mnt/qh2-nas3/00-model/00-fb/mmseg_dino_v2',
           '/mnt/ht2_nas2/00-model/00-fb/mmseg_dino']:
    if _p not in sys.path:
        sys.path.append(_p)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs import NETWORKS, HEAD, DINO_CKPT
from models.segmentor import Segmentor
from datasets.loaders import build_dataset
from convert_checkpoint import convert


class IoUMetric:
    def __init__(self, n):
        self.n, self.mat = n, None
    @torch.no_grad()
    def update(self, pred, target):
        m = (target >= 0) & (target < self.n)
        ind = self.n * target[m].to(torch.int64) + pred[m]
        bins = torch.bincount(ind, minlength=self.n ** 2).reshape(self.n, self.n)
        self.mat = bins if self.mat is None else self.mat + bins
    def compute(self):
        if self.mat is None: return 0.0, 0.0
        h = self.mat.float()
        iou = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h) + 1e-10)
        return iou.nanmean().item() * 100, (torch.diag(h).sum() / (h.sum() + 1e-10)).item() * 100


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--net', required=True, choices=list(NETWORKS.keys()),
                    help='网络名称 (在 configs.py 中定义)')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--ckpt', default=None, help='覆盖默认权重路径')
    ap.add_argument('--split', default=None, help='数据 split (默认 val/valid)')
    ap.add_argument('--limit', type=int, default=0, help='只跑前 N 个样本 (0=全部)')
    ap.add_argument('--force', action='store_true', help='强制重新转换 ckpt')
    a = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    dev = torch.device('cuda:0')
    nc = NETWORKS[a.net]
    ckpt = a.ckpt or nc['ckpt']
    split = a.split or ('val' if nc['dataset'] == 'pastis' else 'valid')

    print(f"=== 推理: {a.net} | ckpt={ckpt} | split={split} ===")

    # 1. 建模型
    model = Segmentor(nc, DINO_CKPT, HEAD).to(dev)

    # 2. 转换 + 加载 (mmdet -> 官方 head 格式)
    conv = os.path.splitext(ckpt)[0] + '_converted.pth'
    if a.force or not os.path.exists(conv):
        print(f"[*] 转换: {ckpt} -> {conv}")
        convert(ckpt, conv)
    sd = torch.load(conv, map_location='cpu', weights_only=False)
    r = model.load_state_dict(sd, strict=True)
    print(f"[*] load_state_dict: {r}")
    model.eval()

    # 3. 数据 + mIoU
    ds = build_dataset(nc, split)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=4)
    if a.limit > 0:
        from torch.utils.data import Subset
        loader = DataLoader(Subset(ds, range(min(a.limit, len(ds)))),
                            batch_size=4, shuffle=False, num_workers=4)
    metric = IoUMetric(nc['num_classes'])
    for i, (img, gt) in enumerate(loader):
        out = model(img.to(dev))
        msk = F.interpolate(out['pred_masks'], size=gt.shape[-2:], mode='bilinear', align_corners=False)
        seg = torch.einsum('bqc,bqhw->bchw', F.softmax(out['pred_logits'], -1)[..., :-1],
                           msk.sigmoid()).argmax(1)
        for b in range(seg.shape[0]):
            metric.update(seg[b].flatten(), gt[b].to(dev).flatten())
        if (i + 1) % 20 == 0:
            m, _ = metric.compute(); print(f"  batch {i+1}/{len(loader)}  mIoU={m:.2f}")
    miou, acc = metric.compute()
    print(f"\n==== [{a.net}] mIoU={miou:.2f} | aAcc={acc:.2f} ====")


if __name__ == '__main__':
    main()
