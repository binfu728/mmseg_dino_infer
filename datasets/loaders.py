# 极简数据加载: 每个数据集一个 Dataset 类, 复现训练时的归一化/resize/标签映射。
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class PASTISDataset(Dataset):
    """PASTIS 时序: Sentinel-2 (12帧x10波段), val=fold4。返回 (img[T*C,H,W], gt[H,W])。"""
    def __init__(self, cfg, split='val'):
        self.img_size = cfg['img_size']
        self.mean = np.array(cfg['mean'], np.float32).reshape(1, -1, 1, 1)
        self.std = np.array(cfg['std'], np.float32).reshape(1, -1, 1, 1)
        self.num_classes = cfg['num_classes']
        root = cfg['data_root']
        fold = 4 if split == 'val' else 1
        g = json.load(open(f"{root}/metadata.geojson"))
        self.pids = [f['properties']['ID_PATCH'] for f in g['features']
                     if f['properties']['Fold'] == fold]
        self.s2m = f"{root}/DATA_S2_M12"
        self.ann = f"{root}/ANNOTATIONS"

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, i):
        pid = self.pids[i]
        fr = np.load(f"{self.s2m}/S2M_{pid}.npy").astype(np.float32)   # (12,10,128,128)
        norm = (fr - self.mean) / self.std
        T, C, H, W = norm.shape
        img = np.stack([cv2.resize(norm.reshape(T * C, H, W)[j],
                      (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                        for j in range(T * C)])
        gt = np.load(f"{self.ann}/TARGET_{pid}.npy")[0]
        gt = cv2.resize(gt, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        gt = gt.astype(np.int64)
        gt[gt >= self.num_classes] = 255
        return torch.from_numpy(np.ascontiguousarray(img)).float(), torch.from_numpy(gt)


class AgriDataset(Dataset):
    """农业单帧 RGB, 二分类。返回 (img[3,H,W], gt[H,W])。标签映射 255->1 (复现训练)。"""
    def __init__(self, cfg, split='valid'):
        self.img_size = cfg['img_size']
        self.mean = np.array(cfg['mean'], np.float32)
        self.std = np.array(cfg['std'], np.float32)
        self.num_classes = cfg['num_classes']
        from pathlib import Path
        root = Path(cfg['data_root'])
        names = [l.strip() for l in open(root / f"{split}.txt") if l.strip()]
        self.imgs = [root / "img_dir" / f"{n}.png" for n in names]
        self.anns = [root / "ann_dir" / f"{n}_mask_seg.png" for n in names]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = cv2.cvtColor(cv2.imread(str(self.imgs[i]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        gt = cv2.imread(str(self.anns[i]), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        gt = cv2.resize(gt, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        img = (img.astype(np.float32) - self.mean) / self.std
        gt = gt.astype(np.int64)
        gt[gt == 255] = 1    # 复现训练时的标签映射
        gt[gt >= self.num_classes] = 255
        return torch.from_numpy(np.ascontiguousarray(img).transpose(2, 0, 1)).float(), \
               torch.from_numpy(gt)


_DATASETS = {'pastis': PASTISDataset, 'agri': AgriDataset}

def build_dataset(net_cfg, split):
    return _DATASETS[net_cfg['dataset']](net_cfg, split)
