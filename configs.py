# 统一推理的"网络注册表": 每个 dinov3+m2f (openmmlab 训练) 网络一个条目。
# 新增网络只需在此添加一个 dict, 无需改其他代码。

# DINOv3 预训练 ViT 权重 (所有网络共用, 仅用于构建 backbone 结构, 推理时被各 ckpt 覆盖)
DINO_CKPT = '/mnt/ht2_nas2/00-model/00-fb/mmseg_data/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'

# 各网络共享的 Mask2Former head 架构参数 (均继承自 mask2former_r50_ade20k base config)
HEAD = dict(hidden_dim=256, nheads=8, pixel_ffn=1024, dec_ffn=2048,
            enc_layers=6, dec_layers=9, num_queries=100)

NETWORKS = {
    # PASTIS 时序 (12帧x10波段, 语义分割 19 类)
    'pastis': dict(
        backbone='temporal',
        arch='vit_large', patch_size=16,
        in_bands=10, n_frames=12, img_size=256,
        num_classes=19, ignore_index=255,
        mean=[1180.2,1387.7,1436.7,1773.7,2735.8,3080.1,3223.6,3338.3,2418.1,1630.2],
        std=[1976.7,1916.8,1996.2,1903.1,1784.9,1796.3,1811.8,1793.3,1474.4,1309.8],
        ckpt='/mnt/ht2_nas2/00-model/00-fb/mmseg_dino/work_dirs/dinov3l_m2f_pastis_temporal_v4/best_mIoU_iter_16000.pth',
        dataset='pastis',
        data_root='/mnt/ht2_nas2/00-model/00-fb/mmseg_data/PASTIS-R',
    ),
    # 农业 (单帧 RGB, 二分类: 背景/农田)
    'agri': dict(
        backbone='single',
        arch='vit_large', patch_size=16,
        in_bands=3, img_size=512,
        num_classes=2, ignore_index=255,
        num_queries=50,
        mean=[72.4085, 89.7399, 69.6123],
        std=[32.8544, 23.9954, 23.1234],
        ckpt='/mnt/qh2-nas3/00-model/00-fb/mmseg_dino_agri/work_dirs/dinov3l_m2f_agri/best_mIoU_iter_60000.pth',
        dataset='agri',
        data_root='/mnt/ht2_nas2/00-model/00-jiangzf/label20000/Segmentation',
    ),
}
