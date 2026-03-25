import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler
from .sampler_ddp import RandomIdentitySampler_DDP
from .ballshow import BallShow
# ========== 新增导入 ==========
from .transforms_extend import MotionBlur, RandomOcclusion  # 导入自定义增强

__factory = {
    'ballshow': BallShow,
}

def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids , _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids,

def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths

def make_dataloader(cfg):
    # ========== 修改train_transforms，插入自定义增强 ==========
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            # ========== 新增：球员场景增强（在ToTensor前，因为是PIL操作） ==========
            RandomOcclusion(prob=0.4),  # 随机遮挡（4060版概率0.4，A5000版可改0.6）
            MotionBlur(prob=0.3),       # 运动模糊（4060版概率0.3，A5000版可改0.5）
            # ========== 原有增强 ==========
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

    train_set = ImageDataset(dataset.train, train_transforms)
    train_set_normal = ImageDataset(dataset.train, val_transforms)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    train_loader_normal = DataLoader(
        train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, train_loader_normal, val_loader, len(dataset.query), num_classes, cam_num, view_num

# 调试代码（可加在make_dataloader.py末尾）
if __name__ == '__main__':
    from yacs.config import CfgNode as CN
    import yaml
    # 加载你的配置文件（替换为实际路径）
    with open("./configs/BallShow/vit_transreid_stride.yml", 'r',encoding='utf-8') as f:
        cfg = CN(yaml.safe_load(f))
    
    train_loader, _, _, _, _, _, _ = make_dataloader(cfg)
    # 取第一个batch查看增强效果
    imgs, pids, camids, viewids = next(iter(train_loader))
    print(f"Image shape: {imgs.shape}")  # 应输出 (batch_size, 3, 256, 128)
    
    # 可视化增强后的图片（反归一化修正：避免数值溢出）
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 反归一化（关键：你的配置中PIXEL_MEAN/STD是[0.5,0.5,0.5]）
    img = imgs[0].permute(1,2,0).cpu().numpy()
    img = img * np.array(cfg.INPUT.PIXEL_STD) + np.array(cfg.INPUT.PIXEL_MEAN)  # 反归一化
    img = np.clip(img, 0, 1)  # 限制范围在0-1（避免数值异常导致图片偏色）
    img = (img * 255).astype(np.uint8)  # 转成0-255的uint8
    
    plt.imshow(img)
    plt.axis('off')  # 关闭坐标轴
    plt.title(f"PID: {pids[0].item()}, CamID: {camids[0].item()}")
    plt.show()
