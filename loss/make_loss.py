# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
Modified to include orthogonality loss and diversity loss for OAG module.
"""

import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss


def orthogonality_loss(group_weights):
    """
    正交损失：促使不同组的权重向量相互正交
    group_weights: (D, G)
    """
    # 按列归一化，防止数值不稳定
    W = F.normalize(group_weights, dim=0)
    # W^T * W
    WtW = W.T @ W
    I = torch.eye(WtW.shape[0], device=WtW.device)
    loss = torch.norm(WtW - I, p='fro') ** 2
    return loss


def diversity_loss(attn):
    """
    多样性损失：鼓励不同特征组的注意力分布多样化
    attn: (B, G, N_patches)  组分配概率（每行和为1）
    """
    B, G, N = attn.shape
    eps = 1e-8
    # 对每个样本、每个组计算熵 H = -sum(p * log(p))
    entropy = -torch.sum(attn * torch.log(attn + eps), dim=2)   # (B, G)
    max_entropy = torch.log(torch.tensor(N, dtype=torch.float, device=attn.device))
    # 平均归一化熵
    normalized_entropy = entropy.mean() / max_entropy
    loss = 1 - normalized_entropy
    return loss


def make_loss(cfg, num_classes, model=None):
    """
    构建损失函数，如果 model 包含 OAG 模块，则自动添加正交损失和多样性损失。
    """
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)

    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet, but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)
    else:
        xent = None

    # 辅助损失权重（论文中设为1.0，可调）
    lambda_ortho = 1.0
    lambda_div = 1.0

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif sampler == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam):
            # ---------- 计算 ID loss ----------
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                if isinstance(score, list):
                    ID_LOSS = [xent(scor, target) for scor in score[1:]]
                    ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                    ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
                else:
                    ID_LOSS = xent(score, target)
            else:
                if isinstance(score, list):
                    ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                    ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                    ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                else:
                    ID_LOSS = F.cross_entropy(score, target)

            # ---------- 计算 Triplet loss ----------
            if isinstance(feat, list):
                TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
            else:
                TRI_LOSS = triplet(feat, target)[0]

            base_loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

            # ---------- 辅助损失（如果模型包含 OAG 模块）----------
            aux_loss = 0.0
            if model is not None and hasattr(model, 'oag') and hasattr(model, 'attn'):
                # 正交损失
                group_weights = model.oag.group_weights
                if group_weights is not None:
                    aux_loss += lambda_ortho * orthogonality_loss(group_weights)
                # 多样性损失
                attn = model.attn
                if attn is not None:
                    aux_loss += lambda_div * diversity_loss(attn)

            total_loss = base_loss + aux_loss
            return total_loss

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center, but got {}'.format(sampler))

    return loss_func, center_criterion