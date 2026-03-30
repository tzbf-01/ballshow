import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
import copy
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
import torch.nn.functional as F

def shuffle_unit(features, shift, group, begin=1):
    # ... (保持不变，但 OAG 不再使用 shuffle_unit)
    batchsize = features.size(0)
    dim = features.size(-1)
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)
    return x

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class Backbone(nn.Module):
    # ... (保持不变，略)
    def __init__(self, num_classes, cfg):
        # ... 原代码不变
        pass
    def forward(self, x, label=None):
        # ... 原代码不变
        pass
    def load_param(self, trained_path):
        # ... 原代码不变
        pass
    def load_param_finetune(self, model_path):
        # ... 原代码不变
        pass

class build_transformer(nn.Module):
    # ... 保持不变（无 JPM 的 baseline）
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        # ... 原代码不变
        pass
    def forward(self, x, label=None, cam_label=None, view_label=None):
        # ... 原代码不变
        pass
    def load_param(self, trained_path):
        # ... 原代码不变
        pass
    def load_param_finetune(self, model_path):
        # ... 原代码不变
        pass

# ========== 新增 OAG 模块 ==========
class OcclusionAwareGrouping(nn.Module):
    """
    遮挡感知分组模块 (Occlusion-Aware Grouping)
    输入: Transformer 输出的 token 序列 (B, N, D), 其中 N = num_patches+1 (包含 cls token)
    输出: 分组后的特征列表 [global_feat, group_feat_1, ..., group_feat_G]
    """
    def __init__(self, num_groups=17, num_occluded=10, temp=0.1):
        super().__init__()
        self.num_groups = num_groups          # G, 通常设为 17 (对应人体关键点)
        self.num_occluded = num_occluded      # 选择多少个 patch 作为遮挡特征
        self.temp = temp                      # 温度参数用于 softmax

        # 可学习的组权重矩阵 (D x G)
        self.group_weights = nn.Parameter(torch.Tensor(768, num_groups))
        nn.init.xavier_uniform_(self.group_weights)

        # 每个组的可学习缩放因子 gamma (用于自适应组大小)
        self.gamma = nn.Parameter(torch.ones(num_groups))

    def forward(self, x):
        """
        x: (B, N, D)  token 序列, 其中 x[:,0,:] 为 cls token
        """
        B, N, D = x.shape
        cls_token = x[:, 0:1, :]          # (B,1,D)
        patch_tokens = x[:, 1:, :]        # (B, N-1, D)

        # 1. 计算每个 patch token 与 cls token 的余弦相似度
        cls_norm = F.normalize(cls_token, dim=-1)           # (B,1,D)
        patch_norm = F.normalize(patch_tokens, dim=-1)      # (B,N-1,D)
        sim = torch.bmm(patch_norm, cls_norm.transpose(1,2)).squeeze(-1)  # (B, N-1)

        # 2. 选择相似度最低的 k 个 patch 作为遮挡特征
        _, occluded_indices = torch.topk(sim, k=self.num_occluded, dim=1, largest=False)
        # 创建掩膜: 1 表示非遮挡特征, 0 表示遮挡特征
        mask = torch.ones_like(sim)
        mask.scatter_(1, occluded_indices, 0)   # (B, N-1)

        # 3. 非遮挡特征
        non_occluded_tokens = patch_tokens * mask.unsqueeze(-1)  # 遮挡位置置零
        # 去除零向量（但保留形状，后面通过加权求和忽略零贡献）

        # 4. 计算每个非遮挡特征与组权重的相关性得分
        # 组权重: (D, G) -> (B, D, G)
        group_weights = self.group_weights.unsqueeze(0).expand(B, -1, -1)  # (B, D, G)
        # 非遮挡特征: (B, N-1, D) -> (B, D, N-1)
        non_occluded_tokens_T = non_occluded_tokens.transpose(1,2)         # (B, D, N-1)
        # 相关性得分: (B, G, N-1)
        scores = torch.bmm(group_weights.transpose(1,2), non_occluded_tokens_T) / self.temp
        # 对每个 patch 在组间做 softmax
        attn = F.softmax(scores, dim=1)          # (B, G, N-1)

        # 5. 自适应组大小: 根据 gamma 和 attn 计算每组应该聚合的特征数（这里简化为加权求和）
        # 组特征 = 所有非遮挡特征的加权和（权重为 attn）
        group_features = torch.bmm(attn, non_occluded_tokens)   # (B, G, D)

        # 6. 可选：添加正交损失和多样性损失（在外部计算，此处不实现）
        # 返回全局特征和组特征列表
        global_feat = cls_token.squeeze(1)   # (B, D)
        group_feat_list = [group_features[:, i, :] for i in range(self.num_groups)]
        return [global_feat] + group_feat_list, attn

# ========== 修改后的 build_transformer_local（集成 OAG，替换 JPM）==========
class build_transformer_local(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
        super(build_transformer_local, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        # 注意：这里仍然使用 local_feature=True 以获取完整序列（未 norm）
        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num,
                                                        stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        # 不再使用原来的 b1, b2, 改用 OAG
        self.oag = OcclusionAwareGrouping(num_groups=17, num_occluded=10, temp=0.1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

        # 动态创建分类器和 BNNeck: 1 个全局 + G 个分组
        G = self.oag.num_groups
        self.num_features = G + 1   # 全局 + 分组

        # 创建 BNNeck 列表
        self.bottleneck = nn.ModuleList()
        self.bottleneck.append(nn.BatchNorm1d(self.in_planes))
        for _ in range(G):
            self.bottleneck.append(nn.BatchNorm1d(self.in_planes))
        for bn in self.bottleneck:
            bn.bias.requires_grad_(False)
            bn.apply(weights_init_kaiming)

        # 创建分类器列表
        self.classifier = nn.ModuleList()
        for _ in range(self.num_features):
            if self.ID_LOSS_TYPE == 'arcface':
                cls = Arcface(self.in_planes, self.num_classes,
                              s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
            elif self.ID_LOSS_TYPE == 'cosface':
                cls = Cosface(self.in_planes, self.num_classes,
                              s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
            elif self.ID_LOSS_TYPE == 'amsoftmax':
                cls = AMSoftmax(self.in_planes, self.num_classes,
                                s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
            elif self.ID_LOSS_TYPE == 'circle':
                cls = CircleLoss(self.in_planes, self.num_classes,
                                 s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
            else:
                cls = nn.Linear(self.in_planes, self.num_classes, bias=False)
                cls.apply(weights_init_classifier)
            self.classifier.append(cls)

    def forward(self, x, label=None, cam_label=None, view_label=None):
        # 获取 Transformer 输出序列 (B, N, D)，其中 N = num_patches+1，且未经过 norm
        features = self.base(x, cam_label=cam_label, view_label=view_label)

        # 应用 OAG 得到特征列表 [global_feat, group_feat_1, ..., group_feat_G]
        feat_list, attn = self.oag(features)   # 接收 attn   # 每个元素形状 (B, D)

        # 存储 attn 供损失函数使用
        self.attn = attn

        # 对每个特征应用 BNNeck
        feat_bn_list = []
        for i, feat in enumerate(feat_list):
            feat_bn_list.append(self.bottleneck[i](feat))

        if self.training:
            # 计算分类分数
            cls_scores = []
            for i, feat_bn in enumerate(feat_bn_list):
                if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                    cls_score = self.classifier[i](feat_bn, label)
                else:
                    cls_score = self.classifier[i](feat_bn)
                cls_scores.append(cls_score)
            # 返回所有分类分数和原始特征（未经过 BN，用于 triplet loss）
            return cls_scores, feat_list
        else:
            # 测试时：拼接所有特征（经过 BN 或原始，根据 neck_feat）
            if self.neck_feat == 'after':
                out_feat = torch.cat(feat_bn_list, dim=1)
            else:
                out_feat = torch.cat(feat_list, dim=1)
            return out_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

# 原有的 __factory_T_type 和 make_model 保持不变
__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}

def make_model(cfg, num_class, camera_num, view_num):
    if cfg.MODEL.NAME == 'transformer':
        if cfg.MODEL.JPM:
            # 注意：现在 JPM 被 OAG 替代，但为了兼容，我们仍使用 build_transformer_local
            model = build_transformer_local(num_class, camera_num, view_num, cfg, __factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE)
            print('===========building transformer with OAG (replacing JPM) ===========')
        else:
            model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
            print('===========building transformer===========')
    else:
        model = Backbone(num_class, cfg)
        print('===========building ResNet===========')
    return model