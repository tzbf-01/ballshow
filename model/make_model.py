import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
import copy
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
import torch.nn.functional as F

def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
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
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
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

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate= cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None, cam_label= None, view_label=None):
        global_feat = self.base(x, cam_label=cam_label, view_label=view_label)

        feat = self.bottleneck(global_feat)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)

            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

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

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_1.apply(weights_init_classifier)
            self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_2.apply(weights_init_classifier)
            self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_3.apply(weights_init_classifier)
            self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_4.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)

        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = cfg.MODEL.SHIFT_NUM
        print('using shift_num size:{}'.format(self.shift_num))
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH
        print('using divide_length size:{}'.format(self.divide_length))
        self.rearrange = rearrange

    def forward(self, x, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'

        features = self.base(x, cam_label=cam_label, view_label=view_label)

        # global branch
        b1_feat = self.b1(features) # [64, 129, 768]
        global_feat = b1_feat[:, 0]

        # JPM branch
        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        token = features[:, 0:1]

        if self.rearrange:
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
        else:
            x = features[:, 1:]
        # lf_1
        b1_local_feat = x[:, :patch_length]
        b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
        local_feat_1 = b1_local_feat[:, 0]

        # lf_2
        b2_local_feat = x[:, patch_length:patch_length*2]
        b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        local_feat_2 = b2_local_feat[:, 0]

        # lf_3
        b3_local_feat = x[:, patch_length*2:patch_length*3]
        b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        local_feat_3 = b3_local_feat[:, 0]

        # lf_4
        b4_local_feat = x[:, patch_length*3:patch_length*4]
        b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        local_feat_4 = b4_local_feat[:, 0]

        feat = self.bottleneck(global_feat)

        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
                cls_score_1 = self.classifier_1(local_feat_1_bn)
                cls_score_2 = self.classifier_2(local_feat_2_bn)
                cls_score_3 = self.classifier_3(local_feat_3_bn)
                cls_score_4 = self.classifier_4(local_feat_4_bn)
            return [cls_score, cls_score_1, cls_score_2, cls_score_3,
                        cls_score_4
                        ], [global_feat, local_feat_1, local_feat_2, local_feat_3,
                            local_feat_4]  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1)
            else:
                return torch.cat(
                    [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1)

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


_factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}

class CrossDimensionalMultiScaleFusion(nn.Module):
    def __init__(self, in_channels_trans, in_channels_cnn, out_channels_trans, out_channels_cnn, reduction=16):
        super().__init__()
        self.in_channels_trans = in_channels_trans
        self.in_channels_cnn = in_channels_cnn
        self.out_channels_trans = out_channels_trans
        self.out_channels_cnn = out_channels_cnn
        
        # 投影层保持输入通道数不变（可选，若需要改变通道可在此调整）
        self.proj_trans = nn.Conv2d(in_channels_trans, in_channels_trans, 1)
        self.proj_cnn = nn.Conv2d(in_channels_cnn, in_channels_cnn, 1)
        
        # 融合后的总通道数
        fusion_channels = in_channels_trans + in_channels_cnn
        self.conv_after_pool = nn.Conv2d(fusion_channels, fusion_channels, 3, padding=1)
        
        # 明确记录切分点，为 Transformer 分支的通道数
        self.split_idx = in_channels_trans

    def forward(self, trans_feat, cnn_feat):
        B, _, H, W = trans_feat.shape
        cnn_resized = F.interpolate(cnn_feat, size=(H, W), mode='bilinear', align_corners=False)
        
        trans_proj = self.proj_trans(trans_feat)
        cnn_proj = self.proj_cnn(cnn_resized)
        
        X = torch.cat([trans_proj, cnn_proj], dim=1)  # (B, D+C, H, W)
        C_total = X.shape[1]
        
        # ===== 修改点 1：安全处理通道分组（使用 min 确保 chunk_size 不为 0）=====
        chunk_size = max(1, C_total // 4)
        X_chunks = list(torch.chunk(X, 4, dim=1))
        # 如果因整除问题导致最后一个 chunk 尺寸不一致，手动调整
        if len(X_chunks) != 4:
            # 简单回退：直接对整个 X 做多尺度池化
            branch1 = self._multi_scale_pool(X, H, W)
        else:
            multi_scale_features = []
            for chunk in X_chunks:
                pooled = self._multi_scale_pool(chunk, H, W)
                multi_scale_features.append(pooled)
            branch1 = torch.cat(multi_scale_features, dim=1)
        
        # 分支2和分支3也使用相同安全分组方式
        X_rot_h = X.transpose(2, 3)
        X_rot_h_chunks = torch.chunk(X_rot_h, 4, dim=1)
        multi_scale_h = [self._multi_scale_pool(c, H, W) for c in X_rot_h_chunks]
        branch2_temp = torch.cat(multi_scale_h, dim=1)
        branch2 = branch2_temp.transpose(2, 3)
        
        # 分支3：宽度交互（真正的逆时针旋转90度）
        X_rot_w = X.permute(0, 1, 3, 2)  # 交换 H 和 W，但需要确保后续恢复正确
        X_rot_w_chunks = torch.chunk(X_rot_w, 4, dim=1)
        multi_scale_w = [self._multi_scale_pool(c, H, W) for c in X_rot_w_chunks]
        branch3_temp = torch.cat(multi_scale_w, dim=1)
        branch3 = branch3_temp.permute(0, 1, 3, 2)
        
        # 平均融合
        fused = (branch1 + branch2 + branch3) / 3.0
        fused = self.conv_after_pool(fused)
        
        # ===== 修改点 2：精确按输入通道数切分 =====
        trans_enhanced = fused[:, :self.split_idx, :, :]
        cnn_enhanced = fused[:, self.split_idx:, :, :]
        
        _, _, Hc, Wc = cnn_feat.shape
        cnn_enhanced_resized = F.interpolate(cnn_enhanced, size=(Hc, Wc), mode='bilinear', align_corners=False)
        
        return trans_enhanced, cnn_enhanced_resized

    def _multi_scale_pool(self, x, H, W):
        """对输入张量进行多尺度池化并平均"""
        scales = [1, 2, 4, 8]
        pooled_list = []
        for s in scales:
            if s == 1:
                pooled = x
            else:
                pooled = F.adaptive_max_pool2d(x, (max(1, H//s), max(1, W//s)))
                pooled = F.interpolate(pooled, size=(H, W), mode='bilinear', align_corners=False)
            pooled_list.append(pooled)
        return torch.stack(pooled_list, dim=0).mean(dim=0)

class HybridModel(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(HybridModel, self).__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.neck_feat = cfg.TEST.NECK_FEAT


        # 计算 Transformer 特征图的空间尺寸
        img_h, img_w = cfg.INPUT.SIZE_TRAIN
        patch_size = 16
        stride = cfg.MODEL.STRIDE_SIZE[0]  # 假设 stride 在宽高上相同
        self.num_h = (img_h - patch_size) // stride + 1
        self.num_w = (img_w - patch_size) // stride + 1
        self.trans_hw = (self.num_h, self.num_w)

        # 1. Transformer 分支
        transformer_cfg = {
            'img_size': cfg.INPUT.SIZE_TRAIN,
            'sie_xishu': cfg.MODEL.SIE_COE,
            'camera': camera_num if cfg.MODEL.SIE_CAMERA else 0,
            'view': view_num if cfg.MODEL.SIE_VIEW else 0,
            'stride_size': cfg.MODEL.STRIDE_SIZE,
            'drop_path_rate': cfg.MODEL.DROP_PATH,
            'drop_rate': cfg.MODEL.DROP_OUT,
            'attn_drop_rate': cfg.MODEL.ATT_DROP_RATE,
            'return_intermediate': True  # 要求返回中间层特征
        }
        self.transformer = _factory_T_type[cfg.MODEL.TRANSFORMER_TYPE](**transformer_cfg)
        # 加载预训练权重
        if cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
            self.transformer.load_param(cfg.MODEL.PRETRAIN_PATH)
        # 加载 ballshow 上训练好的 transformer 权重（选择性加载）
        if cfg.MODEL.TRANSFORMER_FINETUNE_PATH:
            finetune_path = cfg.MODEL.TRANSFORMER_FINETUNE_PATH
            finetune_dict = torch.load(finetune_path, map_location='cpu')
            # 如果保存的是完整 checkpoint，提取模型权重
            if 'model_state_dict' in finetune_dict:
                finetune_dict = finetune_dict['model_state_dict']
            # 只加载与当前 transformer 分支匹配的键（忽略 token_se 等新模块）
            model_dict = self.transformer.state_dict()
            pretrained_dict = {k: v for k, v in finetune_dict.items() 
                            if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(pretrained_dict)
            self.transformer.load_state_dict(model_dict)
            print(f'Loaded ballshow-trained transformer weights from {finetune_path}')

        # 2. CNN 分支
        self.cnn = ResNet(last_stride=cfg.MODEL.LAST_STRIDE, block=Bottleneck, layers=[3,4,6,3])
        # 单独为 CNN 分支加载预训练权重（使用 resnet50-0676ba61.pth）
        if cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
            cnn_pretrain_path = cfg.MODEL.CNN_PRETRAIN_PATH if cfg.MODEL.CNN_PRETRAIN_PATH else cfg.MODEL.PRETRAIN_PATH
            self.cnn.load_param(cnn_pretrain_path)
            print(f'Loading pretrained ResNet model from {cnn_pretrain_path}')

        # 3. 用于映射 CNN 全局特征到 Transformer 维度的线性层
        self.proj_cnn_shallow = nn.Linear(512, 768)   # stage2 输出通道 512 -> 768
        self.proj_cnn_mid = nn.Linear(1024, 768)      # stage3 输出通道 1024 -> 768
        self.proj_cnn_deep = nn.Linear(2048, 768)     # stage4 输出通道 2048 -> 768

        # 4. 跨纬度多尺度特征融合模块
        self.fusion_shallow = CrossDimensionalMultiScaleFusion(
            in_channels_trans=768, in_channels_cnn=512,
            out_channels_trans=768, out_channels_cnn=512
        )
        self.fusion_mid = CrossDimensionalMultiScaleFusion(
            in_channels_trans=768, in_channels_cnn=1024,
            out_channels_trans=768, out_channels_cnn=1024
        )
        self.fusion_deep = CrossDimensionalMultiScaleFusion(
            in_channels_trans=768, in_channels_cnn=2048,
            out_channels_trans=768, out_channels_cnn=2048
        )

        # JPM 模块（参考 build_transformer_local）
        block = self.transformer.blocks[-1]  # 使用最后一个 Transformer 块
        layer_norm = self.transformer.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        self.shift_num = cfg.MODEL.SHIFT_NUM
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH
        self.rearrange = cfg.MODEL.RE_ARRANGE

        # 局部特征的分类器和 BNNeck
        self.bottleneck_1 = nn.BatchNorm1d(768)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(768)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(768)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(768)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)

        self.classifier_1 = nn.Linear(768, num_classes, bias=False)
        self.classifier_1.apply(weights_init_classifier)
        self.classifier_2 = nn.Linear(768, num_classes, bias=False)
        self.classifier_2.apply(weights_init_classifier)
        self.classifier_3 = nn.Linear(768, num_classes, bias=False)
        self.classifier_3.apply(weights_init_classifier)
        self.classifier_4 = nn.Linear(768, num_classes, bias=False)
        self.classifier_4.apply(weights_init_classifier)

        # 5. 最终融合后的分类头（ID loss）和 BNNeck
        self.bottleneck = nn.BatchNorm1d(768)   # 融合后的特征维度为 768
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.classifier = nn.Linear(768, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x, label=None, cam_label=None, view_label=None):
        # 1. 获取原始特征
        trans_features = self.transformer.forward_features(x, cam_label, view_label)  # 列表 [layer3, layer5, last]
        cnn_features = self.cnn(x)  # 列表 [f2, f3, f4]

        B, N, D = trans_features[0].shape
        H_t, W_t = self.trans_hw

        # ---------- 改动点：保存原始最后一层特征用于 JPM ----------
        raw_last_trans_seq = trans_features[2]   # (B, N, 768)，未经过融合

        # ---------- 浅层融合 (layer3 + stage2) ----------
        trans_spatial = trans_features[0][:, 1:, :].transpose(1, 2).reshape(B, D, H_t, W_t)
        cnn_shallow = cnn_features[0]
        trans_enhanced, cnn_enhanced_shallow = self.fusion_shallow(trans_spatial, cnn_shallow)

        trans_seq_shallow = trans_enhanced.flatten(2).transpose(1, 2)
        cls_token_original = trans_features[0][:, 0:1, :]
        trans_seq_shallow = torch.cat([cls_token_original, trans_seq_shallow], dim=1)

        # ---------- 中层融合 (layer5 + stage3) ----------
        mid_spatial = trans_seq_shallow[:, 1:, :].transpose(1, 2).reshape(B, D, H_t, W_t)
        cnn_mid = cnn_features[1]
        trans_enhanced_mid, cnn_enhanced_mid = self.fusion_mid(mid_spatial, cnn_mid)

        trans_seq_mid = trans_enhanced_mid.flatten(2).transpose(1, 2)
        cls_token_mid = trans_seq_shallow[:, 0:1, :]
        trans_seq_mid = torch.cat([cls_token_mid, trans_seq_mid], dim=1)

        # ---------- 深层融合 (last + stage4) ----------
        deep_spatial = trans_seq_mid[:, 1:, :].transpose(1, 2).reshape(B, D, H_t, W_t)
        cnn_deep = cnn_features[2]
        trans_enhanced_deep, cnn_enhanced_deep = self.fusion_deep(deep_spatial, cnn_deep)

        trans_seq_deep = trans_enhanced_deep.flatten(2).transpose(1, 2)
        cls_token_deep = trans_seq_mid[:, 0:1, :]
        trans_seq_deep = torch.cat([cls_token_deep, trans_seq_deep], dim=1)   # 融合后的全局序列

        # ========== JPM 分支（改动：使用原始序列而非融合后序列） ==========
        # 使用 raw_last_trans_seq 作为 JPM 的输入特征
        features = raw_last_trans_seq   # (B, N, D)，包含 cls token

        # 全局分支：通过 b1 后取 cls token（这里 b1 输入是融合前的序列）
        b1_feat = self.b1(features)
        global_feat_jpm = b1_feat[:, 0]   # 注意：这个全局特征与融合后的全局特征可能不同，可根据需要选择或结合

        # JPM 局部分支
        feature_length = features.size(1) - 1
        # 改进：处理不能整除情况（应用改动3）
        group_size = feature_length // self.divide_length
        remainder = feature_length % self.divide_length
        token = features[:, 0:1]

        if self.rearrange:
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
        else:
            x = features[:, 1:]

        # 提取各局部特征
        local_feats = []
        start = 0
        for i in range(self.divide_length):
            length = group_size + (1 if i < remainder else 0)
            end = start + length
            local_part = x[:, start:end]
            local_out = self.b2(torch.cat((token, local_part), dim=1))
            local_feat = local_out[:, 0]   # cls token 输出
            local_feats.append(local_feat)
            start = end

        local_feat_1, local_feat_2, local_feat_3, local_feat_4 = local_feats

        # BNNeck 处理（局部特征）
        feat_local1 = self.bottleneck_1(local_feat_1)
        feat_local2 = self.bottleneck_2(local_feat_2)
        feat_local3 = self.bottleneck_3(local_feat_3)
        feat_local4 = self.bottleneck_4(local_feat_4)

        # ---------- 全局特征的选择 ----------
        # 方案一：使用融合后的全局特征（推荐）
        global_feat_fused = trans_seq_deep[:, 0]   # 融合序列的 cls token
        feat_global = self.bottleneck(global_feat_fused)

        # 方案二：如果希望结合 JPM 全局特征，可加权融合，此处采用方案一

        if self.training:
            cls_score = self.classifier(feat_global)
            cls_score_1 = self.classifier_1(feat_local1)
            cls_score_2 = self.classifier_2(feat_local2)
            cls_score_3 = self.classifier_3(feat_local3)
            cls_score_4 = self.classifier_4(feat_local4)
            
            return [cls_score, cls_score_1, cls_score_2, cls_score_3, cls_score_4], \
                [feat_global, feat_local1, feat_local2, feat_local3, feat_local4]
        else:
            if self.neck_feat == 'after':
                return torch.cat([feat_global, feat_local1/4, feat_local2/4, feat_local3/4, feat_local4/4], dim=1)
            else:
                return torch.cat([global_feat_fused, local_feat_1/4, local_feat_2/4, local_feat_3/4, local_feat_4/4], dim=1)

        
    def load_param(self, trained_path):
        # 加载训练好的权重（用于恢复训练）
        param_dict = torch.load(trained_path, map_location='cpu')
        # 处理常见的 checkpoint 格式
        if 'model_state_dict' in param_dict:
            param_dict = param_dict['model_state_dict']
        elif 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        # 兼容分布式训练保存的键（去除 'module.' 前缀）
        new_state_dict = {}
        for k, v in param_dict.items():
            if k.startswith('module.'):
                new_k = k[7:]  # 去掉 'module.'
            else:
                new_k = k
            new_state_dict[new_k] = v
        # 加载权重（只加载匹配的键，避免 shape 不匹配）
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in new_state_dict.items() 
                        if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print('Loading pretrained model from {}'.format(trained_path))


def make_model(cfg, num_class, camera_num, view_num):
    if cfg.MODEL.NAME == 'transformer':
        if cfg.MODEL.JPM:
            model = build_transformer_local(num_class, camera_num, view_num, cfg, _factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE)
            print('===========building transformer with JPM module ===========')
        else:
            model = build_transformer(num_class, camera_num, view_num, cfg, _factory_T_type)
            print('===========building transformer===========')
    elif cfg.MODEL.NAME == 'hybrid':
        model = HybridModel(num_class, camera_num, view_num, cfg)
        print('===========building hybrid CNN-Transformer model===========')
    else:
        model = Backbone(num_class, cfg)
        print('===========building ResNet===========')
    return model