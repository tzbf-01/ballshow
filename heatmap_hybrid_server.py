import sys
import os
import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont

# ==================== 配置区域（请根据实际情况修改） ====================
PROJECT_ROOT = '/root/giiit/双分支特征混合模型'      # 项目根目录
sys.path.append(PROJECT_ROOT)

WEIGHT_PATH = '/root/giiit/日志文件保存/CNN/checkpoint_ep120(轻量化模型).pth'             # 混合模型权重路径
IMAGE_PATH = '/root/giiit/双分支特征混合模型/data/BallShow/bounding_box_test/0002_c1s1_000009_00.jpg'                 # 测试图像路径
OUTPUT_PATH = './heatmap_hybrid.png'             # 输出图片路径


num_classes = 751                                     # 根据训练时调整
camera_num = 6
view_num = 0
use_jpm = True
img_size = (256, 128)
target_layer_idx = 8                                  # 提取第9层注意力
# ======================================================================

from model.make_model import make_model

class SimpleConfig:
    def __init__(self):
        self.MODEL = type('MODEL', (), {})()
        self.MODEL.NAME = 'hybrid'                    # 轻量化版本也是混合模型
        self.MODEL.TRANSFORMER_TYPE = 'vit_base_patch16_224_TransReID'
        self.MODEL.STRIDE_SIZE = [12, 12]
        self.MODEL.JPM = use_jpm
        self.MODEL.RE_ARRANGE = True
        self.MODEL.SIE_CAMERA = True
        self.MODEL.SIE_VIEW = False
        self.MODEL.SIE_COE = 3.0
        self.MODEL.DROP_PATH = 0.1
        self.MODEL.DROP_OUT = 0.0
        self.MODEL.ATT_DROP_RATE = 0.0
        self.MODEL.ID_LOSS_TYPE = 'softmax'
        self.MODEL.COS_LAYER = False
        self.MODEL.NECK = 'bnneck'
        self.MODEL.PRETRAIN_CHOICE = ''
        self.MODEL.PRETRAIN_PATH = ''
        self.MODEL.LAST_STRIDE = 1
        self.MODEL.SHUFFLE_GROUP = 2
        self.MODEL.SHIFT_NUM = 5
        self.MODEL.DEVIDE_LENGTH = 4
        self.MODEL.CNN_PRETRAIN_PATH = './osnet_x1_0_imagenet.pth'   # 轻量化 CNN 预训练权重
        self.MODEL.TRANSFORMER_FINETUNE_PATH = ''

        self.INPUT = type('INPUT', (), {})()
        self.INPUT.SIZE_TRAIN = img_size
        self.INPUT.PIXEL_MEAN = [0.5, 0.5, 0.5]
        self.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]

        self.TEST = type('TEST', (), {})()
        self.TEST.NECK_FEAT = 'before'
        self.TEST.FEAT_NORM = 'yes'

        self.SOLVER = type('SOLVER', (), {})()
        self.SOLVER.COSINE_SCALE = 30
        self.SOLVER.COSINE_MARGIN = 0.5


def load_model(weight_path):
    cfg = SimpleConfig()
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    checkpoint = torch.load(weight_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    filtered_dict = {k: v for k, v in new_state_dict.items()
                     if k in model.state_dict() and model.state_dict()[k].shape == v.shape}
    model.load_state_dict(filtered_dict, strict=False)
    model.eval()
    return model


def preprocess_image(image_path):
    """手动转换为 Tensor，避免 torchvision 依赖"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((img_size[1], img_size[0]))  # PIL 的 resize 参数是 (宽, 高)
    img_np = np.array(img)  # 0-255

    # 手动转换为 Tensor 并归一化
    img_tensor = torch.from_numpy(img_np).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    img_tensor = img_tensor.unsqueeze(0)  # 增加 batch 维度

    return img_tensor, img_np


def get_attention_heatmap(model, img_tensor):
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    cam_label = torch.zeros(1, dtype=torch.long, device=device)
    view_label = torch.zeros(1, dtype=torch.long, device=device)

    with torch.no_grad():
        _ = model(img_tensor, cam_label=cam_label, view_label=view_label)

    # 提取注意力矩阵
    if hasattr(model, 'base'):
        blocks = model.base.blocks
    elif hasattr(model, 'transformer'):
        blocks = model.transformer.blocks
    else:
        raise AttributeError("无法找到 Transformer blocks")

    attn_module = blocks[target_layer_idx].attn
    attn_matrix = attn_module.attn_matrix.cpu()
    attn_heads = attn_matrix[0].mean(dim=0)
    cls_attn = attn_heads[0, 1:]  # CLS 对其他 patch 的注意力

    H, W = img_size
    patch_size = 16
    stride = 12
    h_patches = (H - patch_size) // stride + 1
    w_patches = (W - patch_size) // stride + 1
    heatmap = cls_attn.reshape(h_patches, w_patches).numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_resized = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_CUBIC)
    return heatmap_resized


def create_overlay(img_np, heatmap):
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    if heatmap_colored.shape[:2] != img_np.shape[:2]:
        heatmap_colored = np.transpose(heatmap_colored, (1, 0, 2))
    overlay = (0.6 * img_np + 0.4 * heatmap_colored).astype(np.uint8)
    return overlay


def add_title_to_image(img_np, title, font_size=20):
    """用 PIL 在图像顶部添加标题"""
    img = Image.fromarray(img_np)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("simhei.ttf", font_size)
    except:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), title, font=font)
    text_width = bbox[2] - bbox[0]
    text_x = (img.width - text_width) // 2
    # 白色文字，黑色描边
    draw.text((text_x, 5), title, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    return np.array(img)


def main():
    print("加载轻量化混合模型...")
    model = load_model(WEIGHT_PATH)
    device = next(model.parameters()).device
    print(f"模型加载完成，设备: {device}")

    print(f"预处理图像: {IMAGE_PATH}")
    img_tensor, img_np = preprocess_image(IMAGE_PATH)

    print("提取注意力热图...")
    heatmap = get_attention_heatmap(model, img_tensor)

    print("生成叠加效果...")
    overlay = create_overlay(img_np, heatmap)

    # 为三张图添加标题
    img_original = add_title_to_image(img_np, "Original Image")
    heatmap_viz = (heatmap * 255).astype(np.uint8)
    heatmap_viz = cv2.applyColorMap(heatmap_viz, cv2.COLORMAP_JET)
    heatmap_viz = cv2.cvtColor(heatmap_viz, cv2.COLOR_BGR2RGB)
    if heatmap_viz.shape[:2] != img_np.shape[:2]:
        heatmap_viz = np.transpose(heatmap_viz, (1, 0, 2))
    heatmap_viz = add_title_to_image(heatmap_viz, "Attention Heatmap (Lightweight)")
    overlay = add_title_to_image(overlay, "Overlay")

    # 水平拼接三张图
    combined = np.hstack([img_original, heatmap_viz, overlay])
    Image.fromarray(combined).save(OUTPUT_PATH)
    print(f"轻量化模型热力图已保存至 {OUTPUT_PATH}")


if __name__ == '__main__':
    main()