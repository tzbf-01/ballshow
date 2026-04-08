import torch
import torch.nn.functional as F
import math

def anisotropic_diffusion_filter(img_tensor, iterations=3, w1=0.5, w2=0.5, gamma=5.0, h=10.0):
    """
    改进的各向异性扩散滤波
    img_tensor: (C, H, W) 或 (B, C, H, W)，值域 [0,1] 或 [0,255]
    返回滤波后的同尺寸张量
    """
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)  # 加 batch 维度
    batch, ch, h_img, w_img = img_tensor.shape
    device = img_tensor.device
    v = img_tensor.clone()
    
    # 预定义 Sobel 核并复制到每个通道
    sobel_x = torch.tensor([[[-1,0,1],[-2,0,2],[-1,0,1]]], dtype=torch.float32, device=device).repeat(ch, 1, 1, 1)
    sobel_y = torch.tensor([[[-1,-2,-1],[0,0,0],[1,2,1]]], dtype=torch.float32, device=device).repeat(ch, 1, 1, 1)

    for _ in range(iterations):
        # 计算梯度 (使用 Sobel 或简单差分)
        grad_x = F.conv2d(v, sobel_x, padding=1, groups=ch)
        grad_y = F.conv2d(v, sobel_y, padding=1, groups=ch)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        
        # 改进的扩散系数 f1 (公式3)
        beta = grad_mag.std(dim=(2,3), keepdim=True)  # 局部标准差，简化计算
        f1 = 1.0 / (1.0 + w1 * grad_mag * (beta ** w2))
        
        # 边缘区域扩散系数为0 (公式6)
        edge_mask = (grad_mag > h).float()
        f = f1 * (1 - edge_mask)  # 非边缘用 f1，边缘为 0
        
        # 迭代更新 (公式7 简化版)
        delta_v = f * grad_mag
        v = v + (1.0 / gamma) * delta_v
    
    v = torch.clamp(v, 0, 1)   # 值域裁剪
    return v.squeeze(0) if img_tensor.dim() == 3 else v