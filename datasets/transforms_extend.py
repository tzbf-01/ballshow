import cv2
import numpy as np
from PIL import Image
import random
import torch

class MotionBlur(object):
    """模拟球员高速移动的运动模糊（适配PyTorch transforms接口）
    Args:
        prob: 应用模糊的概率
        kernel_size: 模糊核尺寸（奇数，越大模糊越明显）
    """
    def __init__(self, prob=0.3, kernel_size=5):
        self.prob = prob
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1  # 保证奇数

    def __call__(self, img):
        if random.uniform(0, 1) > self.prob:
            return img
        
        # 兼容PIL Image和Tensor两种输入（你的代码中ToTensor在增强后，所以这里是PIL Image）
        if isinstance(img, torch.Tensor):
            # 若输入是Tensor（极少情况），转回PIL
            img = img.permute(1,2,0).cpu().numpy()
            img = Image.fromarray((img * 255).astype(np.uint8))
        
        # PIL → OpenCV
        img_cv = np.array(img)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        
        # 生成运动模糊核（模拟球员移动方向）
        kernel = np.zeros((self.kernel_size, self.kernel_size))
        kernel[int((self.kernel_size-1)/2), :] = 1  # 水平模糊（球员横向移动为主）
        kernel = kernel / self.kernel_size
        
        # 随机旋转核（模拟不同移动方向）
        angle = random.randint(0, 360)
        M = cv2.getRotationMatrix2D((self.kernel_size/2, self.kernel_size/2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (self.kernel_size, self.kernel_size))
        
        # 应用模糊
        img_cv = cv2.filter2D(img_cv, -1, kernel)
        
        # OpenCV → PIL
        img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        return img

class RandomOcclusion(object):
    """模拟球员被遮挡（多人包夹/球体遮挡）
    Args:
        prob: 应用遮挡的概率
        occlusion_area: 遮挡区域占比范围（高/宽）
    """
    def __init__(self, prob=0.4, occlusion_area=(0.1, 0.3)):
        self.prob = prob
        self.occlusion_area = occlusion_area

    def __call__(self, img):
        if random.uniform(0, 1) > self.prob:
            return img
        
        # 兼容PIL Image输入
        if isinstance(img, torch.Tensor):
            img = img.permute(1,2,0).cpu().numpy()
            img = Image.fromarray((img * 255).astype(np.uint8))
        
        img_np = np.array(img)
        h, w, c = img_np.shape
        
        # 随机生成遮挡区域（避开全图遮挡）
        area_min, area_max = self.occlusion_area
        occl_h = int(h * random.uniform(area_min, area_max))
        occl_w = int(w * random.uniform(area_min, area_max))
        x = random.randint(0, max(1, w - occl_w))
        y = random.randint(0, max(1, h - occl_h))
        
        # 随机填充（模拟手臂/球体/其他球员遮挡）
        occl_value = np.random.randint(0, 255, (occl_h, occl_w, c))
        img_np[y:y+occl_h, x:x+occl_w, :] = occl_value
        
        return Image.fromarray(img_np)