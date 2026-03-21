import cv2
import numpy as np
from PIL import Image
import random
import torch

class MotionBlur(object):
    def __init__(self, prob=0.5, kernel_size=7):  # 升级：概率0.5，核尺寸7
        self.prob = prob
        self.kernel_size = kernel_size

    def __call__(self, img):
        if random.random() < self.prob:
            # 生成运动模糊核
            kernel = np.zeros((self.kernel_size, self.kernel_size))
            kernel[int((self.kernel_size-1)/2), :] = 1
            kernel = kernel / self.kernel_size
            img = cv2.filter2D(np.array(img), -1, kernel)
            img = Image.fromarray(img)
        return img

class RandomOcclusion(object):
    def __init__(self, prob=0.6, occlusion_area=(0.1, 0.4)):  # 升级：概率0.6，更大遮挡区域
        self.prob = prob
        self.occlusion_area = occlusion_area

    def __call__(self, img):
        if random.random() < self.prob:
            img_np = np.array(img)
            h, w, c = img_np.shape
            # 随机生成遮挡区域大小
            area_ratio = random.uniform(*self.occlusion_area)
            occl_h = int(h * area_ratio)
            occl_w = int(w * area_ratio)
            # 随机生成遮挡位置
            x = random.randint(0, w - occl_w)
            y = random.randint(0, h - occl_h)
            # 随机填充颜色（0-255）
            occl_color = np.random.randint(0, 256, (occl_h, occl_w, c), dtype=np.uint8)
            img_np[y:y+occl_h, x:x+occl_w, :] = occl_color
            img = Image.fromarray(img_np)
        return img