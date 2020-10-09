from torchvision import transforms as T
import torch
import numpy as np
import PIL
import cv2



class BasicNormalize():
    def __init__(self, norm=255):
        self.norm = norm
    def __call__(self, img):
        assert isinstance(img, torch.Tensor)
        return img / 255


base_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    BasicNormalize()
])