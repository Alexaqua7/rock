from albumentations.core.transforms_interface import ImageOnlyTransform
import numpy as np
import albumentations as A
import cv2

class PadSquare(ImageOnlyTransform):
    def __init__(self, border_mode=0, value=0, always_apply=False, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.border_mode = border_mode
        self.value = value

    def apply(self, image, **params):
        h, w, c = image.shape
        max_dim = max(h, w)
        pad_h = max_dim - h
        pad_w = max_dim - w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.value)
        return image

    def get_transform_init_args_names(self):
        return ("border_mode", "value", "p")
    
    def __str__(self):
        return f"PadSquare(p={self.p}, border_mode={self.border_mode}, value={self.value})"

class RandomCenterCrop(ImageOnlyTransform):
    def __init__(self, min_size=75, max_size=200, always_apply=False, p=1.0):
        # 명시적으로 p 값 전달
        super(RandomCenterCrop, self).__init__(always_apply=always_apply, p=p)
        self.min_size = min_size
        self.max_size = max_size
        # 초기화 시 p 값 확인
    
    def apply(self, img, **params):
        h = np.random.randint(self.min_size, self.max_size)
        w = np.random.randint(self.min_size, self.max_size)
        crop = A.CenterCrop(height=h, width=w, pad_if_needed=True, p=1)
        return crop(image=img)['image']
    
    def get_transform_init_args_names(self):
        return ("min_size", "max_size", "p")
    
    def __str__(self):
        return f"RandomCenterCrop(p={self.p}, min_size={self.min_size}, max_size={self.max_size})"