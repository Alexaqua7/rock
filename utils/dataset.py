from torch.utils.data import Dataset
import cv2
import numpy as np
import torch
import os

class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, transforms=None, save_images=False, return_index=False):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms
        self.save_images = save_images
        self.return_index = return_index  # 인덱스 반환 여부를 결정하는 플래그

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        image = cv2.imread(img_path)
        if self.transforms is not None:
            transformed = self.transforms(image=image)
            image = transformed['image']

            if self.save_images:
                self.save_image(transformed, index)

        if self.label_list is not None:
            label = self.label_list[index]
            if self.return_index:
                return image, label, index  # 인덱스도 함께 반환
            else:
                return image, label
        else:
            if self.return_index:
                return image, index  # 인덱스도 함께 반환
            else:
                return image
        
    def __len__(self):
        return len(self.img_path_list)
    
    def save_image(self, transformed, index):
        # 텐서인 경우
        save_img = transformed['image']
        save_img = cv2.cvtColor(save_img, cv2.COLOR_BGR2RGB)
        if isinstance(save_img, torch.Tensor):
            img_np = save_img.permute(1, 2, 0).cpu().numpy()  # (C, H, W) → (H, W, C)
            img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)  # [0,1] → [0,255] 후 uint8
        else:
            img_np = transformed['image']
            if img_np.dtype != np.uint8:
                img_np = np.clip(img_np, 0, 255).astype(np.uint8)

        output_dir = "./saved_images"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f'transformed_image_{index}.jpg')
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, img_bgr)
        print(f"변환된 이미지 저장 완료: {save_path}")