import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import glob

class ImageDataset(Dataset):
    def __init__(self, image_dir):
        # 모든 하위 폴더에서 이미지 찾기
        self.image_files = glob.glob(os.path.join(image_dir, '**/*.png'), recursive=True)
        self.image_files += glob.glob(os.path.join(image_dir, '**/*.jpg'), recursive=True)
        self.image_files += glob.glob(os.path.join(image_dir, '**/*.jpeg'), recursive=True)
        
        print(f"총 {len(self.image_files)}개 이미지를 찾았습니다.")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            tensor = self.transform(image)
            return tensor
        except Exception as e:
            print(f"이미지 로드 중 오류 발생 ({img_path}): {e}")
            # 오류 발생 시 대체 이미지 반환 (검은색 1x1 이미지)
            return torch.zeros((3, 1, 1))

def calculate_mean_std_pytorch(image_dir, batch_size=32):
    dataset = ImageDataset(image_dir)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    mean = 0.0
    std = 0.0
    total_images = 0

    print("계산 중: 평균...")
    for batch in tqdm(loader):
        batch_size = batch.size(0)
        batch = batch.view(batch_size, 3, -1)
        mean += batch.mean(2).sum(0)
        total_images += batch_size

    mean /= total_images

    print("계산 중: 표준편차...")
    var = 0.0
    total_pixels = 0

    for batch in tqdm(loader):
        batch_size = batch.size(0)
        pixels_per_image = batch.shape[2] * batch.shape[3]
        batch = batch.view(batch_size, 3, -1)
        var += ((batch - mean.unsqueeze(1)) ** 2).sum([0, 2])
        total_pixels += batch_size * pixels_per_image

    std = torch.sqrt(var / total_pixels)
    return mean.numpy(), std.numpy()


if __name__ == "__main__":
    image_directory = "C:/Users/Windows/Desktop/Rock/rock/train"
    mean, std = calculate_mean_std_pytorch(image_directory)
    print(f"평균: {mean}")
    print(f"표준편차: {std}")