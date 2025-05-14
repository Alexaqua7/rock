import os
import cv2
import torch
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

class RandomCenterCrop(A.BasicTransform):
    def __init__(self, crop_size=224, p=1.0):
        super(RandomCenterCrop, self).__init__(p=p)
        self.crop_size = crop_size

    def apply(self, image, **params):
        h, w, _ = image.shape
        crop_h = crop_w = self.crop_size
        if h < crop_h or w < crop_w:
            raise ValueError(f"Image too small for cropping: {h}x{w} vs crop size {crop_h}")

        center_y = random.randint(crop_h // 2, h - crop_h // 2)
        center_x = random.randint(crop_w // 2, w - crop_w // 2)
        top = center_y - crop_h // 2
        left = center_x - crop_w // 2

        return image[top:top+crop_h, left:left+crop_w]

    def get_transform_init_args_names(self):
        return ("crop_size",)

    @property
    def targets(self):
        return {'image': self.apply}

class RockCropDataset(Dataset):
    def __init__(self, image_dir, transform_full, transform_crop):
        self.image_paths = []
        
        # 서브 폴더 내의 이미지들 모두 불러오기 (클래스 이름은 필요 없으므로 제외)
        for class_name in os.listdir(image_dir):
            class_path = os.path.join(image_dir, class_name)
            if os.path.isdir(class_path):
                for fname in os.listdir(class_path):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(class_path, fname))
        
        print(f"[INFO] Found {len(self.image_paths)} images in total.")

        self.transform_full = transform_full
        self.transform_crop = transform_crop

    def __len__(self):
        return 2 * len(self.image_paths)  # 크롭과 풀 이미지를 합쳐서 두 배의 길이

    def __getitem__(self, idx):
        real_idx = idx // 2
        is_crop = idx % 2 == 1  # 홀수: crop, 짝수: full
        image = cv2.imread(self.image_paths[real_idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if is_crop:
            image = self.transform_crop(image=image)['image']
            label = 1  # crop 이미지 -> 레이블 1
        else:
            image = self.transform_full(image=image)['image']
            label = 0  # full 이미지 -> 레이블 0

        return image, label

def get_transforms():
    transform_full = A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.Normalize(),
        ToTensorV2()
    ])
    transform_crop = A.Compose([
        A.Resize(256, 256),
        RandomCenterCrop(crop_size=224),
        A.Normalize(),
        ToTensorV2()
    ])
    return transform_full, transform_crop

# 실행 예
CFG = {'BATCH_SIZE': 128}

if __name__ == '__main__':
    transform_full, transform_crop = get_transforms()

    dataset = RockCropDataset(
        image_dir='./train',
        transform_full=transform_full,
        transform_crop=transform_crop
    )
    loader = DataLoader(dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

    for images, labels in loader:
        print(f"Image batch shape: {images.shape}, Label batch: {labels}")
        break

    class RockClassifier(nn.Module):
        def __init__(self):
            super(RockClassifier, self).__init__()
            self.model = models.resnet18(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # 2개 클래스 (crop, full)

        def forward(self, x):
            return self.model(x)

# 모델 초기화
    model = RockClassifier()
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # 손실 함수와 옵티마이저 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 학습 루프
    epochs = 5  # 예시로 5 에폭만 학습
    for epoch in range(epochs):
        model.train()  # 모델을 학습 모드로 설정
        running_loss = 0.0

        with tqdm(loader, unit="batch") as tepoch:
            for images, labels in tepoch:
                images, labels = images.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')

                # 옵티마이저 기울기 초기화
                optimizer.zero_grad()

                # 순전파
                outputs = model(images)
                loss = criterion(outputs, labels)

                # 역전파
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # tqdm을 이용한 학습 상태 출력
                tepoch.set_description(f"Epoch {epoch+1}/{epochs}")
                tepoch.set_postfix(loss=running_loss / (tepoch.n + 1))

        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {running_loss / len(loader)}")
