import random
import pandas as pd
import numpy as np
import os
import re
import glob
import cv2
import timm
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Sampler, DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import torch.nn.functional as F
from utils.loss import FocalLoss, weighted_normalized_CrossEntropyLoss, CenterLoss, CombinedLoss
import warnings
import json
from PIL import Image
import platform
import socket
import getpass
from torch.amp import autocast, GradScaler

warnings.filterwarnings(action='ignore')

CFG = {
    'IMG_SIZE': 384,
    'EPOCHS': 18,
    'LEARNING_RATE': 3e-5,
    'BATCH_SIZE': 16,
    'SEED': 41,
    'WARM_UP': 3,
    'HARD_NEGATIVE_MEMORY_SIZE': 1000,
    'HARD_NEGATIVE_RATIO': 0.2,
    'BALANCED_BATCH': True,
    'NUM_WORKERS': 16,
    'ACCUMULATION_STEPS': 8,
    'FOLD': 5,
    'START_FOLD': 2,
}

class HardNegativeMiner:
    def __init__(self, dataset_size, memory_size=1000):
        self.hard_negative_indices = set()
        self.hard_negative_scores = {}  # 각 샘플의 손실값 저장
        self.memory_size = memory_size
        self.dataset_size = dataset_size
    
    def update(self, indices, losses):
        """
        손실값이 높은 샘플들을 하드 네거티브로 추가
        
        Args:
            indices: 샘플 인덱스 리스트
            losses: 각 샘플의 손실값 리스트
        """
        # 인덱스와 손실값 매핑
        for idx, loss in zip(indices, losses):
            self.hard_negative_scores[idx] = loss.item()
        
        # 손실값 기준으로 정렬하여 상위 memory_size개만 유지
        sorted_items = sorted(self.hard_negative_scores.items(), 
                              key=lambda x: x[1], reverse=True)
        
        if len(sorted_items) > self.memory_size:
            sorted_items = sorted_items[:self.memory_size]
        
        # 새로운 하드 네거티브 인덱스 집합 구성
        self.hard_negative_indices = set([idx for idx, _ in sorted_items])
        self.hard_negative_scores = {idx: loss for idx, loss in sorted_items}
    
    def get_hard_negatives(self, count):
        """
        하드 네거티브 샘플에서 'count'만큼 샘플링
        
        Args:
            count: 필요한 하드 네거티브 샘플 수
            
        Returns:
            선택된 하드 네거티브 인덱스 리스트
        """
        if not self.hard_negative_indices:
            # 하드 네거티브 샘플이 없으면 랜덤 인덱스 반환
            return random.sample(range(self.dataset_size), min(count, self.dataset_size))
        
        # 가중치 기반 샘플링을 위한 준비
        indices = list(self.hard_negative_indices)
        weights = [self.hard_negative_scores[idx] for idx in indices]
        
        # 가중치 정규화
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # 가중치 기반 샘플링
        if len(indices) <= count:
            return indices
        else:
            return np.random.choice(indices, count, replace=False, p=weights).tolist()

class BalancedHardNegativeBatchSampler(Sampler):
    def __init__(self, dataset_size, batch_size, hard_negative_miner, labels, num_classes, 
                 hard_negative_ratio=0.3, accumulation_steps=1):
        """
        Gradient Accumulation을 고려한 하드 네거티브 샘플과 클래스 균형 샘플러
        
        Args:
            dataset_size: 데이터셋 크기
            batch_size: 개별 배치 크기
            hard_negative_miner: 하드 네거티브 마이너 인스턴스
            labels: 각 샘플의 클래스 레이블 (numpy array)
            num_classes: 클래스 수
            hard_negative_ratio: effective batch에서 하드 네거티브 샘플의 비율 (0~1)
            accumulation_steps: gradient accumulation 스텝 수
        """
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.hard_negative_miner = hard_negative_miner
        self.labels = labels
        self.num_classes = num_classes
        self.hard_negative_ratio = hard_negative_ratio
        self.accumulation_steps = accumulation_steps
        
        # Effective batch size 계산
        self.effective_batch_size = batch_size * accumulation_steps
        
        # Effective batch당 하드 네거티브 샘플 수
        self.hard_negative_per_effective_batch = int(self.effective_batch_size * hard_negative_ratio)
        self.random_per_effective_batch = self.effective_batch_size - self.hard_negative_per_effective_batch
        
        # 클래스별 인덱스 구성
        self.class_indices = [[] for _ in range(num_classes)]
        for idx, label in enumerate(labels):
            self.class_indices[label].append(idx)
        
        # 에포크당 effective batch 수 계산
        self.num_effective_batches = (dataset_size + self.effective_batch_size - 1) // self.effective_batch_size
        self.num_batches = self.num_effective_batches * accumulation_steps
        
        print(f"Sampler Configuration:")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Accumulation steps: {accumulation_steps}")
        print(f"  - Effective batch size: {self.effective_batch_size}")
        print(f"  - Hard negative per effective batch: {self.hard_negative_per_effective_batch}")
        print(f"  - Random samples per effective batch: {self.random_per_effective_batch}")
        print(f"  - Samples per class per effective batch: {self.random_per_effective_batch // num_classes}")
        
    def _get_balanced_samples_for_effective_batch(self, hard_indices_set, class_positions):
        """Effective batch에 대해 클래스별로 균등하게 샘플 선택"""
        samples_per_class = self.random_per_effective_batch // self.num_classes
        remaining_samples = self.random_per_effective_batch % self.num_classes
        
        balanced_indices = []
        
        # 각 클래스에서 samples_per_class 개의 샘플 선택
        for class_idx in range(self.num_classes):
            class_samples_needed = samples_per_class + (1 if class_idx < remaining_samples else 0)
            if class_samples_needed == 0:
                continue
            
            selected_indices = []
            checked_indices = 0
            class_indices = self.class_indices[class_idx]
            
            # 이미 선택된 하드 네거티브와 중복되지 않는 샘플 선택
            while len(selected_indices) < class_samples_needed and checked_indices < len(class_indices) * 2:
                pos = (class_positions[class_idx] + checked_indices) % len(class_indices)
                idx = class_indices[pos]
                checked_indices += 1
                
                if idx not in hard_indices_set:
                    selected_indices.append(idx)
            
            # 충분한 샘플이 없으면 하드 네거티브와 중복 허용
            if len(selected_indices) < class_samples_needed:
                additional_needed = class_samples_needed - len(selected_indices)
                # 무작위로 해당 클래스에서 추가 샘플 선택
                additional_candidates = [idx for idx in class_indices if idx not in selected_indices]
                if additional_candidates:
                    additional_indices = random.sample(
                        additional_candidates, 
                        min(additional_needed, len(additional_candidates))
                    )
                    selected_indices.extend(additional_indices)
                
                # 여전히 부족하면 중복 허용하여 채움
                while len(selected_indices) < class_samples_needed:
                    selected_indices.append(random.choice(class_indices))
            
            # 다음 effective batch를 위해 클래스 위치 업데이트
            class_positions[class_idx] = (class_positions[class_idx] + len(selected_indices)) % len(class_indices)
            balanced_indices.extend(selected_indices)
        
        return balanced_indices
    
    def __iter__(self):
        # 클래스별 인덱스 섞기
        for class_idx in range(self.num_classes):
            random.shuffle(self.class_indices[class_idx])
        
        # 각 클래스별 현재 인덱스 위치 초기화
        class_positions = [0] * self.num_classes
        
        for effective_batch_idx in range(self.num_effective_batches):
            # Effective batch에 대해 하드 네거티브 샘플 선택
            hard_indices = self.hard_negative_miner.get_hard_negatives(self.hard_negative_per_effective_batch)
            hard_indices_set = set(hard_indices)
            
            # Effective batch에 대해 클래스별로 균등하게 샘플 선택
            balanced_indices = self._get_balanced_samples_for_effective_batch(
                hard_indices_set, class_positions
            )
            
            # 전체 effective batch 인덱스
            effective_batch_indices = hard_indices + balanced_indices
            random.shuffle(effective_batch_indices)
            
            # Effective batch를 개별 배치들로 분할
            for step in range(self.accumulation_steps):
                start_idx = step * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(effective_batch_indices))
                
                batch_indices = effective_batch_indices[start_idx:end_idx]
                
                # 배치 크기가 부족한 경우 패딩 (마지막 배치에서 발생 가능)
                while len(batch_indices) < self.batch_size and len(effective_batch_indices) > 0:
                    batch_indices.append(random.choice(effective_batch_indices))
                
                yield batch_indices
    
    def __len__(self):
        return self.num_batches


# 사용 예시를 위한 수정된 데이터로더 생성 함수
def create_train_loader_with_accumulation(train_dataset, hard_negative_miner, labels, num_classes, 
                                        batch_size, accumulation_steps, hard_negative_ratio=0.2, num_workers=16, prefetch_factor=4, pin_memory=True):
    """
    Gradient Accumulation을 고려한 train loader 생성
    """
    balanced_batch_sampler = BalancedHardNegativeBatchSampler(
        dataset_size=len(train_dataset),
        batch_size=batch_size,
        hard_negative_miner=hard_negative_miner,
        labels=labels,
        num_classes=num_classes,
        hard_negative_ratio=hard_negative_ratio,
        accumulation_steps=accumulation_steps
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_sampler=balanced_batch_sampler,
        num_workers=num_workers, 
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor
    )
    
    return train_loader

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class PadSquare(ImageOnlyTransform):
    def __init__(self, border_mode=0, value=0, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
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

def train(model, optimizer, train_loader, val_loader, scheduler, device, class_names, criterion=nn.CrossEntropyLoss(reduction='none'), hard_negative_miner=None, best_score=0, epochs=15, cur_epoch=1, experiment_name="base", folder_path="base", accumulation_steps=1):
    model.to(device)
    
    best_model = None
    save_path = os.path.join(folder_path, f"{experiment_name}-best.pth")
    
    for epoch in range(cur_epoch, epochs + 1):
        model.train()
        train_loss = []
        optimizer.zero_grad()
        
        # 그래디언트 누적을 위한 배치 손실 및 인덱스 저장소
        accumulated_indices = []
        accumulated_losses = []
        
        progress_bar = tqdm(enumerate(iter(train_loader)), total=len(train_loader), desc=f"Epoch {epoch}/{epochs}")
        for step, batch in progress_bar:
            if len(batch) == 3:  # 인덱스 포함하는 경우
                imgs, labels, indices = batch
            else:
                imgs, labels = batch
                indices = None
            
            imgs = imgs.float().to(device)
            labels = labels.to(device).long()

            output = model(imgs)
            
            # 샘플별 손실값 계산 - 원래 스케일 유지 (accumulation_steps로 나누지 않음)
            sample_losses = criterion(output, labels)
            loss = sample_losses.mean() / accumulation_steps  # 그래디언트 스케일링을 위해서만 나눔

            # 역전파 수행
            loss.backward()
            
            # 현재 배치의 손실값과 인덱스 저장
            if indices is not None:
                accumulated_indices.extend(indices.cpu().numpy())
                accumulated_losses.extend(sample_losses.detach().cpu())

            train_loss.append(loss.item() * accumulation_steps)  # 원래 손실값 저장 (scaling 제거)
            
            # 그래디언트 업데이트 및 하드 네거티브 마이너 업데이트
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                # 옵티마이저 스텝
                optimizer.step()
                optimizer.zero_grad()
                
                # 누적된 배치들에 대한 하드 네거티브 마이너 업데이트
                if hard_negative_miner is not None and accumulated_indices:
                    hard_negative_miner.update(accumulated_indices, accumulated_losses)
                
                # 누적 저장소 초기화
                accumulated_indices = []
                accumulated_losses = []
            
            # 진행상황 표시
            progress_bar.set_postfix(loss=loss.item() * accumulation_steps)  # 원래 손실값 표시

        # 검증 단계
        _val_loss, _val_score, class_f1_dict, wandb_cm = validation(model, criterion, val_loader, device, class_names)

        log_data = {
            'epoch': epoch,
            'train_loss': sum(train_loss) / len(train_loss),
            'val_loss': _val_loss,
            'val_macro_f1': _val_score,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'confusion_matrix': wandb_cm
        }
        log_data.update(class_f1_dict)
        wandb.log(log_data)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'best_score': best_score
        }
        torch.save(checkpoint, os.path.join(folder_path, f'{experiment_name}-epoch_{epoch}.pth'))
        print(f"Checkpoint at epoch {epoch} saved → {experiment_name}-epoch_{epoch}.pth")

        prev_path = os.path.join(folder_path, f'{experiment_name}-epoch_{epoch-1}.pth')
        if epoch > 1 and os.path.exists(prev_path):
            os.remove(prev_path)
            print(f"Previous checkpoint deleted → {prev_path}")

        if scheduler is not None:
            scheduler.step()

        if best_score < _val_score:
            best_score = _val_score
            best_model = model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'best_score': best_score
            }
            torch.save(checkpoint, save_path)
            print(f"Best model saved (epoch {epoch}, F1={_val_score:.4f}) → {save_path}")

    return best_model

def validation(model, criterion, val_loader, device, class_names):
    model.eval()
    val_loss = []
    preds, true_labels = [], []

    with torch.no_grad():
        for batch in tqdm(iter(val_loader)):
            if len(batch) == 3:  # 인덱스 포함하는 경우
                imgs, labels, _ = batch
            else:
                imgs, labels = batch
            
            imgs = imgs.float().to(device)
            labels = labels.to(device).long()

            pred = model(imgs)
            sample_losses = criterion(pred, labels)  # 샘플별 손실값 계산
            loss = sample_losses.mean()  # 평균 손실값 계산

            preds += pred.argmax(1).detach().cpu().numpy().tolist()
            true_labels += labels.detach().cpu().numpy().tolist()
            val_loss.append(loss.item())

        _val_loss = np.mean(val_loss)
        _val_score = f1_score(true_labels, preds, average='macro')

    # 🔹 class별 F1 (클래스명 포함)
    class_f1 = f1_score(true_labels, preds, average=None)
    class_f1_dict = {f'{class_names[i]}_f1': f for i, f in enumerate(class_f1)}

    # 🔹 confusion matrix
    cm = confusion_matrix(true_labels, preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    
    # 🔹 wandb 이미지로 저장
    wandb_cm = wandb.Image(fig)
    plt.close(fig)

    return _val_loss, _val_score, class_f1_dict, wandb_cm

if __name__ == '__main__':
    trained_path = "" # 이어서 학습을 진행할 경우, 학습된 모델 경로 설정. 처음부터 학습을 진행시킬 것이라면, 공백으로 설정
    model_name = "mambaout_base_plus_rw.sw_e150_r384_in12k_ft_in1k" # TIMM 모델명 설정
    test_size = 0.2

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    seed_everything(CFG['SEED'])

    all_img_list = glob.glob('./train/*/*')
    df = pd.DataFrame(columns=['img_path', 'rock_type'])
    df['img_path'] = all_img_list
    df['rock_type'] = df['img_path'].apply(lambda x : str(x).replace('\\','/').split('/')[2])

    le = preprocessing.LabelEncoder()
    df['rock_type'] = le.fit_transform(df['rock_type'])

    # train_data, val_data, _, _ = train_test_split(df, df['rock_type'], test_size=test_size, stratify=df['rock_type'], random_state=CFG['SEED'])

    # K-Fold Cross Validation
    kfold = StratifiedKFold(n_splits=CFG['FOLD'], shuffle=True, random_state=CFG['SEED'])

    class_names = le.classes_
    num_classes = len(class_names)

    train_transform = A.Compose([
    RandomCenterCrop(min_size=75, max_size=200, p=0.5),
    A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
    A.HorizontalFlip(p=0.5),  # 50% 확률로 좌우 반전
    A.VerticalFlip(p=0.5),    # 50% 확률로 상하 반전
    A.CLAHE(p=0.5),
    A.GaussNoise(std_range=(0.1,0.15), p=0.5),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])
    test_transform = A.Compose([
        RandomCenterCrop(min_size=75, max_size=200, p=0.4),
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
    idx = 1
    n_folds = CFG['FOLD']
    for fold, (train_idx, val_idx) in enumerate(kfold.split(df, df['rock_type'])):
        print(f"\n========== FOLD {fold + 1} / {n_folds} ==========")
        if fold in [0,1]:
            continue

        if trained_path == "":
            idx = len([x for x in os.listdir('./experiments') if x.startswith(model_name)]) if idx is None else idx
            experiment_name = f"{model_name.replace('.','_')}_{idx+1}_fold{fold+1}"  # 실험이 저장될 folder 이름
        else:
            experiment_name = os.path.splitext(os.path.basename(trained_path))[0].split('-')[0]
        folder_path = os.path.join("./experiments", experiment_name)
        wandb.init(
            project="rock-classification",
            config=CFG,
            name=f"{experiment_name}_{socket.gethostname()}_fold{fold+1}",
            resume='must',
            id="5fjrgtaw",
            entity = "alexseo-inha-university"
        )

        train_data = df.iloc[train_idx].copy().reset_index(drop=True)
        val_data = df.iloc[val_idx].copy().reset_index(drop=True)

        train_dataset = CustomDataset(train_data['img_path'].values, train_data['rock_type'].values, train_transform, return_index = True)
        hard_negative_miner = HardNegativeMiner(
                        dataset_size=len(train_data), 
                        memory_size=CFG['HARD_NEGATIVE_MEMORY_SIZE']
                    )
                    
        train_loader = create_train_loader_with_accumulation(train_dataset, hard_negative_miner, train_data['rock_type'].values, num_classes, 
                            CFG['BATCH_SIZE'], CFG['ACCUMULATION_STEPS'], hard_negative_ratio=CFG['HARD_NEGATIVE_RATIO'], num_workers=CFG['NUM_WORKERS'])

        val_dataset = CustomDataset(val_data['img_path'].values, val_data['rock_type'].values, test_transform)
        val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=16, pin_memory=True, prefetch_factor=4, persistent_workers=True)

        model = timm.create_model(model_name, pretrained=True, num_classes=len(class_names))
        optimizer = torch.optim.Adam(params=model.parameters(), lr=CFG["LEARNING_RATE"])

        # 1. Warmup 스케줄러
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1/3,   # 초기 lr = base_lr * 1e-1
            end_factor=1.0,      # warmup 끝나면 base_lr
            total_iters=CFG['WARM_UP']
        )

        # 2. 이후 CosineAnnealing
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=CFG['EPOCHS'] - CFG['WARM_UP'],
            eta_min=1e-6
        )

        # 3. Sequential하게 연결
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[CFG['WARM_UP']]
        )

        wandb.config.update({
            "optimizer": optimizer.__class__.__name__,
            "scheduler": scheduler.__class__.__name__,
            "model": model_name
        })



        ###
        config = {'experiment': {}, 'model':{}, 'train':{}, 'validation':{}, 'split':{}, 'seed': {}}
        config['experiment']['name'] = experiment_name

        config['model']['name'] = model_name
        config['model']['IMG_size'] = CFG['IMG_SIZE']

        config['train']['epoch'] = CFG['EPOCHS']
        config['train']['lr'] = CFG['LEARNING_RATE']
        config['train']['train_transform'] = [str(x) for x in train_transform]
        config['train']['optimizer'] = {}
        config['train']['optimizer']['name'] = optimizer.__class__.__name__
        config['train']['scheduler'] = {}
        config['train']['scheduler']['name'] = scheduler.__class__.__name__
        config['train']['scheduler']['warmup'] = CFG['WARM_UP']

        config['validation']['test_transform'] = [str(x) for x in test_transform]

        config['split'] = test_size

        config['seed'] = CFG['SEED']

        config['system'] = {
        'hostname': socket.gethostname(),
        'username': getpass.getuser(),
        'platform': platform.system(),
        'platform-release': platform.release(),
        'architecture': platform.machine(),
        'processor': platform.processor(),
        }

        for k, v in optimizer.state_dict()['param_groups'][0].items():
            if k == 'params': continue
            config['train']['optimizer'][k] = v

        for k, v in scheduler.state_dict().items():
            if k == 'params': continue
            config['train']['scheduler'][k] = v
        experiment_dir = f"./experiments/{experiment_name}"
        os.makedirs(experiment_dir, exist_ok=True)
        config_path = os.path.join(experiment_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

        ###


        if trained_path != "":
            checkpoint = torch.load(trained_path, map_location=device)
            model.to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_score = checkpoint['best_score']
            infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device, class_names, best_score=best_score, cur_epoch=start_epoch, experiment_name=experiment_name, epochs = CFG["EPOCHS"], folder_path = folder_path, hard_negative_miner = hard_negative_miner)
        else:
            infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device, class_names, experiment_name=experiment_name, folder_path = folder_path, hard_negative_miner = hard_negative_miner)

        wandb.finish()
