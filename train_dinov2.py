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
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import torch.nn.functional as F
from utils.loss import FocalLoss, weighted_normalized_CrossEntropyLoss, CenterLoss, CombinedLoss
import warnings
import json
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import platform
import socket
import uuid
import getpass
import requests # 임시
warnings.filterwarnings(action='ignore')
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler, Sampler

CFG = {
    'IMG_SIZE': 224,
    'EPOCHS': 15,
    'LEARNING_RATE': 3e-6,
    'BATCH_SIZE': 32,
    'SEED': 41,
    'ACCUMULATION_STEPS': 2,
    'HARD_NEGATIVE_RATIO': 0.2,
    'HARD_NEGATIVE_MEMORY_SIZE': 1000
}

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
    def __init__(self, img_path_list, label_list, transforms=None, save_images=False):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms
        self.save_images = save_images

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
            return image, label
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
    def __init__(self, dataset_size, batch_size, hard_negative_miner, labels, num_classes, hard_negative_ratio=0.3):
        """
        하드 네거티브 샘플과 클래스 균형이 맞는 랜덤 샘플을 혼합하여 배치를 구성하는 샘플러
        
        Args:
            dataset_size: 데이터셋 크기
            batch_size: 배치 크기
            hard_negative_miner: 하드 네거티브 마이너 인스턴스
            labels: 각 샘플의 클래스 레이블 (numpy array)
            num_classes: 클래스 수
            hard_negative_ratio: 배치에서 하드 네거티브 샘플의 비율 (0~1)
        """
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.hard_negative_miner = hard_negative_miner
        self.labels = labels
        self.num_classes = num_classes
        self.hard_negative_ratio = hard_negative_ratio
        
        # 배치당 하드 네거티브 샘플 수
        self.hard_negative_per_batch = int(batch_size * hard_negative_ratio)
        self.random_per_batch = batch_size - self.hard_negative_per_batch
        
        # 클래스별 인덱스 구성
        self.class_indices = [[] for _ in range(num_classes)]
        for idx, label in enumerate(labels):
            self.class_indices[label].append(idx)
        
        # 에포크당 배치 수 계산
        self.num_batches = (dataset_size + batch_size - 1) // batch_size
        
    def __iter__(self):
        # 클래스별 인덱스 섞기
        for class_idx in range(self.num_classes):
            random.shuffle(self.class_indices[class_idx])
        
        # 각 클래스별 현재 인덱스 위치 초기화
        class_positions = [0] * self.num_classes
        
        for _ in range(self.num_batches):
            # 하드 네거티브 샘플 선택
            hard_indices = self.hard_negative_miner.get_hard_negatives(self.hard_negative_per_batch)
            hard_indices_set = set(hard_indices)
            
            # 클래스별로 균등하게 샘플 선택
            samples_per_class = self.random_per_batch // self.num_classes
            remaining_samples = self.random_per_batch % self.num_classes
            
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
                while len(selected_indices) < class_samples_needed and checked_indices < len(class_indices):
                    pos = (class_positions[class_idx] + checked_indices) % len(class_indices)
                    idx = class_indices[pos]
                    checked_indices += 1
                    
                    if idx not in hard_indices_set:
                        selected_indices.append(idx)
                
                # 충분한 샘플이 없으면 하드 네거티브와 중복 허용
                if len(selected_indices) < class_samples_needed:
                    additional_needed = class_samples_needed - len(selected_indices)
                    # 무작위로 해당 클래스에서 추가 샘플 선택
                    additional_indices = random.sample(
                        class_indices, 
                        min(additional_needed, len(class_indices))
                    )
                    selected_indices.extend(additional_indices)
                
                # 다음 에포크를 위해 클래스 위치 업데이트
                class_positions[class_idx] = (class_positions[class_idx] + len(selected_indices)) % len(class_indices)
                balanced_indices.extend(selected_indices)
            
            # 하드 네거티브와 균형 잡힌 샘플 합치기
            batch_indices = hard_indices + balanced_indices
            random.shuffle(batch_indices)  # 배치 내에서도 섞기
            
            yield batch_indices
    
    def __len__(self):
        return self.num_batches

def hard_negative_train(model, optimizer, train_loader, val_loader, scheduler, device, class_names, criterion=nn.CrossEntropyLoss(reduction='none'), hard_negative_miner=None, best_score=0, epochs=15, cur_epoch=1, experiment_name="base", folder_path="base", accumulation_steps=1):
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
            inputs = {'pixel_values':imgs}
            labels = labels.to(device).long()

            outputs = model(**inputs)
            logits = outputs.logits
            
            # 샘플별 손실값 계산 - 원래 스케일 유지 (accumulation_steps로 나누지 않음)
            sample_losses = criterion(logits, labels)
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
        _val_loss, _val_score, class_f1_dict, wandb_cm = hard_negative_validation(model, criterion, val_loader, device, class_names)

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

def hard_negative_validation(model, criterion, val_loader, device, class_names):
    model.eval()
    val_loss = []
    preds, true_labels = [], []

    for batch in tqdm(iter(val_loader)):
        if len(batch) == 3:  # 인덱스 포함하는 경우
            imgs, labels, _ = batch
        else:
            imgs, labels = batch
        
        imgs = imgs.float().to(device)
        inputs = {'pixel_values':imgs}
        labels = labels.to(device).long()

        outputs = model(**inputs)
        pred = outputs.logits
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
    model_name = "dinov2-reg" # TIMM 모델명 설정
    test_size = 0.2

    if trained_path == "":
        idx = len([x for x in os.listdir('./experiments') if x.startswith(model_name)])
        experiment_name = f"{model_name.replace('.','_')}_{idx+1}" # 실험이 저장될 folder 이름
    else:
        experiment_name = os.path.splitext(os.path.basename(trained_path))[0].split('-')[0]
    folder_path = os.path.join("./experiments", experiment_name)
    wandb.init(
        project="rock-classification",
        config=CFG,
        name=experiment_name,
        # resume='must',
        # id="3mdietie"
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    seed_everything(CFG['SEED'])

    all_img_list = glob.glob('./train/*/*')
    df = pd.DataFrame(columns=['img_path', 'rock_type'])
    df['img_path'] = all_img_list
    df['rock_type'] = df['img_path'].apply(lambda x : str(x).replace('\\','/').split('/')[2])

    train_data, val_data, _, _ = train_test_split(df, df['rock_type'], test_size=test_size, stratify=df['rock_type'], random_state=CFG['SEED'])

    le = preprocessing.LabelEncoder()
    train_data['rock_type'] = le.fit_transform(train_data['rock_type'])
    val_data['rock_type'] = le.transform(val_data['rock_type'])

    class_names = le.classes_
    num_classes = len(class_names)

    label_counts = Counter(train_data['rock_type'])

# le.classes_ 순서에 맞춰 클래스별 count 매핑
    class_counts = {
    class_name: label_counts[i] for i, class_name in enumerate(le.classes_)
}

    train_transform = A.Compose([
    RandomCenterCrop(min_size=75, max_size=200, p=0.5),
    A.HorizontalFlip(p=0.5),  # 50% 확률로 좌우 반전
    A.VerticalFlip(p=0.5),    # 50% 확률로 상하 반전
    A.GaussNoise(std_range=(0.1,0.15), p=0.5),
    A.Transpose(p=0.5),
    A.CLAHE(p=0.5),
    A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
    test_transform = A.Compose([
    RandomCenterCrop(min_size=75, max_size=200, p=0.4),
    A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
    ])

    train_dataset = CustomDataset(train_data['img_path'].values, train_data['rock_type'].values, train_transform)
    hard_negative_miner = HardNegativeMiner(
                    dataset_size=len(dataset['train_data']), 
                    memory_size=CFG['HARD_NEGATIVE_MEMORY_SIZE']
                )
                
    balanced_batch_sampler = BalancedHardNegativeBatchSampler(
        dataset_size=len(train_dataset),
        batch_size=CFG['BATCH_SIZE'],
        hard_negative_miner=hard_negative_miner,
        labels=train_data['rock_type'].values,
        num_classes=num_classes,
        hard_negative_ratio=CFG['HARD_NEGATIVE_RATIO']
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_sampler=balanced_batch_sampler,
        num_workers=16, 
        pin_memory=True,
        prefetch_factor=4
    )

    val_dataset = CustomDataset(val_data['img_path'].values, val_data['rock_type'].values, test_transform)
    val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=16, pin_memory=True, prefetch_factor=4)

    model = AutoModelForImageClassification.from_pretrained('./weights/facebook/dinov2-with-registers-large-imagenet1k-1-layer')

    in_features = model.classifier.in_features  # 기존 분류기의 입력 특성 수 (여기서는 2048)

    # 새로운 분류기 레이어 정의
    new_classifier = nn.Linear(in_features, len(class_names))

    # 모델의 분류기 레이어 교체
    model.classifier = new_classifier

    # 파라미터 그룹핑
    classifier_params = model.classifier.parameters()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=CFG["LEARNING_RATE"])
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG['EPOCHS'], eta_min=5e-7)

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

    config['validation']['test_transform'] = [str(x) for x in test_transform]

    config['split'] = test_size

    config['seed'] = CFG['SEED']

    for k, v in optimizer.state_dict()['param_groups'][0].items():
        if k == 'params': continue
        config['train']['optimizer'][k] = v

    for k, v in scheduler.state_dict().items():
        if k == 'params': continue
        config['train']['scheduler'][k] = v
    system_info = {
    'hostname': socket.gethostname(),
    'ip_address': socket.gethostbyname(socket.gethostname()),
    'user': getpass.getuser(),
    'platform': platform.platform(),
    'processor': platform.processor(),
    'machine': platform.machine(),
    'uuid': hex(uuid.getnode())
}

    config['system'] = system_info
    experiment_dir = f"./experiments/{experiment_name}"
    os.makedirs(experiment_dir, exist_ok=True)
    config_path = os.path.join(experiment_dir, "config2.json")
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
        infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device, class_names, best_score=best_score, cur_epoch=start_epoch, experiment_name=experiment_name, folder_path = folder_path, class_counts=class_counts, accumulation_steps=CFG['ACCUMULATION_STEPS'])
    else:
        infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device, class_names, experiment_name=experiment_name, folder_path = folder_path, class_counts=class_counts, accumulation_steps=CFG['ACCUMULATION_STEPS'])

    wandb.finish()