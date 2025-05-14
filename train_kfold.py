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

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import torch.nn.functional as F
from loss import FocalLoss, weighted_normalized_CrossEntropyLoss, CenterLoss, CombinedLoss
import warnings
import json
from PIL import Image
import platform
import socket
import getpass
from torch.amp import autocast, GradScaler
import argparse

warnings.filterwarnings(action='ignore')

CFG = {
    'IMG_SIZE': 224,
    'EPOCHS': 18,
    'LEARNING_RATE': 3e-5,
    'BATCH_SIZE': 16,
    'SEED': 41,
    'WARM_UP': 3,
    'FOLDS': 3,        # K-Fold 수 추가
    'RUN_FOLDS': [1, 2, 3]  # 실행할 폴드 지정 (기본값은 모든 폴드)
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
        super(RandomCenterCrop, self).__init__(always_apply=always_apply, p=p)
        self.min_size = min_size
        self.max_size = max_size
    
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
    def __init__(self, img_path_list, label_list, transforms=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms
        
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        image = cv2.imread(img_path)
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        if self.label_list is not None:
            label = self.label_list[index]
            return image, label
        else:
            return image
        
    def __len__(self):
        return len(self.img_path_list)

def train(model, optimizer, train_loader, val_loader, scheduler, device, class_names, best_score=0, cur_epoch=1, experiment_name="base", folder_path="base", fold=None, accumulation_steps=4):
    model.to(device)
    criterion = weighted_normalized_CrossEntropyLoss(return_weights=False).to(device)
    
    best_model = None
    # fold 정보를 파일명에 추가
    save_path = os.path.join(folder_path, f"{experiment_name}-fold{fold}-best.pth") if fold is not None else os.path.join(folder_path, f"{experiment_name}-best.pth")

    for epoch in range(cur_epoch, CFG['EPOCHS'] + 1):
        model.train()
        train_loss = []
        optimizer.zero_grad()
        fold_str = f" (Fold {fold})" if fold is not None else ""
        progress_bar = tqdm(enumerate(iter(train_loader)), total=len(train_loader), desc=f"Epoch {epoch}/{CFG['EPOCHS']}{fold_str}")
        for step, (imgs, labels) in progress_bar:
            imgs = imgs.float().to(device)
            labels = labels.to(device).long()

            output = model(imgs)
            loss = criterion(output, labels)

            loss = loss / accumulation_steps

            loss.backward()
            if (step+1) % accumulation_steps == 0 or (step+1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            train_loss.append(loss.item())
            if step % 50 == 0:
                progress_bar.set_postfix(loss=loss.item() * accumulation_steps)
        _val_loss, _val_score, class_f1_dict, wandb_cm = validation(model, criterion, val_loader, device, class_names)

        log_data = {
            'epoch': epoch,
            'train_loss': sum(train_loss) / len(train_loss) * accumulation_steps,
            'val_loss': _val_loss * accumulation_steps,
            'val_macro_f1': _val_score,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'confusion_matrix': wandb_cm
        }
        
        # fold 정보 로그에 추가
        if fold is not None:
            log_data['fold'] = fold
            
        log_data.update(class_f1_dict)
        wandb.log(log_data)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'best_score': best_score,
            'fold': fold
        }
        
        # fold 정보를 파일명에 추가
        checkpoint_name = f'{experiment_name}-fold{fold}-epoch_{epoch}.pth' if fold is not None else f'{experiment_name}-epoch_{epoch}.pth'
        torch.save(checkpoint, os.path.join(folder_path, checkpoint_name))
        print(f"Checkpoint at epoch {epoch} saved → {checkpoint_name}")

        # 이전 체크포인트 삭제 (fold 포함)
        prev_checkpoint_name = f'{experiment_name}-fold{fold}-epoch_{epoch-1}.pth' if fold is not None else f'{experiment_name}-epoch_{epoch-1}.pth'
        prev_path = os.path.join(folder_path, prev_checkpoint_name)
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
                'best_score': best_score,
                'fold': fold
            }
            torch.save(checkpoint, save_path)
            print(f"Best model saved (epoch {epoch}, F1={_val_score:.4f}) → {save_path}")

    return best_model

def validation(model, criterion, val_loader, device, class_names):
    model.eval()
    val_loss = []
    preds, true_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(iter(val_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device).long()

            pred = model(imgs)
            loss = criterion(pred, labels)

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

def create_model_and_optimizers(model_name, num_classes, lr):
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    
    # 1. Warmup 스케줄러
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1/3,   # 초기 lr = base_lr * 1e-1
        end_factor=1.0,     # warmup 끝나면 base_lr
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
    
    return model, optimizer, scheduler

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rock Classification with K-Fold')
    parser.add_argument('--folds', nargs='+', type=int, help='Specify which folds to run (e.g., --folds 2 3 for folds 2 and 3)')
    parser.add_argument('--trained_path', type=str, default="", help='Path to trained model to continue training')
    parser.add_argument('--model_name', type=str, default="mambaout_base_plus_rw.sw_e150_in12k_ft_in1k", help='TIMM model name')
    
    args = parser.parse_args()
    
    if args.folds:
        CFG['RUN_FOLDS'] = args.folds
        
    trained_path = args.trained_path if args.trained_path else ""
    model_name = args.model_name

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    seed_everything(CFG['SEED'])

    all_img_list = glob.glob('./train/*/*')
    df = pd.DataFrame(columns=['img_path', 'rock_type'])
    df['img_path'] = all_img_list
    df['rock_type'] = df['img_path'].apply(lambda x : str(x).replace('\\','/').split('/')[2])

    le = preprocessing.LabelEncoder()
    df['encoded_rock_type'] = le.fit_transform(df['rock_type'])
    class_names = le.classes_

    # 실험 이름 설정
    if trained_path == "":
        base_name = model_name.replace('.','_')
        idx = len([x for x in os.listdir('./experiments') if x.startswith(base_name)])
        experiment_name = f"{base_name}_{idx+1}_kfold{CFG['FOLDS']}"
    else:
        experiment_name = os.path.splitext(os.path.basename(trained_path))[0].split('-')[0]
        if not experiment_name.endswith(f"_kfold{CFG['FOLDS']}"):
            experiment_name = f"{experiment_name}_kfold{CFG['FOLDS']}"
    
    folder_path = os.path.join("./experiments", experiment_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # 변환 정의
    train_transform = A.Compose([
        RandomCenterCrop(min_size=75, max_size=200, p=0.5),
        PadSquare(value=(0, 0, 0)),
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.GaussNoise(std_range=(0.1,0.15), p=0.5),
        A.CLAHE(p=0.5),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
    
    test_transform = A.Compose([
        RandomCenterCrop(min_size=75, max_size=200, p=0.4),
        PadSquare(value=(0, 0, 0)),
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
    
    # K-Fold 정의
    kfold = StratifiedKFold(n_splits=CFG['FOLDS'], shuffle=True, random_state=CFG['SEED'])
    fold_scores = []
    
    # 설정 저장
    config = {
        'experiment': {}, 
        'model': {}, 
        'train': {}, 
        'validation': {}, 
        'kfold': {}, 
        'seed': {}
    }
    
    config['experiment']['name'] = experiment_name
    config['model']['name'] = model_name
    config['model']['IMG_size'] = CFG['IMG_SIZE']
    config['train']['epoch'] = CFG['EPOCHS']
    config['train']['lr'] = CFG['LEARNING_RATE']
    config['train']['train_transform'] = [str(x) for x in train_transform]
    config['validation']['test_transform'] = [str(x) for x in test_transform]
    config['kfold']['n_splits'] = CFG['FOLDS']
    config['kfold']['run_folds'] = CFG['RUN_FOLDS']
    config['seed'] = CFG['SEED']
    
    config['system'] = {
        'hostname': socket.gethostname(),
        'username': getpass.getuser(),
        'platform': platform.system(),
        'platform-release': platform.release(),
        'architecture': platform.machine(),
        'processor': platform.processor(),
    }
    
    # optimizer 및 scheduler 세부 정보는 첫 fold 실행 시 추가됨
    
    # 설정 파일 저장
    config_path = os.path.join(folder_path, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    # K-Fold 반복
    for fold, (train_idx, val_idx) in enumerate(kfold.split(df, df['encoded_rock_type']), 1):
        # 지정된 fold만 실행
        if fold not in CFG['RUN_FOLDS']:
            print(f"Skipping fold {fold} as requested")
            continue
            
        print(f"\n{'='*30} Fold {fold} / {CFG['FOLDS']} {'='*30}\n")
        
        # wandb 초기화 (fold 별로 다른 run 생성)
        wandb_run = wandb.init(
            project="rock-classification",
            config=CFG,
            name=f"{experiment_name}_fold{fold}_{socket.gethostname()}",
            group=experiment_name,  # 동일 실험의 다른 fold들을 그룹화
            job_type=f"fold-{fold}",
            # id와 entity는 필요에 따라 설정
            # id=f"fold{fold}_by3jitd6",
            # entity="alexseo-inha-university",
            reinit=True  # 매 fold마다 새로운 run 생성
        )
        
        # 데이터 분할
        train_data = df.iloc[train_idx].reset_index(drop=True)
        val_data = df.iloc[val_idx].reset_index(drop=True)
        
        # 데이터셋 및 데이터로더 생성
        train_dataset = CustomDataset(train_data['img_path'].values, train_data['encoded_rock_type'].values, train_transform)
        train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=4, persistent_workers=True)

        val_dataset = CustomDataset(val_data['img_path'].values, val_data['encoded_rock_type'].values, test_transform)
        val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=4, persistent_workers=True)
        
        # 모델, 옵티마이저, 스케줄러 생성
        model, optimizer, scheduler = create_model_and_optimizers(
            model_name=model_name, 
            num_classes=len(class_names), 
            lr=CFG["LEARNING_RATE"]
        )
        
        # wandb 설정 업데이트
        wandb.config.update({
            "optimizer": optimizer.__class__.__name__,
            "scheduler": scheduler.__class__.__name__,
            "model": model_name,
            "fold": fold
        })
        
        # 첫 번째 fold에서만 config에 옵티마이저 및 스케줄러 세부 정보 추가
        if fold == CFG['RUN_FOLDS'][0]:
            config['train']['optimizer'] = {}
            config['train']['optimizer']['name'] = optimizer.__class__.__name__
            config['train']['scheduler'] = {}
            config['train']['scheduler']['name'] = scheduler.__class__.__name__
            config['train']['scheduler']['warmup'] = CFG['WARM_UP']
            
            for k, v in optimizer.state_dict()['param_groups'][0].items():
                if k == 'params': continue
                config['train']['optimizer'][k] = v
            
            for k, v in scheduler.state_dict().items():
                if k == 'params': continue
                config['train']['scheduler'][k] = v
                
            # 업데이트된 설정 파일 다시 저장
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
        
        # trained_path가 지정되었고, 현재 fold에 해당하는 체크포인트가 있는 경우
        best_score = 0
        start_epoch = 1
        
        if trained_path:
            fold_trained_path = trained_path
            if "-fold" not in trained_path:
                # 기존 체크포인트를 fold 버전으로 변환
                fold_trained_path = trained_path.replace("-best.pth", f"-fold{fold}-best.pth")
                
            if os.path.exists(fold_trained_path):
                checkpoint = torch.load(fold_trained_path, map_location=device)
                model.to(device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if checkpoint['scheduler_state_dict']:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_score = checkpoint['best_score']
                print(f"Loaded checkpoint from {fold_trained_path}")
        
        # 모델 학습
        fold_model = train(
            model=model, 
            optimizer=optimizer, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            scheduler=scheduler, 
            device=device, 
            class_names=class_names, 
            best_score=best_score, 
            cur_epoch=start_epoch, 
            experiment_name=experiment_name, 
            folder_path=folder_path,
            fold=fold
        )
        
        # 각 fold의 최종 성능 평가
        with torch.no_grad():
            criterion = weighted_normalized_CrossEntropyLoss(return_weights=False).to(device)
            val_loss, val_score, _, _ = validation(fold_model, criterion, val_loader, device, class_names)
            fold_scores.append(val_score)
            print(f"Fold {fold} validation score: {val_score:.4f}")
        
        # wandb 종료
        wandb_run.finish()
    
    # 모든 fold의 평균 성능 출력
    if fold_scores:
        avg_score = sum(fold_scores) / len(fold_scores)
        print(f"\nAverage validation score across {len(fold_scores)} folds: {avg_score:.4f}")
        
        # 최종 결과를 파일로 저장
        with open(os.path.join(folder_path, "fold_results.txt"), "w") as f:
            for i, score in enumerate(fold_scores, 1):
                f.write(f"Fold {i}: {score:.4f}\n")
            f.write(f"Average: {avg_score:.4f}\n")