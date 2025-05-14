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

from sklearn.model_selection import train_test_split
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

warnings.filterwarnings(action='ignore')

CFG = {
    'IMG_SIZE': 224,
    'EPOCHS': 18,
    'LEARNING_RATE': 3e-5,
    'BATCH_SIZE': 16,
    'SEED': 41,
    'WARM_UP': 3
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

def train(model, optimizer, train_loader, val_loader, scheduler, device, class_names, best_score=0, cur_epoch=1, experiment_name="base", folder_path = "base", accumulation_steps=4):
    model.to(device)
    criterion = weighted_normalized_CrossEntropyLoss(return_weights=False).to(device)
    
    best_model = None
    save_path = os.path.join(folder_path, f"{experiment_name}-best.pth")

    for epoch in range(cur_epoch, CFG['EPOCHS'] + 1):
        model.train()
        train_loss = []
        optimizer.zero_grad()
        progress_bar = tqdm(enumerate(iter(train_loader)), total=len(train_loader), desc=f"Epoch {epoch}/{CFG['EPOCHS']}")
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

if __name__ == '__main__':
    trained_path = "./experiments/mambaout_base_plus_rw_sw_e150_in12k_ft_in1k_1/mambaout_base_plus_rw_sw_e150_in12k_ft_in1k_1-best.pth" # 이어서 학습을 진행할 경우, 학습된 모델 경로 설정. 처음부터 학습을 진행시킬 것이라면, 공백으로 설정
    model_name = "mambaout_base_plus_rw.sw_e150_in12k_ft_in1k" # TIMM 모델명 설정
    test_size = 0.3



    if trained_path == "":
        idx = len([x for x in os.listdir('./experiments') if x.startswith(model_name)])
        experiment_name = f"{model_name.replace('.','_')}_{idx+1}" # 실험이 저장될 folder 이름
    else:
        experiment_name = os.path.splitext(os.path.basename(trained_path))[0].split('-')[0]
    folder_path = os.path.join("./experiments", experiment_name)
    wandb.init(
        project="rock-classification",
        config=CFG,
        name=f"{experiment_name}_{socket.gethostname()}",
        resume='must',
        id="by3jitd6",
        entity = "alexseo-inha-university"
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

    train_transform = A.Compose([
    RandomCenterCrop(min_size=75, max_size=200, p=0.5),
    PadSquare(value=(0, 0, 0)),
    A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
    A.HorizontalFlip(p=0.5),  # 50% 확률로 좌우 반전
    A.VerticalFlip(p=0.5),    # 50% 확률로 상하 반전
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

    train_dataset = CustomDataset(train_data['img_path'].values, train_data['rock_type'].values, train_transform)
    train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=4, persistent_workers=True)

    val_dataset = CustomDataset(val_data['img_path'].values, val_data['rock_type'].values, test_transform)
    val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=4, persistent_workers=True)

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
        infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device, class_names, best_score=best_score, cur_epoch=start_epoch, experiment_name=experiment_name, folder_path = folder_path)
    else:
        infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device, class_names, experiment_name=experiment_name, folder_path = folder_path)

    wandb.finish()
