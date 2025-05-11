import random
import pandas as pd
import numpy as np
import os
import re
import glob
import cv2
import timm
import wandb
import yaml
import argparse

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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
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

# Default config (will be overridden by sweep)
CFG = {
    'IMG_SIZE': 448,
    'EPOCHS': 15,
    'LEARNING_RATE': 3e-4,
    'BATCH_SIZE': 8,
    'ACCUMULATION_STEPS': 8,
    'MIN_LR': 1e-8,
    'SEED': 41
}

# # Default sweep configuration
# DEFAULT_SWEEP_CONFIG = {
#     'method': 'bayes',
#     'metric': {
#         'name': 'val_macro_f1',
#         'goal': 'maximize'
#     },
#     'parameters': {
#         'learning_rate': {
#             'min': 1e-5,
#             'max': 1e-3,
#             'distribution': 'log_uniform'
#         },
#         'weight_decay': {
#             'min': 1e-6,
#             'max': 1e-2,
#             'distribution': 'log_uniform'
#         },
#         'model_name': {
#             'value': 'davit_base'
#         },
#         'img_size': {
#             'value': 224
#         },
#         'batch_size': {
#             'value': 32
#         },
#         'epochs': {
#             'value': 15
#         },
#         'seed': {
#             'value': 41
#         }
#     }
# }

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

def train(model, optimizer, train_loader, val_loader, scheduler, device, class_names, best_score=0, cur_epoch=1, experiment_name="base", folder_path = "base"):
    model.to(device)
    criterion = weighted_normalized_CrossEntropyLoss(return_weights=False).to(device)
    
    best_model = None
    save_path = os.path.join(folder_path, f"{experiment_name}-best.pth")

    for epoch in range(cur_epoch, 2):
        model.train()
        train_loss = []

        progress_bar = tqdm(iter(train_loader), desc=f"Epoch {epoch}/{CFG['EPOCHS']}")
        for imgs, labels in progress_bar:
            imgs = imgs.float().to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            progress_bar.set_postfix(loss=loss.item())

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

    return best_model, best_score

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

def train_single_run():
    """Function to run a single training with wandb config"""
    # Initialize a wandb run
    wandb.init()
    
    # Update config with wandb values
    CFG['LEARNING_RATE'] = wandb.config.LEARNING_RATE
    CFG['MIN_LR'] = wandb.config.MIN_LR
    model_name = 'vit_so150m2_patch16_reg1_gap_448.sbb_e200_in12k_ft_in1k'
    
    test_size = 0.3
    
    # Create experiment name based on sweep parameters
    experiment_name = f"{model_name.replace('.','_')}_lr{CFG['LEARNING_RATE']:.2e}_acc{CFG['ACCUMULATION_STEPS']}"
    folder_path = os.path.join("./experiments", experiment_name)
    os.makedirs(folder_path, exist_ok=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    seed_everything(CFG['SEED'])

    all_img_list = glob.glob('./train/*/*')
    df = pd.DataFrame(columns=['img_path', 'rock_type'])
    df['img_path'] = all_img_list
    df['rock_type'] = df['img_path'].apply(lambda x : str(x).replace('\\','/').split('/')[2])

    df, _, _, _ = train_test_split(df, df['rock_type'], test_size=0.98, stratify=df['rock_type'], random_state=CFG['SEED'])
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
    train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)

    val_dataset = CustomDataset(val_data['img_path'].values, val_data['rock_type'].values, test_transform)
    val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)

    model = timm.create_model(model_name, pretrained=True, num_classes=len(class_names))
    
    # Use AdamW with weight decay from sweep
    optimizer = torch.optim.AdamW(
        params=model.parameters(), 
        lr=CFG["LEARNING_RATE"],
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG['EPOCHS'], eta_min=CFG['MIN_LR'])

    wandb.config.update({
        "optimizer": optimizer.__class__.__name__,
        "scheduler": scheduler.__class__.__name__,
        "model": model_name
    })

    # Save config
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
    
    config_path = os.path.join(folder_path, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    # Train the model
    _, best_score = train(model, optimizer, train_loader, val_loader, scheduler, device, class_names, experiment_name=experiment_name, folder_path=folder_path)
    
    # Return final score for the sweep
    wandb.run.summary["best_val_macro_f1"] = best_score

def main():
    parser = argparse.ArgumentParser(description='Run training with wandb sweep or single run')
    parser.add_argument('--mode', type=str, choices=['sweep', 'single'], default='sweep', help='Run mode: sweep or single training')
    parser.add_argument('--project', type=str, default='vit_so150m2_sweep', help='wandb project name')
    parser.add_argument('--entity', type=str, default='alexseo-inha-university', help='wandb entity name')
    parser.add_argument('--count', type=int, default=15, help='number of sweep runs to execute')
    parser.add_argument('--config', type=str, default="sweep.yaml", help='sweep configuration file (optional)')
    args = parser.parse_args()
    
    if args.mode == 'sweep':
        # Load sweep configuration from file or use default
        if args.config and os.path.exists(args.config):
            with open(args.config, 'r') as file:
                sweep_config = yaml.safe_load(file)
        else:
            print("Using default sweep configuration")
            # sweep_config = DEFAULT_SWEEP_CONFIG
            
            # Save the default config to a file for reference
            with open('default_sweep_config.yaml', 'w') as file:
                yaml.dump(sweep_config, file)
            
        # Add program field to sweep config
        sweep_config['program'] = __file__
        
        # Initialize sweep
        sweep_id = wandb.sweep(sweep_config, project=args.project, entity=args.entity)
        
        print(f"Sweep initialized with ID: {sweep_id}")
        print(f"View sweep at: https://wandb.ai/{args.entity}/{args.project}/sweeps/{sweep_id}")
        
        # Start the sweep agent
        wandb.agent(sweep_id, function=train_single_run, count=args.count)
        
    elif args.mode == 'single':
        # For single run mode, use default config
        wandb.init(project=args.project, entity=args.entity)
        wandb.config.update({
            'learning_rate': CFG['LEARNING_RATE'],
            'img_size': CFG['IMG_SIZE'],
            'epochs': CFG['EPOCHS'],
            'min_lr': CFG['MIN_LR'],
            'accumulation_steps': CFG['ACCUMULATION_STEPS'],
            'seed': CFG['SEED'],
            'model_name': 'vit_so400m_patch14_siglip_378.v2_webli'
        })
        train_single_run()

if __name__ == '__main__':
    main()