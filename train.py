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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import torch.nn.functional as F
from loss import FocalLoss, weighted_normalized_CrossEntropyLoss, CenterLoss, CombinedLoss

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 클래스별 가중치 텐서 or None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)  # 예측 확률 (logit을 softmax로 변환한 것)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
import warnings
warnings.filterwarnings(action='ignore')

CFG = {
    'IMG_SIZE': 224,
    'EPOCHS': 15,
    'LEARNING_RATE': 3e-4,
    'BATCH_SIZE': 128,
    'SEED': 41
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
        return ("border_mode", "value")

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

def train(model, optimizer, train_loader, val_loader, scheduler, device, class_names, best_score=0, cur_epoch=1, saved_name="base"):
    model.to(device)
    weights = weighted_normalized_CrossEntropyLoss(return_weights=True).to(device)
    weighted_CE = weighted_normalized_CrossEntropyLoss(return_weights=False).to(device)
    center_loss = CenterLoss(num_classes=len(class_names), feat_dim=feature_dim, device=device)  # feature_dim은 모델 임베딩 차원
    criterion = CombinedLoss(focal, center_loss, lambda_c=0.01).to(device)

    
    best_model = None
    save_path = f"best_{saved_name}.pth"

    for epoch in range(cur_epoch, CFG['EPOCHS'] + 1):
        model.train()
        train_loss = []

        progress_bar = tqdm(iter(train_loader), desc=f"Epoch {epoch}/{CFG['EPOCHS']}")
        for imgs, labels in progress_bar:
            imgs = imgs.float().to(device)
            labels = labels.to(device)

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
        torch.save(checkpoint, f'{saved_name}_epoch_{epoch}.pth')
        print(f"Checkpoint at epoch {epoch} saved → {saved_name}_epoch_{epoch}.pth")

        prev_path = f'{saved_name}_epoch_{epoch-1}.pth'
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
            labels = labels.to(device)

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
    trained = False
    # trained_path = "./best_davit_base.pth"
    model_name = "davit_base"
    saved_name = f"{model_name.replace('.','_')}__FocalLoss_gamma_2.0_alpha_weights"
    wandb.init(
        project="rock-classification",
        config=CFG,
        name=f"{model_name}_FocalLoss_gamma_2.0_alpha_weights",
        # resume='must',
        # id="38lkdqqs"
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    seed_everything(CFG['SEED'])

    all_img_list = glob.glob('./train/*/*')
    df = pd.DataFrame(columns=['img_path', 'rock_type'])
    df['img_path'] = all_img_list
    df['rock_type'] = df['img_path'].apply(lambda x : str(x).replace('\\','/').split('/')[2])

    train_data, val_data, _, _ = train_test_split(df, df['rock_type'], test_size=0.3, stratify=df['rock_type'], random_state=CFG['SEED'])

    le = preprocessing.LabelEncoder()
    train_data['rock_type'] = le.fit_transform(train_data['rock_type'])
    val_data['rock_type'] = le.transform(val_data['rock_type'])

    class_names = le.classes_

    train_transform = A.Compose([
    PadSquare(value=(0, 0, 0)),
    A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
    A.HorizontalFlip(p=0.5),  # 50% 확률로 좌우 반전
    A.VerticalFlip(p=0.5),    # 50% 확률로 상하 반전
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])
    test_transform = A.Compose([
        PadSquare(value=(0, 0, 0)),
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    train_dataset = CustomDataset(train_data['img_path'].values, train_data['rock_type'].values, train_transform)
    train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2)

    val_dataset = CustomDataset(val_data['img_path'].values, val_data['rock_type'].values, test_transform)
    val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=2)

    model = timm.create_model(model_name, pretrained=True, num_classes=len(class_names))
    optimizer = torch.optim.Adam(params=model.parameters(), lr=CFG["LEARNING_RATE"])
    # # Warmup 스케줄러 정의 (선형적으로 0부터 LEARNING_RATE까지 증가)
    # warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=5)

    # # Cosine Annealing LR 스케줄러 정의
    # cosine_scheduler = CosineAnnealingLR(optimizer, T_max=CFG['EPOCHS'] - 5, eta_min=1e-8)

    # # SequentialLR로 결합
    # scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[5])
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG['EPOCHS'], eta_min=1e-8)

    wandb.config.update({
        "optimizer": optimizer.__class__.__name__,
        "scheduler": scheduler.__class__.__name__,
        "model": model_name,
        'Loss': "Cross Entropy Loss with Normalized[1,2] weights"
    })

    if trained:
        checkpoint = torch.load(trained_path, map_location=device)
        model.to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_score = checkpoint['best_score']
        infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device, class_names, best_score=best_score, cur_epoch=start_epoch,saved_name=saved_name)
    else:
        infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device, class_names, saved_name=saved_name)

    wandb.finish()
    # import timm

    # for model in timm.list_models()[:1000]:
    #     print(model)