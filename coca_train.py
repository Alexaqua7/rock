# pretrained CoCa 정의 (위 코드 그대로 사용하되 이미지 임베딩만 사용)
from coca_pytorch.coca_pytorch import CoCa
from vit_pytorch.simple_vit_with_patch_dropout import SimpleViT
from vit_pytorch.extractor import Extractor
import os
import random
import numpy as np
import pandas as pd
import glob
import cv2
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import f1_score, confusion_matrix
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
from coca_pytorch.coca_pytorch import CoCa
from vit_pytorch.extractor import Extractor
import timm

# ⚙️ 설정
CFG = {
    'IMG_SIZE': 224,
    'EPOCHS': 15,
    'LEARNING_RATE': 3e-4,
    'BATCH_SIZE': 128,
    'SEED': 41,
    'NUM_CLASSES': 1000,
    'TEXT_SEQ_LEN': 512,
    'NUM_TOKENS': 20000
}

# 🔧 시드 설정
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# 🖼️ 이미지 전처리 클래스
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

# 🗂️ 커스텀 데이터셋
class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list=None, transforms=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms
        
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        if self.label_list is not None:
            label = self.label_list[index]
            return image, label
        else:
            return image
        
    def __len__(self):
        return len(self.img_path_list)

# 🧠 모델 정의
class DaViTExtractor(nn.Module):
    def __init__(self, model_name='davit_base', pretrained=True):
        super(DaViTExtractor, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='')
        self.embed_dim = self.model.num_features

    def forward(self, x):
        x = self.model.forward_features(x)
        return x  # (batch, seq_len, embed_dim)

# 🧪 검증 함수
def validation(model, criterion, val_loader, device, class_names):
    model.eval()
    val_loss = []
    preds, true_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Validation"):
            imgs = imgs.float().to(device)
            labels = labels.to(device)

            outputs = model(images=imgs, text=torch.randint(0, CFG['NUM_TOKENS'], (imgs.size(0), CFG['TEXT_SEQ_LEN'])).to(device), return_loss=False)
            loss = criterion(outputs, labels)

            preds += outputs.argmax(1).detach().cpu().numpy().tolist()
            true_labels += labels.detach().cpu().numpy().tolist()
            val_loss.append(loss.item())

    _val_loss = np.mean(val_loss)
    _val_score = f1_score(true_labels, preds, average='macro')

    # 🔹 클래스별 F1 점수
    class_f1 = f1_score(true_labels, preds, average=None)
    class_f1_dict = {f'{class_names[i]}_f1': f for i, f in enumerate(class_f1)}

    # 🔹 혼동 행렬
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

# 🏋️‍♂️ 학습 함수
def train(model, optimizer, train_loader, val_loader, scheduler, device, class_names, best_score=0, cur_epoch=1, saved_name="base"):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    best_model = None
    save_path = f"best_{saved_name}.pth"

    for epoch in range(cur_epoch, CFG['EPOCHS'] + 1):
        model.train()
        train_loss = []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{CFG['EPOCHS']}")
        for imgs, labels in progress_bar:
            imgs = imgs.float().to(device)
            labels = labels.to(device)
            texts = torch.randint(0, CFG['NUM_TOKENS'], (imgs.size(0), CFG['TEXT_SEQ_LEN'])).to(device)

            optimizer.zero_grad()
            outputs = model(images=imgs, text=texts, return_loss=False)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            progress_bar.set_postfix(loss=loss.item())

davit = DaViTExtractor()

coca = CoCa(
    dim = 512,
    img_encoder = davit,
    image_dim = 384,
    num_tokens = 20000,
    unimodal_depth = 6,
    multimodal_depth = 6,
    dim_head = 64,
    heads = 8,
    caption_loss_weight = 1.,
    contrastive_loss_weight = 1.,
)

class CoCaClassifier(nn.Module):
    def __init__(self, coca_model, num_classes):
        super().__init__()
        self.coca = coca_model
        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, images):
        _, image_embeds = self.coca(images=images, return_embeddings=True)
        # image_embeds: (B, 512)
        return self.classifier(image_embeds)