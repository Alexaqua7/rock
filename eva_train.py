import random
import pandas as pd
import numpy as np
import os
import re
import glob
import cv2
import timm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
import torchvision.models as models

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings(action='ignore') 

CFG = {
    'IMG_SIZE':336,
    'EPOCHS':15,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':16,
    'SEED':41
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

def train(model, optimizer, train_loader, val_loader, scheduler, device, best_score=0, cur_epoch=1):
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    
    best_score = 0
    best_model = None
    save_path = "best_eva.pth"

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

        _val_loss, _val_score = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)

        print(f'Epoch [{epoch}], Train Loss: {_train_loss:.5f}, Val Loss: {_val_loss:.5f}, Val Macro F1: {_val_score:.5f}')
        checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'best_score': best_score
    }
        torch.save(checkpoint, f'eva_epoch_{epoch}.pth')
        print(f"Checkpoint at epoch {epoch} saved → model_epoch_{epoch}.pth")
        prev_path = f'eva_epoch_{epoch-1}.pth'
        if epoch > 1 and os.path.exists(prev_path):
            os.remove(prev_path)
            print(f"Previous checkpoint deleted → {prev_path}")

        if scheduler is not None:
            scheduler.step(_val_score)

        if best_score < _val_score:
            best_score = _val_score
            best_model = model

            # 모델 가중치 저장
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved (epoch {epoch}, F1={_val_score:.4f}) → {save_path}")

    return best_model

def validation(model, criterion, val_loader, device):
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
    
    return _val_loss, _val_score




if __name__ == '__main__':
    trained = False
    # try:
    #     trained_path = os.path.join('./', [x for x in os.listdir('./') if x.startswith("model_epoch")][0])
    #     trained = True
    # except:
    #     trained = False



    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    seed_everything(CFG['SEED']) # Seed 고정

    all_img_list = glob.glob('./train/*/*')

    df = pd.DataFrame(columns=['img_path', 'rock_type'])
    df['img_path'] = all_img_list
    df['rock_type'] = df['img_path'].apply(lambda x : str(x).replace('\\','/').split('/')[2])

    train_data, val_data, _, _ = train_test_split(df, df['rock_type'], test_size=0.3, stratify=df['rock_type'], random_state=CFG['SEED'])

    le = preprocessing.LabelEncoder()
    train_data['rock_type'] = le.fit_transform(train_data['rock_type'])
    val_data['rock_type'] = le.transform(val_data['rock_type'])

    # Augmentation 정의
    train_transform = A.Compose([
    PadSquare(value=(0, 0, 0)),
    A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

    test_transform = A.Compose([
        PadSquare(value=(0, 0, 0)),
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
])

    # 데이터셋 및 데이터로더 정의
    train_dataset = CustomDataset(train_data['img_path'].values, train_data['rock_type'].values, train_transform)
    train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=8,pin_memory=True,prefetch_factor=2)

    val_dataset = CustomDataset(val_data['img_path'].values, val_data['rock_type'].values, test_transform)
    val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=8,pin_memory=True,prefetch_factor=2)

    # 모델 정의
    model = timm.create_model('eva_large_patch14_336.in22k_ft_in1k', pretrained=True, num_classes=7)
    optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, threshold_mode='abs', min_lr=1e-8, verbose=True)

    if trained:
        checkpoint = torch.load(trained_path, map_location=device)
        model.to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # 다음 에폭부터 시작
        best_score = checkpoint['best_score']
        infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device, best_score=best_score, cur_epoch=start_epoch)
    else:
        infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device)