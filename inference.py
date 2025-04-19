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
    'IMG_SIZE':224,
    'EPOCHS':5,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':32,
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

def inference(model, test_loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs in tqdm(iter(test_loader)):
            imgs = imgs.float().to(device)
            
            pred = model(imgs)
            
            preds += pred.argmax(1).detach().cpu().numpy().tolist()
    
    preds = le.inverse_transform(preds)
    return preds



if __name__ == '__main__':
    model_name = "convnextv2_base.fcmae_ft_in22k_in1k_384"
    saved_name = f"{model_name.replace('.','_')}"
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

    test_transform = A.Compose([
        PadSquare(value=(0, 0, 0)),
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
])

    # 데이터셋 및 데이터로더 정의
    test = pd.read_csv('./test.csv')

    test_dataset = CustomDataset(test['img_path'].values, None, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

    # 모델 정의
    model = timm.create_model(model_name, pretrained=True, num_classes=7).to(device)
    checkpoint = torch.load(f'best_{saved_name}.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    preds = inference(model, test_loader, device)
    submit = pd.read_csv('./sample_submission.csv')

    submit['rock_type'] = preds

    submit.to_csv(f"./{saved_name}_submit_{checkpoint['epoch']}epoch.csv", index=False)