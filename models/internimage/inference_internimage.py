import glob
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
from PIL import Image
from transformers import AutoModelForImageClassification, CLIPImageProcessor
import numpy as np
import os
import random
from tqdm import tqdm
from torch.cuda.amp import autocast as amp_autocast, GradScaler
from contextlib import suppress
import time
from albumentations.core.transforms_interface import ImageOnlyTransform
import wandb
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


CFG = {
    'IMG_SIZE': 384,
    'BATCH_SIZE': 8,
    'SEED': 41
}

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

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if norm_type == torch.inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]), norm_type
        )
    return total_norm

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

@torch.no_grad()
def inference(data_loader, model, device):
    model.eval()
    preds = []

    progress_bar = tqdm(iter(data_loader), desc="Validation")
    for idx, images in enumerate(progress_bar):
        if type(images) == list:
            images = [item.float().to(device) for item in images]
        else:
            images = images.float().to(device)
        output = model(images)
        _, predicted = torch.max(output.logits, 1)
        preds.extend(predicted.cpu().numpy().tolist())
    
    preds = le.inverse_transform(preds)

    return preds

if __name__ == '__main__':
    import cv2  # PadSquare와 CustomDataset에서 사용
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    trained_path = './best_internimage_l_22kto1k_384.pth'
    model_name = "../../../weights/OpenGVLab/internimage_l_22kto1k_384"
    saved_name = "internimage_l_22kto1k_384"
    model = AutoModelForImageClassification.from_pretrained(model_name, trust_remote_code=True)

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

    num_classes = len(class_names)

    # 모델 head 레이어 재정의
    in_features = model.model.head.in_features
    model.model.head = torch.nn.Linear(in_features, num_classes)
    model = model.to(device)

    test_transform = A.Compose([
        PadSquare(value=(0, 0, 0)),
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    test = pd.read_csv('../../test.csv')

    test_dataset = CustomDataset(test['img_path'].values, None, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

    checkpoint = torch.load(trained_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    preds = inference(test_loader, model, device)
    submit = pd.read_csv('../../sample_submission.csv')

    submit['rock_type'] = preds

    submit.to_csv(f"./{saved_name}_submit_{checkpoint['epoch']}epoch.csv", index=False)