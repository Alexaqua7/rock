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

# transformers 라이브러리 추가 (InternImage를 위해)
from transformers import AutoModelForImageClassification

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from tqdm.auto import tqdm

import warnings

warnings.filterwarnings(action='ignore') 

# 설정 변수 정의
CFG = {
    'SEED': 41,
    'IMG_SIZE': 224,
    'BATCH_SIZE': 16,
    'NUM_CLASSES': 7
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

def soft_voting_inference(models, test_img_paths, device, transforms_list, model_types):
    for model in models:
        model.eval()
    
    preds = []
    batch_size = CFG['BATCH_SIZE']
    
    # 배치 단위로 처리
    for i in tqdm(range(0, len(test_img_paths), batch_size)):
        batch_img_paths = test_img_paths[i:i+batch_size]
        
        # 모델별 예측 결과 저장
        outputs = []
        for model_idx, model in enumerate(models):
            # 각 모델에 맞는 transform으로 이미지 로드 및 변환
            batch_imgs = []
            for img_path in batch_img_paths:
                # OpenCV로 이미지 로드 (BGR 형식)
                img = cv2.imread(img_path)
                # 모델별 transform 적용
                transformed_img = transforms_list[model_idx](image=img)['image']
                batch_imgs.append(transformed_img)
            
            # 변환된 이미지들을 배치로 묶어 모델에 입력
            batch_tensor = torch.stack(batch_imgs).to(device)
            
            # 모델 타입에 따라 다른 방식으로 추론
            model_type = model_types[model_idx]
            if model_type == "timm":
                output = model(batch_tensor)
                output = F.softmax(output, dim=1)
            elif model_type == "transformers":
                output = model(batch_tensor)
                # transformers 모델은 output.logits 형태로 결과 반환
                if hasattr(output, 'logits'):
                    output = torch.softmax(output.logits, dim=1)
                else:
                    output = torch.softmax(output, dim=1)
                
            outputs.append(output)
        
        # 예측 확률을 평균
        avg_pred = torch.mean(torch.stack(outputs), dim=0)
        
        # 가장 높은 확률을 가진 클래스 선택
        pred = avg_pred.argmax(1).detach().cpu().numpy().tolist()
        preds += pred
    
    return preds


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    seed_everything(CFG['SEED'])  # Seed 고정

    # 라벨 인코더 준비 (학습 데이터에서 클래스 정보 가져오기)
    all_img_list = glob.glob('./train/*/*')
    df = pd.DataFrame(columns=['img_path', 'rock_type'])
    df['img_path'] = all_img_list
    df['rock_type'] = df['img_path'].apply(lambda x : str(x).replace('\\','/').split('/')[2])
    
    le = preprocessing.LabelEncoder()
    le.fit(df['rock_type'])

    # 모델 정의
    model_names = [
        "mambaout_base_plus_rw.sw_e150_r384_in12k_ft_in1k",  # 첫 번째 모델 이름
        "mambaout_base_plus_rw.sw_e150_r384_in12k_ft_in1k",  # 두 번째 모델 이름, # 세 번째 모델 이름
        "internimage_xl_22kto1k_384",  # 네 번째 모델 이름 (InternImage)
        "internimage_xl_22kto1k_384",
        "internimage_xl_22kto1k_384"
    ]
    
    saved_names = [
        "mambaout_base_plus_rw_sw_e150_r384_in12k_ft_in1k_1-epoch_18.pth",  # 첫 번째 모델 가중치 경로
        "mambaout_base_plus_rw_sw_e150_r384_in12k_ft_in1k_5_kfold5-fold1-epoch_18.pth",  # 두 번째 모델 가중치 경로  # 세 번째 모델 가중치 경로
        "internimage_xl_22kto1k_384_fold_1-epoch20.pth",  # 네 번째 모델 가중치 경로 (InternImage)
        "best_internimage_xl_22kto1k_384_fold_5.pth",  # 두 번째 모델 가중치 경로
        "best_internimage_xl_22kto1k_384_fold_2.pth"
    ]
    
    # 모델 타입 지정 (timm 또는 transformers)
    model_types = ["timm", "timm", "transformers","transformers","transformers"]
    
    # 각각의 모델에 맞는 transform을 설정합니다.
    test_transforms = [
        A.Compose([  # 첫 번째 모델의 transform (384 사이즈, ImageNet normalize)
            A.Resize(384, 384),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ]),
        A.Compose([  # 두 번째 모델의 transform (384 사이즈, 다른 normalize)
            A.Resize(384, 384),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ]),
        A.Compose([  # 네 번째 모델의 transform (384 사이즈, InternImage normalize)
            A.Resize(384, 384),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ]),
         A.Compose([  # 두 번째 모델의 transform (384 사이즈, 다른 normalize)
            A.Resize(384, 384),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ]),
        A.Compose([  # 세 번째 모델의 transform (384 사이즈, 다른 normalize)
            A.Resize(384, 384),
            A.Normalize((0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    ]
    
    # 모델 로드
    models = []
    for model_idx, (model_name, saved_name, model_type) in enumerate(zip(model_names, saved_names, model_types)):
        if model_type == "timm":
            # timm 모델 로드
            model = timm.create_model(model_name, pretrained=False, num_classes=CFG['NUM_CLASSES']).to(device)
            try:
                checkpoint = torch.load(saved_name, map_location=device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                models.append(model)
                print(f"Successfully loaded model: {model_name} from {saved_name}")
            except Exception as e:
                print(f"Failed to load model {model_name}: {e}")
        
        elif model_type == "transformers":
            # transformers 모델 로드 (InternImage)
            if model_name == "internimage_xl_22kto1k_384":
                # InternImage 모델 로드
                model_path = "./weights/OpenGVLab/internimage_xl_22kto1k_384"  # 모델 경로
                
                try:
                    # AutoModelForImageClassification을 통해 모델 로드
                    model = AutoModelForImageClassification.from_pretrained(model_path, trust_remote_code=True)
                    
                    # 모델 head 레이어 재정의 (클래스 수에 맞게)
                    in_features = model.model.head.in_features
                    model.model.head = torch.nn.Linear(in_features, CFG['NUM_CLASSES'])
                    model = model.to(device)
                    
                    # 가중치 로드
                    checkpoint = torch.load(saved_name, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    
                    models.append(model)
                    print(f"Successfully loaded model: {model_name} from {saved_name}")
                except Exception as e:
                    print(f"Failed to load InternImage model: {e}")
    
    # 테스트 데이터 로드
    test = pd.read_csv('./test.csv')
    
    # 경로 변환이 필요하다면 추가 (두 번째 코드에서 사용됨)
    # test['img_path'] = test['img_path'].apply(lambda x: os.path.join("../../", x[2:]))
    
    # Soft voting을 통한 예측
    print("Starting inference with soft voting...")
    with torch.no_grad():
        preds = soft_voting_inference(models, test['img_path'].values, device, test_transforms, model_types)
    
    # 인코딩된 예측값을 원래 클래스 이름으로 변환
    preds_decoded = le.inverse_transform(preds)
    
    # 제출 파일 생성
    submit = pd.read_csv('./sample_submission.csv')
    submit['rock_type'] = preds_decoded
    
    # 저장 경로 설정
    output_path = f"./ensemble_results/soft_voting_submit_{CFG['SEED']}_7_models.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submit.to_csv(output_path, index=False)
    
    print(f"Prediction completed! Saved to {output_path}")