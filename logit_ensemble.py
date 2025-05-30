import random
import pandas as pd
import numpy as np
import os
import re
import glob
import cv2
import timm
import pickle

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
    'BATCH_SIZE': 32,
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

def single_model_inference(model, test_img_paths, device, transform, model_type, model_name, save_predictions=True):
    """
    단일 모델로 추론하고 결과를 저장하는 함수
    
    Args:
        model: 추론할 모델
        test_img_paths: 테스트 이미지 경로들
        device: 디바이스 (cuda/cpu)
        transform: 이미지 전처리 변환
        model_type: 모델 타입 ("timm" 또는 "transformers")
        model_name: 모델 이름 (저장 파일명에 사용)
        save_predictions: 예측 결과를 저장할지 여부
    
    Returns:
        predictions: softmax 확률값들
        logits: raw logit 값들
    """
    model.eval()
    
    all_predictions = []
    all_logits = []
    batch_size = CFG['BATCH_SIZE']
    
    # 배치 단위로 처리
    for i in tqdm(range(0, len(test_img_paths), batch_size), desc=f"Inference {model_name}"):
        batch_img_paths = test_img_paths[i:i+batch_size]
        
        # 이미지 로드 및 변환
        batch_imgs = []
        for img_path in batch_img_paths:
            img = cv2.imread(img_path)
            transformed_img = transform(image=img)['image']
            batch_imgs.append(transformed_img)
        
        # 배치 텐서 생성
        batch_tensor = torch.stack(batch_imgs).to(device)
        
        with torch.no_grad():
            # 모델 타입에 따라 다른 방식으로 추론
            if model_type == "timm":
                logits = model(batch_tensor)
                predictions = F.softmax(logits, dim=1)
            elif model_type == "transformers":
                output = model(batch_tensor)
                if hasattr(output, 'logits'):
                    logits = output.logits
                else:
                    logits = output
                predictions = F.softmax(logits, dim=1)
            
            # CPU로 이동하여 저장
            all_predictions.append(predictions.cpu())
            all_logits.append(logits.cpu())
    
    # 모든 배치 결과 합치기
    all_predictions = torch.cat(all_predictions, dim=0)
    all_logits = torch.cat(all_logits, dim=0)
    
    # 예측 결과 저장
    if save_predictions:
        # 디렉토리 생성
        os.makedirs('./model_predictions', exist_ok=True)
        
        # softmax 확률값 저장
        prob_save_path = f'./model_predictions/{model_name}_probabilities.npy'
        np.save(prob_save_path, all_predictions.numpy())
        
        # logit 값 저장
        logit_save_path = f'./model_predictions/{model_name}_logits.npy'
        np.save(logit_save_path, all_logits.numpy())
        
        print(f"Saved predictions to {prob_save_path}")
        print(f"Saved logits to {logit_save_path}")
    
    return all_predictions.numpy(), all_logits.numpy()

def model_group_inference(model_configs, test_img_paths, device):
    """
    모델 그룹별로 추론을 수행하는 함수
    
    Args:
        model_configs: 모델 설정 리스트 (각각 model_name, saved_name, model_type, transform 포함)
        test_img_paths: 테스트 이미지 경로들
        device: 디바이스
    """
    
    # 모델별로 그룹핑
    model_groups = {}
    for config in model_configs:
        base_model_name = config['model_name'].split('_')[0]  # 'davit', 'mambaout', 'internimage' 등
        if base_model_name not in model_groups:
            model_groups[base_model_name] = []
        model_groups[base_model_name].append(config)
    
    # 각 그룹별로 추론 수행
    for group_name, configs in model_groups.items():
        print(f"\n=== Processing {group_name.upper()} models ===")
        
        for i, config in enumerate(configs):
            model_name = config['model_name']
            saved_name = config['saved_name']
            model_type = config['model_type']
            transform = config['transform']
            
            # 고유한 모델 식별자 생성
            unique_model_name = f"{group_name}_{i+1}_{os.path.basename(saved_name).split('.')[0]}"
            
            print(f"Loading model: {model_name} from {saved_name}")
            
            # 모델 로드
            if model_type == "timm":
                model = timm.create_model(model_name, pretrained=False, num_classes=CFG['NUM_CLASSES']).to(device)
                try:
                    checkpoint = torch.load(saved_name, map_location=device)
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                    print(f"Successfully loaded {model_name}")
                except Exception as e:
                    print(f"Failed to load model {model_name}: {e}")
                    continue
                    
            elif model_type == "transformers":
                if model_name == "internimage_xl_22kto1k_384":
                    model_path = "./weights/OpenGVLab/internimage_xl_22kto1k_384"
                    try:
                        model = AutoModelForImageClassification.from_pretrained(model_path, trust_remote_code=True)
                        in_features = model.model.head.in_features
                        model.model.head = torch.nn.Linear(in_features, CFG['NUM_CLASSES'])
                        model = model.to(device)
                        
                        checkpoint = torch.load(saved_name, map_location=device)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        print(f"Successfully loaded {model_name}")
                    except Exception as e:
                        print(f"Failed to load InternImage model: {e}")
                        continue
            
            # 추론 수행 및 저장
            predictions, logits = single_model_inference(
                model, test_img_paths, device, transform, model_type, unique_model_name
            )
            
            # 메모리 정리
            del model
            torch.cuda.empty_cache()

def load_and_ensemble_predictions(model_group_names, ensemble_method='soft_voting', model_weights=None, weight_strategy='uniform'):
    """
    저장된 예측 결과들을 불러와서 가중치 기반 앙상블하는 함수
    
    Args:
        model_group_names: 앙상블할 모델 그룹 이름들
        ensemble_method: 앙상블 방법 ('soft_voting', 'hard_voting', 'weighted_soft_voting')
        model_weights: 모델별 가중치 딕셔너리 {model_file_name: weight}
        weight_strategy: 가중치 전략 ('uniform', 'manual', 'performance_based', 'group_based')
    
    Returns:
        ensemble_predictions: 앙상블 예측 결과
        weight_info: 사용된 가중치 정보
    """
    
    all_predictions = []
    model_names = []
    
    # 각 모델 그룹의 예측 결과 로드
    for group_name in model_group_names:
        prediction_files = glob.glob(f'./model_predictions/{group_name}_*_probabilities.npy')
        
        for pred_file in prediction_files:
            predictions = np.load(pred_file)
            all_predictions.append(predictions)
            model_name = os.path.basename(pred_file).replace('_probabilities.npy', '')
            model_names.append(model_name)
            print(f"Loaded predictions from {pred_file}, shape: {predictions.shape}")
    
    if not all_predictions:
        raise ValueError("No prediction files found!")
    
    all_predictions = np.array(all_predictions)  # (num_models, num_samples, num_classes)
    
    # 가중치 계산
    weights = calculate_weights(model_names, weight_strategy, model_weights)
    
    # 가중치 정규화
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    
    print(f"\nUsing weights: {dict(zip(model_names, weights))}")
    
    # 앙상블 수행
    if ensemble_method in ['soft_voting', 'weighted_soft_voting']:
        # 가중치 적용된 Soft voting
        weighted_predictions = np.zeros_like(all_predictions[0])
        for i, weight in enumerate(weights):
            weighted_predictions += weight * all_predictions[i]
        ensemble_predictions = np.argmax(weighted_predictions, axis=1)
        
    elif ensemble_method == 'hard_voting':
        # Hard voting with weights
        individual_preds = np.argmax(all_predictions, axis=2)  # (num_models, num_samples)
        ensemble_predictions = []
        
        for sample_idx in range(individual_preds.shape[1]):
            votes = individual_preds[:, sample_idx]
            weighted_votes = {}
            
            for model_idx, vote in enumerate(votes):
                if vote not in weighted_votes:
                    weighted_votes[vote] = 0
                weighted_votes[vote] += weights[model_idx]
            
            # 가중치가 가장 높은 클래스 선택
            ensemble_predictions.append(max(weighted_votes, key=weighted_votes.get))
        
        ensemble_predictions = np.array(ensemble_predictions)
    
    weight_info = {
        'model_names': model_names,
        'weights': weights.tolist(),
        'weight_strategy': weight_strategy
    }
    
    return ensemble_predictions, weight_info

def calculate_weights(model_names, weight_strategy, model_weights=None):
    """
    다양한 전략에 따라 모델 가중치를 계산하는 함수
    
    Args:
        model_names: 모델 이름 리스트
        weight_strategy: 가중치 전략
        model_weights: 수동 가중치 딕셔너리
    
    Returns:
        weights: 계산된 가중치 리스트
    """
    
    if weight_strategy == 'uniform':
        # 균등 가중치
        return [1.0] * len(model_names)
    
    elif weight_strategy == 'manual' and model_weights:
        # 수동 지정 가중치
        weights = []
        for name in model_names:
            weight = model_weights.get(name, 1.0)  # 기본값 1.0
            weights.append(weight)
        return weights
    
    elif weight_strategy == 'performance_based':
        # 성능 기반 가중치 (예시 - 실제로는 validation 성능을 사용해야 함)
        performance_weights = {
            'internimage': 2.0,  # InternImage 모델들에 높은 가중치
            'mambaout': 1.5,     # MambaOut 모델들에 중간 가중치
            'davit': 1.2         # DaviT 모델들에 기본 가중치
        }
        
        weights = []
        for name in model_names:
            # 모델 타입 추출
            model_type = name.split('_')[0]
            weight = performance_weights.get(model_type, 1.0)
            weights.append(weight)
        return weights
    
    elif weight_strategy == 'group_based':
        # 그룹별 가중치
        group_weights = {
            'internimage': 1.5,
            'mambaout': 1.2,
            'davit': 0.9
        }
        
        weights = []
        for name in model_names:
            model_type = name.split('_')[0]
            weight = group_weights.get(model_type, 1.0)
            weights.append(weight)
        return weights
    
    else:
        # 기본값: 균등 가중치
        return [1.0] * len(model_names)

def advanced_ensemble_strategies(model_group_names, ensemble_configs):
    """
    고급 앙상블 전략들을 수행하는 함수
    
    Args:
        model_group_names: 모델 그룹 이름들
        ensemble_configs: 앙상블 설정 리스트
    """
    
    results = {}
    
    for config in ensemble_configs:
        print(f"\n=== {config['name']} ===")
        
        predictions, weight_info = load_and_ensemble_predictions(
            model_group_names=config.get('groups', model_group_names),
            ensemble_method=config.get('method', 'soft_voting'),
            model_weights=config.get('weights', None),
            weight_strategy=config.get('weight_strategy', 'uniform')
        )
        
        results[config['name']] = {
            'predictions': predictions,
            'weight_info': weight_info
        }
        
        print(f"Weight strategy: {weight_info['weight_strategy']}")
        print(f"Model weights: {dict(zip(weight_info['model_names'], weight_info['weights']))}")
    
    return results

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    seed_everything(CFG['SEED'])

    # 라벨 인코더 준비
    all_img_list = glob.glob('./train/*/*')
    df = pd.DataFrame(columns=['img_path', 'rock_type'])
    df['img_path'] = all_img_list
    df['rock_type'] = df['img_path'].apply(lambda x : str(x).replace('\\','/').split('/')[2])
    
    le = preprocessing.LabelEncoder()
    le.fit(df['rock_type'])

    # 모델 설정 정의
    model_configs = [
        # DaviT 모델들
        {
            'model_name': 'davit_base',
            'saved_name': 'davit_base_29_fold0-best.pth',
            'model_type': 'timm',
            'transform': A.Compose([
                A.Resize(224, 224),
                A.Normalize((0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2()
            ])
        },
        {
            'model_name': 'davit_base',
            'saved_name': 'davit_base_29_fold4-best.pth',
            'model_type': 'timm',
            'transform': A.Compose([
                A.Resize(224, 224),
                A.Normalize((0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2()
            ])
        },
        {
            'model_name': 'davit_base',
            'saved_name': 'davit_base_29_fold3-best.pth',
            'model_type': 'timm',
            'transform': A.Compose([
                A.Resize(224, 224),
                A.Normalize((0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2()
            ])
        },
        {
            'model_name': 'davit_base',
            'saved_name': './experiments/davit_base_29_fold2/davit_base_29_fold2-best.pth',
            'model_type': 'timm',
            'transform': A.Compose([
                A.Resize(224, 224),
                A.Normalize((0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2()
            ])
        },
        {
            'model_name': 'davit_base',
            'saved_name': 'davit_base_29_fold1-best.pth',
            'model_type': 'timm',
            'transform': A.Compose([
                A.Resize(224, 224),
                A.Normalize((0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2()
            ])
        },
        # MambaOut 모델들
        {
            'model_name': 'mambaout_base_plus_rw.sw_e150_r384_in12k_ft_in1k',
            'saved_name': 'mambaout_base_plus_rw_sw_e150_r384_in12k_ft_in1k_1-epoch_18.pth',
            'model_type': 'timm',
            'transform': A.Compose([
                A.Resize(384, 384),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2()
            ])
        },
        {
            'model_name': 'mambaout_base_plus_rw.sw_e150_r384_in12k_ft_in1k',
            'saved_name': 'mambaout_base_plus_rw_sw_e150_r384_in12k_ft_in1k_5_kfold5-fold1-epoch_18.pth',
            'model_type': 'timm',
            'transform': A.Compose([
                A.Resize(384, 384),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2()
            ])
        },
        {
            'model_name': 'mambaout_base_plus_rw.sw_e150_r384_in12k_ft_in1k',
            'saved_name': 'mambaout_base_plus_rw_sw_e150_r384_in12k_ft_in1k_1_fold2-best.pth',
            'model_type': 'timm',
            'transform': A.Compose([
                A.Resize(384, 384),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2()
            ])
        },
        # InternImage 모델들
        {
            'model_name': 'internimage_xl_22kto1k_384',
            'saved_name': 'internimage_xl_22kto1k_384_fold_1-epoch20.pth',
            'model_type': 'transformers',
            'transform': A.Compose([
                A.Resize(384, 384),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        },
        {
            'model_name': 'internimage_xl_22kto1k_384',
            'saved_name': 'best_internimage_xl_22kto1k_384_fold_5.pth',
            'model_type': 'transformers',
            'transform': A.Compose([
                A.Resize(384, 384),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        },
        {
            'model_name': 'internimage_xl_22kto1k_384',
            'saved_name': 'best_internimage_xl_22kto1k_384_fold_2.pth',
            'model_type': 'transformers',
            'transform': A.Compose([
                A.Resize(384, 384),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        },
        {
            'model_name': 'internimage_xl_22kto1k_384',
            'saved_name': 'best_internimage_xl_22kto1k_384_fold_3_new.pth',
            'model_type': 'transformers',
            'transform': A.Compose([
                A.Resize(384, 384),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        },
        {
            'model_name': 'internimage_xl_22kto1k_384',
            'saved_name': 'best_internimage_xl_22kto1k_384_fold_4.pth',
            'model_type': 'transformers',
            'transform': A.Compose([
                A.Resize(384, 384),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        }
    ]

    # 테스트 데이터 로드
    test = pd.read_csv('./test.csv')

    
    
    # 1단계: 각 모델별로 추론 수행 및 저장
    print("=== Stage 1: Individual Model Inference ===")
    model_group_inference(model_configs, test['img_path'].values, device)
    
    
    # 2단계: 다양한 가중치 전략으로 앙상블 수행
    print("\n=== Stage 2: Weighted Ensemble Predictions ===")
    
    # 앙상블 설정들
    ensemble_configs = [
        {
            'name': 'Performance_Based_Ensemble',
            'groups': ['davit', 'mambaout', 'internimage'],
            'method': 'soft_voting',
            'weight_strategy': 'performance_based'
        }
    ]
    
    # 고급 앙상블 수행
    ensemble_results = advanced_ensemble_strategies(['davit', 'mambaout', 'internimage'], ensemble_configs)
    
    # 결과 저장
    os.makedirs('./ensemble_results', exist_ok=True)
    submit_template = pd.read_csv('./sample_submission.csv')
    
    for ensemble_name, result in ensemble_results.items():
        predictions = result['predictions']
        weight_info = result['weight_info']
        
        # 예측 결과 디코딩 및 저장
        preds_decoded = le.inverse_transform(predictions)
        submit = submit_template.copy()
        submit['rock_type'] = preds_decoded
        
        # CSV 파일 저장
        output_path = f'./ensemble_results/{ensemble_name}.csv'
        submit.to_csv(output_path, index=False)
        
        # 가중치 정보 저장
        weight_info_path = f'./ensemble_results/{ensemble_name}_weights.txt'
        with open(weight_info_path, 'w') as f:
            f.write(f"Ensemble Strategy: {weight_info['weight_strategy']}\n")
            f.write(f"Model Weights:\n")
            for model, weight in zip(weight_info['model_names'], weight_info['weights']):
                f.write(f"  {model}: {weight:.4f}\n")
        
        print(f"{ensemble_name} predictions and weights saved!")