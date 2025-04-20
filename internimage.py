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
    'EPOCHS': 15,
    'LEARNING_RATE': 2e-5,
    'BATCH_SIZE': 8,
    'SEED': 41,
    'AMP_TYPE': 'bfloat16',  # 예시로 추가
    'AMP_OPT_LEVEL': 'O1',  # 예시로 추가
    'TRAIN': {
        'ACCUMULATION_STEPS': 1, # 예시로 추가
        'CLIP_GRAD': None # 예시로 추가
    }
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

def train_one_epoch(CFG,
                    model,
                    criterion,
                    data_loader,
                    optimizer,
                    epoch,
                    mixup_fn,
                    lr_scheduler,
                    amp_autocast=suppress,
                    loss_scaler=None,
                    model_ema=None):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    amp_type = torch.float16 if CFG['AMP_TYPE'] == 'float16' else torch.bfloat16
    progress_bar = tqdm(enumerate(data_loader), total=num_steps, desc=f"Epoch {epoch}/{CFG['EPOCHS']}")
    start_time = time.time()
    epoch_loss = 0.0

    for idx, (samples, targets) in progress_bar:
        step_start_time = time.time()

        if type(samples) == list:
            samples = [item.cuda(non_blocking=True) for item in samples]
        else:
            samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.amp.autocast('cuda', dtype=amp_type):
            outputs = model(samples)
            loss = criterion(outputs['logits'], targets)

        if CFG['TRAIN']['ACCUMULATION_STEPS'] > 1:
            loss = loss / CFG['TRAIN']['ACCUMULATION_STEPS']
            if CFG['AMP_OPT_LEVEL'] != 'O0':
                loss_scaler.scale(loss).backward()
                if (idx + 1) % CFG['TRAIN']['ACCUMULATION_STEPS'] == 0:
                    if CFG['TRAIN']['CLIP_GRAD']:
                        loss_scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), CFG['TRAIN']['CLIP_GRAD'])
                    loss_scaler.step(optimizer)
                    loss_scaler.update()
                    optimizer.zero_grad()
                    if model_ema is not None:
                        model_ema.update(model)
            else:
                loss.backward()
                if (idx + 1) % CFG['TRAIN']['ACCUMULATION_STEPS'] == 0:
                    if CFG['TRAIN']['CLIP_GRAD']:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), CFG['TRAIN']['CLIP_GRAD'])
                    optimizer.step()
                    optimizer.zero_grad()
                    if model_ema is not None:
                        model_ema.update(model)
        else:
            if CFG['AMP_OPT_LEVEL'] != 'O0':
                loss_scaler.scale(loss).backward()
                if CFG['TRAIN']['CLIP_GRAD']:
                    loss_scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), CFG['TRAIN']['CLIP_GRAD'])
                loss_scaler.step(optimizer)
                loss_scaler.update()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            else:
                loss.backward()
                if CFG['TRAIN']['CLIP_GRAD']:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), CFG['TRAIN']['CLIP_GRAD'])
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        epoch_loss += loss.item() * targets.size(0)
        elapsed_time = time.time() - start_time
        steps_per_sec = (idx + 1) / elapsed_time
        eta = (num_steps - (idx + 1)) / steps_per_sec
        progress_bar.set_postfix(loss=f"{loss.item():.4f}",
                                 elapsed=f"{time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}",
                                 eta=f"{time.strftime('%H:%M:%S', time.gmtime(eta))}")

        torch.cuda.synchronize()

    lr_scheduler.step() # 에폭 종료 시 스케줄러 업데이트
    return epoch_loss / len(data_loader.dataset)

@torch.no_grad()
def validation(CFG, criterion, data_loader, model, device, class_names, epoch=None):
    model.eval()
    total_loss = 0.0
    preds, true_labels = [], []
    total = 0

    progress_bar = tqdm(iter(data_loader), desc="Validation")

    for idx, (images, target) in enumerate(progress_bar):
        if type(images) == list:
            images = [item.float().to(device) for item in images]
        else:
            images = images.float().to(device)
        target = target.to(device)
        output = model(images)
        loss = criterion(output['logits'], target)

        total_loss += loss.item() * target.size(0)
        _, predicted = torch.max(output.logits, 1)
        total += target.size(0)
        preds.extend(predicted.cpu().numpy().tolist())
        true_labels.extend(target.cpu().numpy().tolist())

        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    _val_loss = total_loss / total
    _val_score = f1_score(true_labels, preds, average='macro', zero_division=0)

    # 🔹 class별 F1 (클래스명 포함)
    class_f1 = f1_score(true_labels, preds, average=None, zero_division=0)
    class_f1_dict = {f'{class_names[i]}_f1': f for i, f in enumerate(class_f1)}

    # 🔹 confusion matrix
    cm = confusion_matrix(true_labels, preds)
    fig, ax = plt.subplots(figsize=(len(class_names), len(class_names)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title("Confusion Matrix (Epoch {epoch})".format(epoch=epoch) if epoch is not None else "Confusion Matrix")
    plt.tight_layout()

    # 🔹 wandb 이미지로 저장
    wandb_cm = wandb.Image(fig)
    plt.close(fig)

    return _val_loss, _val_score, class_f1_dict, wandb_cm


def train(CFG, model, criterion, train_loader, val_loader, optimizer, lr_scheduler, scaler, mixup_fn, class_names, model_ema=None, saved_name='model'):
    best_val_score = 0.0 # Validation 점수 기준으로 저장하기 위한 변수

    for epoch in range(1, CFG['EPOCHS']+1):
        train_loss = train_one_epoch(CFG, model, criterion, train_loader, optimizer, epoch, mixup_fn, lr_scheduler, amp_autocast, scaler, model_ema)
        _val_loss, _val_score, class_f1_dict, wandb_cm = validation(CFG, criterion, val_loader, model, device, class_names, epoch)
        print(f"Epoch {epoch}/{CFG['EPOCHS']} - Train Loss: {train_loss:.4f}, Val Loss: {_val_loss:.4f}, macro f1: {_val_score:.2f}%")

        # WandB로 metric 로깅
        log_data = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': _val_loss,
            'val_macro_f1': _val_score,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'confusion_matrix': wandb_cm
        }
        log_data.update(class_f1_dict)
        wandb.log(log_data)

        # Validation 점수 기준으로 최고 모델 저장
        if _val_score > best_val_score:
            best_val_score = _val_score
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'best_val_accuracy': best_val_score
            }, f"./best_{saved_name}.pth")

        # 마지막 에폭 모델 저장
        if epoch > 1:
            os.remove(f"./{saved_name}_epoch{epoch - 1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict()
        }, f"./{saved_name}_epoch{epoch}.pth")

    wandb.finish()


if __name__ == '__main__':
    import cv2  # PadSquare와 CustomDataset에서 사용
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    trained = False
    trained_path = './checkpoint.pth'
    model_name = "../weights/OpenGVLab/internimage_l_22kto1k_384"
    saved_name = "internimage_l_22kto1k_384"
    model = AutoModelForImageClassification.from_pretrained(model_name, trust_remote_code=True)

    wandb.init(
        project="rock-classification",
        config=CFG,
        name=f"{saved_name}",
        # resume='must',
        # id="38lkdqqs"
    )

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

    optimizer = torch.optim.Adam(params=model.parameters(), lr=CFG["LEARNING_RATE"])
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG['EPOCHS'], eta_min=1e-8)
    scaler = GradScaler()

    if trained:
        checkpoint = torch.load(trained_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    train(CFG, model, torch.nn.CrossEntropyLoss(), train_loader, val_loader, optimizer, scheduler, scaler, mixup_fn=None, class_names=class_names, saved_name=saved_name)


    wandb.finish()