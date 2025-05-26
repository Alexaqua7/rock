from torch import nn
import os
from tqdm import tqdm
import wandb
import torch
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from utils.custom_decorator import with_torch_no_grad

def train(model, optimizer, train_loader, val_loader, scheduler, device, class_names, criterion=nn.CrossEntropyLoss(), epochs=15, best_score=0, cur_epoch=1, experiment_name="base", folder_path = "base", accumulation_steps=1):
    model.to(device)
    best_model = None
    save_path = os.path.join(folder_path, f"{experiment_name}-best.pth")

    for epoch in range(cur_epoch, epochs + 1):
        model.train()
        train_loss = []
        optimizer.zero_grad()
        progress_bar = tqdm(enumerate(iter(train_loader)), total=len(train_loader), desc=f"Epoch {epoch}/{epochs}")
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

def hard_negative_train(model, optimizer, train_loader, val_loader, scheduler, device, class_names, criterion=nn.CrossEntropyLoss(reduction='none'), hard_negative_miner=None, best_score=0, epochs=15, cur_epoch=1, experiment_name="base", folder_path="base", accumulation_steps=1):
    model.to(device)
    
    best_model = None
    save_path = os.path.join(folder_path, f"{experiment_name}-best.pth")
    
    for epoch in range(cur_epoch, epochs + 1):
        model.train()
        train_loss = []
        optimizer.zero_grad()
        
        # 그래디언트 누적을 위한 배치 손실 및 인덱스 저장소
        accumulated_indices = []
        accumulated_losses = []
        
        progress_bar = tqdm(enumerate(iter(train_loader)), total=len(train_loader), desc=f"Epoch {epoch}/{epochs}")
        for step, batch in progress_bar:
            if len(batch) == 3:  # 인덱스 포함하는 경우
                imgs, labels, indices = batch
            else:
                imgs, labels = batch
                indices = None
            
            imgs = imgs.float().to(device)
            labels = labels.to(device).long()

            output = model(imgs)
            
            # 샘플별 손실값 계산 - 원래 스케일 유지 (accumulation_steps로 나누지 않음)
            sample_losses = criterion(output, labels)
            loss = sample_losses.mean() / accumulation_steps  # 그래디언트 스케일링을 위해서만 나눔

            # 역전파 수행
            loss.backward()
            
            # 현재 배치의 손실값과 인덱스 저장
            if indices is not None:
                accumulated_indices.extend(indices.cpu().numpy())
                accumulated_losses.extend(sample_losses.detach().cpu())

            train_loss.append(loss.item() * accumulation_steps)  # 원래 손실값 저장 (scaling 제거)
            
            # 그래디언트 업데이트 및 하드 네거티브 마이너 업데이트
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                # 옵티마이저 스텝
                optimizer.step()
                optimizer.zero_grad()
                
                # 누적된 배치들에 대한 하드 네거티브 마이너 업데이트
                if hard_negative_miner is not None and accumulated_indices:
                    hard_negative_miner.update(accumulated_indices, accumulated_losses)
                
                # 누적 저장소 초기화
                accumulated_indices = []
                accumulated_losses = []
            
            # 진행상황 표시
            progress_bar.set_postfix(loss=loss.item() * accumulation_steps)  # 원래 손실값 표시

        # 검증 단계
        _val_loss, _val_score, class_f1_dict, wandb_cm = hard_negative_validation(model, criterion, val_loader, device, class_names)

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

    return best_model

@with_torch_no_grad
def validation(model, criterion, val_loader, device, class_names):
    model.eval()
    val_loss = []
    preds, true_labels = [], []

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

@with_torch_no_grad
def hard_negative_validation(model, criterion, val_loader, device, class_names):
    model.eval()
    val_loss = []
    preds, true_labels = [], []

    for batch in tqdm(iter(val_loader)):
        if len(batch) == 3:  # 인덱스 포함하는 경우
            imgs, labels, _ = batch
        else:
            imgs, labels = batch
        
        imgs = imgs.float().to(device)
        labels = labels.to(device).long()

        pred = model(imgs)
        sample_losses = criterion(pred, labels)  # 샘플별 손실값 계산
        loss = sample_losses.mean()  # 평균 손실값 계산

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

def progressive_hard_negative_train(model, optimizer, train_dataset, val_loader, scheduler, device, 
                                  class_names, labels, num_classes, batch_size, accumulation_steps,
                                  progressive_scheduler, hard_negative_miner=None, 
                                  criterion=nn.CrossEntropyLoss(reduction='none'), 
                                  best_score=0, epochs=15, cur_epoch=1, 
                                  experiment_name="progressive", folder_path="progressive",
                                  num_workers=16, prefetch_factor=4, pin_memory=True):
    """
    Progressive Hard Negative Training 함수
    
    Args:
        model: 훈련할 모델
        optimizer: 옵티마이저
        train_dataset: 훈련 데이터셋 (인덱스를 반환해야 함)
        val_loader: 검증 데이터로더
        scheduler: 학습률 스케줄러
        device: 디바이스
        class_names: 클래스 이름 리스트
        labels: 훈련 데이터의 레이블 (numpy array)
        num_classes: 클래스 수
        batch_size: 배치 크기
        accumulation_steps: gradient accumulation 스텝
        progressive_scheduler: ProgressiveScheduler 인스턴스
        hard_negative_miner: HardNegativeMiner 인스턴스
        criterion: 손실 함수
        best_score: 초기 최고 점수
        epochs: 총 에포크 수
        cur_epoch: 시작 에포크
        experiment_name: 실험 이름
        folder_path: 모델 저장 폴더
        num_workers: 데이터로더 워커 수
        prefetch_factor: 프리페치 팩터
        pin_memory: 메모리 핀 여부
    """
    model.to(device)
    
    best_model = None
    save_path = os.path.join(folder_path, f"{experiment_name}-best.pth")
    
    for epoch in range(cur_epoch, epochs + 1):
        print(f"\n{'='*50}")
        print(f"Starting Epoch {epoch}/{epochs}")
        print(f"{'='*50}")
        
        # Progressive sampler로 새로운 train loader 생성
        from utils.sampling import create_progressive_train_loader  # 위에서 만든 함수 import
        
        train_loader, batch_sampler = create_progressive_train_loader(
            train_dataset=train_dataset,
            hard_negative_miner=hard_negative_miner,
            labels=labels,
            num_classes=num_classes,
            batch_size=batch_size,
            accumulation_steps=accumulation_steps,
            progressive_scheduler=progressive_scheduler,
            current_epoch=epoch - 1,  # 0-based indexing
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory
        )
        
        # 현재 epoch의 hard negative ratio 로깅
        current_ratio = progressive_scheduler.get_ratio(epoch - 1)
        print(f"Current Hard Negative Ratio: {current_ratio:.3f}")
        
        model.train()
        train_loss = []
        optimizer.zero_grad()
        
        # 그래디언트 누적을 위한 배치 손실 및 인덱스 저장소
        accumulated_indices = []
        accumulated_losses = []
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), 
                           desc=f"Epoch {epoch}/{epochs} (HN_ratio: {current_ratio:.3f})")
        
        for step, batch in progress_bar:
            if len(batch) == 3:  # 인덱스 포함하는 경우
                imgs, labels_batch, indices = batch
            else:
                imgs, labels_batch = batch
                indices = None
            
            imgs = imgs.float().to(device)
            labels_batch = labels_batch.to(device).long()

            output = model(imgs)
            
            # 샘플별 손실값 계산 - 원래 스케일 유지 (accumulation_steps로 나누지 않음)
            sample_losses = criterion(output, labels_batch)
            loss = sample_losses.mean() / accumulation_steps  # 그래디언트 스케일링을 위해서만 나눔

            # 역전파 수행
            loss.backward()
            
            # 현재 배치의 손실값과 인덱스 저장
            if indices is not None:
                accumulated_indices.extend(indices.cpu().numpy())
                accumulated_losses.extend(sample_losses.detach().cpu())

            train_loss.append(loss.item() * accumulation_steps)  # 원래 손실값 저장 (scaling 제거)
            
            # 그래디언트 업데이트 및 하드 네거티브 마이너 업데이트
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                # 옵티마이저 스텝
                optimizer.step()
                optimizer.zero_grad()
                
                # 누적된 배치들에 대한 하드 네거티브 마이너 업데이트
                if hard_negative_miner is not None and accumulated_indices:
                    hard_negative_miner.update(accumulated_indices, accumulated_losses)
                
                # 누적 저장소 초기화
                accumulated_indices = []
                accumulated_losses = []
            
            # 진행상황 표시
            progress_bar.set_postfix({
                'loss': f"{loss.item() * accumulation_steps:.4f}",
                'hn_ratio': f"{current_ratio:.3f}"
            })

        # 검증 단계
        _val_loss, _val_score, class_f1_dict, wandb_cm = hard_negative_validation(
            model, criterion, val_loader, device, class_names
        )

        # 하드 네거티브 마이너 상태 로깅
        hard_negative_stats = {}
        if hard_negative_miner is not None:
            hard_negative_stats = {
                'hard_negative_count': len(hard_negative_miner.hard_negative_indices),
                'hard_negative_ratio': current_ratio,
                'hard_negative_memory_usage': len(hard_negative_miner.hard_negative_indices) / hard_negative_miner.memory_size
            }

        log_data = {
            'epoch': epoch,
            'train_loss': sum(train_loss) / len(train_loss),
            'val_loss': _val_loss,
            'val_macro_f1': _val_score,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'confusion_matrix': wandb_cm,
            **hard_negative_stats  # 하드 네거티브 통계 추가
        }
        log_data.update(class_f1_dict)
        wandb.log(log_data)

        # 체크포인트 저장
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'best_score': best_score,
            'progressive_scheduler_state': {
                'initial_ratio': progressive_scheduler.initial_ratio,
                'final_ratio': progressive_scheduler.final_ratio,
                'total_epochs': progressive_scheduler.total_epochs,
                'schedule_type': progressive_scheduler.schedule_type
            },
            'hard_negative_miner_state': {
                'hard_negative_indices': list(hard_negative_miner.hard_negative_indices) if hard_negative_miner else [],
                'hard_negative_scores': hard_negative_miner.hard_negative_scores if hard_negative_miner else {},
                'memory_size': hard_negative_miner.memory_size if hard_negative_miner else 0
            }
        }
        
        epoch_checkpoint_path = os.path.join(folder_path, f'{experiment_name}-epoch_{epoch}.pth')
        torch.save(checkpoint, epoch_checkpoint_path)
        print(f"Checkpoint at epoch {epoch} saved → {experiment_name}-epoch_{epoch}.pth")

        # 이전 에포크 체크포인트 삭제
        prev_path = os.path.join(folder_path, f'{experiment_name}-epoch_{epoch-1}.pth')
        if epoch > 1 and os.path.exists(prev_path):
            os.remove(prev_path)
            print(f"Previous checkpoint deleted → {prev_path}")

        # 스케줄러 업데이트
        if scheduler is not None:
            scheduler.step()

        # 최고 성능 모델 저장
        if best_score < _val_score:
            best_score = _val_score
            best_model = model
            checkpoint['best_score'] = best_score
            torch.save(checkpoint, save_path)
            print(f"Best model saved (epoch {epoch}, F1={_val_score:.4f}) → {save_path}")

        print(f"Epoch {epoch} Summary:")
        print(f"  - Train Loss: {sum(train_loss) / len(train_loss):.4f}")
        print(f"  - Val Loss: {_val_loss:.4f}")
        print(f"  - Val F1: {_val_score:.4f}")
        print(f"  - Hard Negative Ratio: {current_ratio:.3f}")
        if hard_negative_miner:
            print(f"  - Hard Negative Count: {len(hard_negative_miner.hard_negative_indices)}")

    return best_model


def load_progressive_checkpoint(checkpoint_path, model, optimizer, scheduler, 
                              progressive_scheduler, hard_negative_miner, device):
    """
    Progressive training 체크포인트 로드
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 모델과 옵티마이저 상태 로드
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Progressive scheduler 상태 복원
    if 'progressive_scheduler_state' in checkpoint:
        scheduler_state = checkpoint['progressive_scheduler_state']
        progressive_scheduler.initial_ratio = scheduler_state['initial_ratio']
        progressive_scheduler.final_ratio = scheduler_state['final_ratio']
        progressive_scheduler.total_epochs = scheduler_state['total_epochs']
        progressive_scheduler.schedule_type = scheduler_state['schedule_type']
    
    # Hard negative miner 상태 복원
    if hard_negative_miner and 'hard_negative_miner_state' in checkpoint:
        miner_state = checkpoint['hard_negative_miner_state']
        hard_negative_miner.hard_negative_indices = set(miner_state['hard_negative_indices'])
        hard_negative_miner.hard_negative_scores = miner_state['hard_negative_scores']
        hard_negative_miner.memory_size = miner_state['memory_size']
    
    return checkpoint['epoch'], checkpoint['best_score']