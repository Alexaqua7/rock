from torch.utils.data import WeightedRandomSampler, Sampler, DataLoader
import numpy as np
import random

def create_weighted_sampler(labels):
    """각 클래스에서 균등하게 샘플링하는 WeightedRandomSampler 생성"""
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    return sampler

class HardNegativeMiner:
    def __init__(self, dataset_size, memory_size=1000):
        self.hard_negative_indices = set()
        self.hard_negative_scores = {}  # 각 샘플의 손실값 저장
        self.memory_size = memory_size
        self.dataset_size = dataset_size
    
    def update(self, indices, losses):
        """
        손실값이 높은 샘플들을 하드 네거티브로 추가
        
        Args:
            indices: 샘플 인덱스 리스트
            losses: 각 샘플의 손실값 리스트
        """
        # 인덱스와 손실값 매핑
        for idx, loss in zip(indices, losses):
            self.hard_negative_scores[idx] = loss.item()
        
        # 손실값 기준으로 정렬하여 상위 memory_size개만 유지
        sorted_items = sorted(self.hard_negative_scores.items(), 
                              key=lambda x: x[1], reverse=True)
        
        if len(sorted_items) > self.memory_size:
            sorted_items = sorted_items[:self.memory_size]
        
        # 새로운 하드 네거티브 인덱스 집합 구성
        self.hard_negative_indices = set([idx for idx, _ in sorted_items])
        self.hard_negative_scores = {idx: loss for idx, loss in sorted_items}
    
    def get_hard_negatives(self, count):
        """
        하드 네거티브 샘플에서 'count'만큼 샘플링
        
        Args:
            count: 필요한 하드 네거티브 샘플 수
            
        Returns:
            선택된 하드 네거티브 인덱스 리스트
        """
        if not self.hard_negative_indices:
            # 하드 네거티브 샘플이 없으면 랜덤 인덱스 반환
            return random.sample(range(self.dataset_size), min(count, self.dataset_size))
        
        # 가중치 기반 샘플링을 위한 준비
        indices = list(self.hard_negative_indices)
        weights = [self.hard_negative_scores[idx] for idx in indices]
        
        # 가중치 정규화
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # 가중치 기반 샘플링
        if len(indices) <= count:
            return indices
        else:
            return np.random.choice(indices, count, replace=False, p=weights).tolist()

class BalancedHardNegativeBatchSampler(Sampler):
    def __init__(self, dataset_size, batch_size, hard_negative_miner, labels, num_classes, 
                 hard_negative_ratio=0.3, accumulation_steps=1):
        """
        Gradient Accumulation을 고려한 하드 네거티브 샘플과 클래스 균형 샘플러
        
        Args:
            dataset_size: 데이터셋 크기
            batch_size: 개별 배치 크기
            hard_negative_miner: 하드 네거티브 마이너 인스턴스
            labels: 각 샘플의 클래스 레이블 (numpy array)
            num_classes: 클래스 수
            hard_negative_ratio: effective batch에서 하드 네거티브 샘플의 비율 (0~1)
            accumulation_steps: gradient accumulation 스텝 수
        """
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.hard_negative_miner = hard_negative_miner
        self.labels = labels
        self.num_classes = num_classes
        self.hard_negative_ratio = hard_negative_ratio
        self.accumulation_steps = accumulation_steps
        
        # Effective batch size 계산
        self.effective_batch_size = batch_size * accumulation_steps
        
        # Effective batch당 하드 네거티브 샘플 수
        self.hard_negative_per_effective_batch = int(self.effective_batch_size * hard_negative_ratio)
        self.random_per_effective_batch = self.effective_batch_size - self.hard_negative_per_effective_batch
        
        # 클래스별 인덱스 구성
        self.class_indices = [[] for _ in range(num_classes)]
        for idx, label in enumerate(labels):
            self.class_indices[label].append(idx)
        
        # 에포크당 effective batch 수 계산
        self.num_effective_batches = (dataset_size + self.effective_batch_size - 1) // self.effective_batch_size
        self.num_batches = self.num_effective_batches * accumulation_steps
        
        print(f"Sampler Configuration:")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Accumulation steps: {accumulation_steps}")
        print(f"  - Effective batch size: {self.effective_batch_size}")
        print(f"  - Hard negative per effective batch: {self.hard_negative_per_effective_batch}")
        print(f"  - Random samples per effective batch: {self.random_per_effective_batch}")
        print(f"  - Samples per class per effective batch: {self.random_per_effective_batch // num_classes}")
        
    def _get_balanced_samples_for_effective_batch(self, hard_indices_set, class_positions):
        """Effective batch에 대해 클래스별로 균등하게 샘플 선택"""
        samples_per_class = self.random_per_effective_batch // self.num_classes
        remaining_samples = self.random_per_effective_batch % self.num_classes
        
        balanced_indices = []
        
        # 각 클래스에서 samples_per_class 개의 샘플 선택
        for class_idx in range(self.num_classes):
            class_samples_needed = samples_per_class + (1 if class_idx < remaining_samples else 0)
            if class_samples_needed == 0:
                continue
            
            selected_indices = []
            checked_indices = 0
            class_indices = self.class_indices[class_idx]
            
            # 이미 선택된 하드 네거티브와 중복되지 않는 샘플 선택
            while len(selected_indices) < class_samples_needed and checked_indices < len(class_indices) * 2:
                pos = (class_positions[class_idx] + checked_indices) % len(class_indices)
                idx = class_indices[pos]
                checked_indices += 1
                
                if idx not in hard_indices_set:
                    selected_indices.append(idx)
            
            # 충분한 샘플이 없으면 하드 네거티브와 중복 허용
            if len(selected_indices) < class_samples_needed:
                additional_needed = class_samples_needed - len(selected_indices)
                # 무작위로 해당 클래스에서 추가 샘플 선택
                additional_candidates = [idx for idx in class_indices if idx not in selected_indices]
                if additional_candidates:
                    additional_indices = random.sample(
                        additional_candidates, 
                        min(additional_needed, len(additional_candidates))
                    )
                    selected_indices.extend(additional_indices)
                
                # 여전히 부족하면 중복 허용하여 채움
                while len(selected_indices) < class_samples_needed:
                    selected_indices.append(random.choice(class_indices))
            
            # 다음 effective batch를 위해 클래스 위치 업데이트
            class_positions[class_idx] = (class_positions[class_idx] + len(selected_indices)) % len(class_indices)
            balanced_indices.extend(selected_indices)
        
        return balanced_indices
    
    def __iter__(self):
        # 클래스별 인덱스 섞기
        for class_idx in range(self.num_classes):
            random.shuffle(self.class_indices[class_idx])
        
        # 각 클래스별 현재 인덱스 위치 초기화
        class_positions = [0] * self.num_classes
        
        for effective_batch_idx in range(self.num_effective_batches):
            # Effective batch에 대해 하드 네거티브 샘플 선택
            hard_indices = self.hard_negative_miner.get_hard_negatives(self.hard_negative_per_effective_batch)
            hard_indices_set = set(hard_indices)
            
            # Effective batch에 대해 클래스별로 균등하게 샘플 선택
            balanced_indices = self._get_balanced_samples_for_effective_batch(
                hard_indices_set, class_positions
            )
            
            # 전체 effective batch 인덱스
            effective_batch_indices = hard_indices + balanced_indices
            random.shuffle(effective_batch_indices)
            
            # Effective batch를 개별 배치들로 분할
            for step in range(self.accumulation_steps):
                start_idx = step * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(effective_batch_indices))
                
                batch_indices = effective_batch_indices[start_idx:end_idx]
                
                # 배치 크기가 부족한 경우 패딩 (마지막 배치에서 발생 가능)
                while len(batch_indices) < self.batch_size and len(effective_batch_indices) > 0:
                    batch_indices.append(random.choice(effective_batch_indices))
                
                yield batch_indices
    
    def __len__(self):
        return self.num_batches


# 사용 예시를 위한 수정된 데이터로더 생성 함수
def create_train_loader_with_accumulation(train_dataset, hard_negative_miner, labels, num_classes, 
                                        batch_size, accumulation_steps, hard_negative_ratio=0.2, num_workers=16, prefetch_factor=4, pin_memory=True):
    """
    Gradient Accumulation을 고려한 train loader 생성
    """
    balanced_batch_sampler = BalancedHardNegativeBatchSampler(
        dataset_size=len(train_dataset),
        batch_size=batch_size,
        hard_negative_miner=hard_negative_miner,
        labels=labels,
        num_classes=num_classes,
        hard_negative_ratio=hard_negative_ratio,
        accumulation_steps=accumulation_steps
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_sampler=balanced_batch_sampler,
        num_workers=num_workers, 
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor
    )
    
    return train_loader