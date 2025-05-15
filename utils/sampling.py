from torch.utils.data import WeightedRandomSampler, Sampler
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
    def __init__(self, dataset_size, batch_size, hard_negative_miner, labels, num_classes, hard_negative_ratio=0.3):
        """
        하드 네거티브 샘플과 클래스 균형이 맞는 랜덤 샘플을 혼합하여 배치를 구성하는 샘플러
        
        Args:
            dataset_size: 데이터셋 크기
            batch_size: 배치 크기
            hard_negative_miner: 하드 네거티브 마이너 인스턴스
            labels: 각 샘플의 클래스 레이블 (numpy array)
            num_classes: 클래스 수
            hard_negative_ratio: 배치에서 하드 네거티브 샘플의 비율 (0~1)
        """
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.hard_negative_miner = hard_negative_miner
        self.labels = labels
        self.num_classes = num_classes
        self.hard_negative_ratio = hard_negative_ratio
        
        # 배치당 하드 네거티브 샘플 수
        self.hard_negative_per_batch = int(batch_size * hard_negative_ratio)
        self.random_per_batch = batch_size - self.hard_negative_per_batch
        
        # 클래스별 인덱스 구성
        self.class_indices = [[] for _ in range(num_classes)]
        for idx, label in enumerate(labels):
            self.class_indices[label].append(idx)
        
        # 에포크당 배치 수 계산
        self.num_batches = (dataset_size + batch_size - 1) // batch_size
        
    def __iter__(self):
        # 클래스별 인덱스 섞기
        for class_idx in range(self.num_classes):
            random.shuffle(self.class_indices[class_idx])
        
        # 각 클래스별 현재 인덱스 위치 초기화
        class_positions = [0] * self.num_classes
        
        for _ in range(self.num_batches):
            # 하드 네거티브 샘플 선택
            hard_indices = self.hard_negative_miner.get_hard_negatives(self.hard_negative_per_batch)
            hard_indices_set = set(hard_indices)
            
            # 클래스별로 균등하게 샘플 선택
            samples_per_class = self.random_per_batch // self.num_classes
            remaining_samples = self.random_per_batch % self.num_classes
            
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
                while len(selected_indices) < class_samples_needed and checked_indices < len(class_indices):
                    pos = (class_positions[class_idx] + checked_indices) % len(class_indices)
                    idx = class_indices[pos]
                    checked_indices += 1
                    
                    if idx not in hard_indices_set:
                        selected_indices.append(idx)
                
                # 충분한 샘플이 없으면 하드 네거티브와 중복 허용
                if len(selected_indices) < class_samples_needed:
                    additional_needed = class_samples_needed - len(selected_indices)
                    # 무작위로 해당 클래스에서 추가 샘플 선택
                    additional_indices = random.sample(
                        class_indices, 
                        min(additional_needed, len(class_indices))
                    )
                    selected_indices.extend(additional_indices)
                
                # 다음 에포크를 위해 클래스 위치 업데이트
                class_positions[class_idx] = (class_positions[class_idx] + len(selected_indices)) % len(class_indices)
                balanced_indices.extend(selected_indices)
            
            # 하드 네거티브와 균형 잡힌 샘플 합치기
            batch_indices = hard_indices + balanced_indices
            random.shuffle(batch_indices)  # 배치 내에서도 섞기
            
            yield batch_indices
    
    def __len__(self):
        return self.num_batches