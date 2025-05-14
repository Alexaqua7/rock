import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 클래스별 가중치 텐서 or None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)  # 예측 확률 (logit을 softmax로 변환한 것)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def weighted_normalized_CrossEntropyLoss(class_counts=None, return_weights = False, label_smoothing=0.0, factor=1):
    if class_counts is None:
        class_counts = {
                'Andesite': 43802,
                'Basalt': 26810,
                'Etc': 15395,
                'Gneiss': 73914,
                'Granite': 92923,
                'Mud_Sandstone': 89467,
                'Weathered_Rock': 37169
            }

    # 클래스 이름 순서 정의
    class_names = list(class_counts.keys())

    # 총 샘플 수
    total_samples = sum(class_counts.values())

    # 원래 weight 계산
    raw_weights = torch.tensor([
        total_samples / (len(class_counts) * v) for v in class_counts.values()
    ], dtype=torch.float)

    # Min-Max 정규화 (1~2 사이로 조정)
    min_val = raw_weights.min()
    max_val = raw_weights.max()
    normalized_weights = 1 + factor*(raw_weights - min_val) / (max_val - min_val)  # 범위를 [1, 2]로 조정

    # GPU로 이동
    weights = normalized_weights
    if return_weights:
        return weights
    # 손실 함수 정의
    else:
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
        return criterion

def weighted_normalized_CrossEntropyLoss_custom(class_counts=None, return_weights=False):
    if class_counts is None:
        class_counts = {
            'Andesite': 43802,
            'Basalt': 26810,
            'Etc': 15395,
            'Gneiss': 73914,
            'Granite': 92923,
            'Mud_Sandstone': 89467,
            'Weathered_Rock': 37169
        }
    class_names_list = list(class_counts.keys())

    etc_counts = 0
    non_etc_counts = {}
    etc_class_names = []
    non_etc_class_names = []

    for name in class_names_list:
        if name.startswith('Etc'):
            if class_counts and name in class_counts:
                etc_counts += class_counts[name]
            etc_class_names.append(name)
        else:
            if class_counts and name in class_counts:
                non_etc_counts[name] = class_counts[name]
                non_etc_class_names.append(name)

    non_etc_class_counts = non_etc_counts.copy()

    non_etc_class_names = list(non_etc_class_counts.keys())
    non_etc_samples = sum(non_etc_class_counts.values())
    num_etc_classes = len(non_etc_class_counts)

    raw_weights_merged = {}
    for name, count in non_etc_class_counts.items():
        raw_weights_merged[name] = non_etc_samples / (num_etc_classes * count)

    min_raw_weight_merged = min(raw_weights_merged.values())
    max_raw_weight_merged = max(raw_weights_merged.values())

    normalized_weights_merged = {}
    for name, raw_weight in raw_weights_merged.items():
        normalized_weights_merged[name] = 1 + (raw_weight - min_raw_weight_merged) / (max_raw_weight_merged - min_raw_weight_merged)

    final_weights = torch.zeros(len(class_names_list), dtype=torch.float)
    name_to_index = {name: i for i, name in enumerate(class_names_list)}

    for name in non_etc_class_names:
        if name in normalized_weights_merged:
            final_weights[name_to_index[name]] = normalized_weights_merged[name]

    for name in etc_class_names:
        if name in name_to_index:
            final_weights[name_to_index[name]] = max(normalized_weights_merged.values())
                
    if return_weights:
        return final_weights
    else:
        criterion = nn.CrossEntropyLoss(weight=final_weights)
        return criterion
    
def weighted_normalized_CrossEntropyLoss_diff_weighted(class_counts=None, return_weights=False):
    if class_counts is None:
        class_counts = {
            'Andesite': 43802,
            'Basalt': 26810,
            'Etc': 15395,
            'Gneiss': 73914,
            'Granite': 92923,
            'Mud_Sandstone': 89467,
            'Weathered_Rock': 37169
        }
    class_names_list = list(class_counts.keys())

    etc_counts = 0
    non_etc_counts = {}
    etc_class_names = []
    non_etc_class_names = []

    for name in class_names_list:
        if name.startswith('Etc'):
            if class_counts and name in class_counts:
                etc_counts += class_counts[name]
            etc_class_names.append(name)
        else:
            if class_counts and name in class_counts:
                non_etc_counts[name] = class_counts[name]
                non_etc_class_names.append(name)

    merged_class_counts = non_etc_counts.copy()
    if etc_counts > 0:
        merged_class_counts['Etc'] = etc_counts

    merged_class_names = list(merged_class_counts.keys())
    total_merged_samples = sum(merged_class_counts.values())
    num_merged_classes = len(merged_class_counts)

    raw_weights_merged = {}
    for name, count in merged_class_counts.items():
        raw_weights_merged[name] = total_merged_samples / (num_merged_classes * count)

    min_raw_weight_merged = min(raw_weights_merged.values())
    max_raw_weight_merged = max(raw_weights_merged.values())

    normalized_weights_merged = {}
    for name, raw_weight in raw_weights_merged.items():
        normalized_weights_merged[name] = 1 + (raw_weight - min_raw_weight_merged) / (max_raw_weight_merged - min_raw_weight_merged)

    final_weights = torch.zeros(len(class_names_list), dtype=torch.float)
    name_to_index = {name: i for i, name in enumerate(class_names_list)}

    for name in non_etc_class_names:
        if name in normalized_weights_merged:
            final_weights[name_to_index[name]] = normalized_weights_merged[name]

    if 'Etc' in normalized_weights_merged:
        etc_weight = normalized_weights_merged['Etc']
        for name in etc_class_names:
            if name in name_to_index:
                final_weights[name_to_index[name]] = etc_weight
                
    if return_weights:
        return final_weights
    else:
        criterion = nn.CrossEntropyLoss(weight=final_weights)
        return criterion

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device='cuda'):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))

    def forward(self, features, labels):
        batch_size = features.size(0)
        centers_batch = self.centers.index_select(0, labels)
        loss = ((features - centers_batch) ** 2).sum() / 2.0 / batch_size
        return loss

class CombinedLoss(nn.Module):
    def __init__(self, base_criterion, center_loss, lambda_c=0.01):
        super(CombinedLoss, self).__init__()
        self.base_criterion = base_criterion  # e.g., FocalLoss
        self.center_loss = center_loss
        self.lambda_c = lambda_c

    def forward(self, outputs, features, labels):
        ce = self.base_criterion(outputs, labels)
        cl = self.center_loss(features, labels)
        return ce + self.lambda_c * cl