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

def weighted_normalized_CrossEntropyLoss(return_weights = False):
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
    normalized_weights = 1 + 0.5*(raw_weights - min_val) / (max_val - min_val)  # 범위를 [1, 2]로 조정

    # GPU로 이동
    weights = normalized_weights
    if return_weights:
        return weights
    # 손실 함수 정의
    else:
        criterion = nn.CrossEntropyLoss(weight=weights)
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