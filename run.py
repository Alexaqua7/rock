import albumentations as A
import warnings
from utils.transforms import RandomCenterCrop, PadSquare
from albumentations.pytorch import ToTensorV2
from utils.trainer import Trainer
warnings.filterwarnings(action='ignore')



img_size = 224 # IMG_SIZE를 수정할 시, 이 부분을 수정할 것

CFG = {
    # 필수적인 인자
    'SEED': 41, # SEED 설정
    'TEST_SIZE': 0.2, #Train/Val 구성 시, Test Proportion 설정
    'TRAIN_TRANSFORM': A.Compose([ #Train 시, 사용할 Transform 정의
        RandomCenterCrop(min_size=75, max_size=200, p=0.5),
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),  # 50% 확률로 좌우 반전
        A.VerticalFlip(p=0.5),    # 50% 확률로 상하 반전
        A.CLAHE(p=0.5),
        A.GaussNoise(std_range=(0.1,0.15), p=0.5),
        A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        ToTensorV2()
    ]),
    'TEST_TRANSFORM': A.Compose([ #(trainer.train()사용 시, Validation // trainer.predict사용 시, Test) 시, 사용할 Transform 정의 #########Inference 시 필수 입력#############
        RandomCenterCrop(min_size=75, max_size=200, p=0.4),
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        ToTensorV2()
    ]),
    'MODEL_NAME': 'davit_base', # 모델 이름 정의
    'IMG_SIZE': img_size, 
    'EPOCHS': 15, # 전체 Epoch 수 정의
    'BATCH_SIZE': 32, # Batch Size 정의
    'NUM_WORKERS': 16, # Num Workers 정의
    'LEARNING_RATE': 3e-4, # LR 정의 (Warmup 사용 시, Peak(가장 높은 LR)에 도달하였을 때 LR을 의미함)

    # kFold 시 필수 입력 (FOLD==0이면 single_mode_train으로 실행) 0-indexed
    'FOLD': 5,
    'START_FOLD': 3, # Fold를 중간부터 시작할 때 (처음부터 수행하려면 None 또는 0으로 설정)
    'END_FOLD': None, #Fold를 중간에서 끝내고자 할 때 (END_FOLD까지 수행) (끝까지 수행하려면 None 또는 FOLD와 동일 값으로 설정)

    # 선택적인 인자 (기본값 사용 또는 명시적 설정)
    'DATA_PATH': './train',
    'EXPERIMENT_PATH': './experiments',
    # HARD_NEGATIVE_MEMORY_SIZE, HARD_NEGATIVE_RATIO이 0이 아니면 HARD NEGATIVE TRAIN 수행
    # BALANCED_BATCH가 TRUE면 OVERSAMPLING TRAIN 수행
    # HARD_NEGATIVE_MEMORY_SIZE, HARD_NEGATIVE_RATIO, BALANCED_BATCH가 (0,0,False)면 DEFAULT (BASE) TRAIN 수행
    'HARD_NEGATIVE_MEMORY_SIZE': 1000, #HARD NEGATIVE SAMPLING 사용 시 필수 설정! 얼만큼의 Hard Negative Samples 수를 저장하고 있을 지 설정
    'HARD_NEGATIVE_RATIO': 0.2, #HARD NEGATIVE SAMPLING 사용 시 필수 설정! 전체 Batch Size 중 얼마의 비율을 Hard Negative Samples로 가져갈 것인지 설정
    'BALANCED_BATCH': True, #OVERSAMPLING 사용 시 True로 설정! 각 클래스에서 동일한 개수의 데이터를 샘플링하여 배치 구성을 할 수 있는 설정 
    'OPTIMIZER': 'adam', #Optimizer 설정 (adam, adamw 중 택 1); Default: adam
    'WARM_UP': 0, #WarmUp 사용 시, Warmup할 Epoch 수 설정. WarmUp은 LinearLR로 수행됨. 0이라면 Warmup을 사용하지 않음; Default: 0
    'ETA_MIN': 1e-8, #CosineAnnealingLR 사용 시, Min_LR을 설정; Default: 1e-8
    'SCHEDULER': 'cosineannealing', #현재 cosineannealing만 사용할 수 있도록 설정됨
    'START_FACTOR': 1/3, #(Warmup 사용 시에만 작동함), InitialLR을 얼마로 설정할 것인지 결정. InitialLR = LEARNING_RATE * START_FACTOR, MaximumLR (BaseLR) = LEARNING_RATE; Default: 1/3
    'ACCUMULATION_STEPS': 16, #Gradient Accumulation 사용 시, 몇 Step마다 Update할 것인지 결정. Effective Batch Size = BATCH_SIZE * ACCUMULATION_STEPS; Default: 1
    'WANDB_PROJECT': 'rock-classification',
    'LOSS_TYPE': 'CE', #Loss Function 설정 (weighted_normalized, CE, weighted_normalized_custom, weighted_normalized_diff_weighted 중 택 1); Default: CE (CrossEntropyLoss)
    'FACTOR': 1, #(LOSS_TYPE으로 weighted_normalized 사용 시에만 작동함) Class_weights를 어느 범위로 가져갈 것인지 결정 [1, 1 + FACTOR]; Default: 1 --> Default 범위는 [1, 2]
    'LABEL_SMOOTHING': 0.1, #Label_Smoothing 사용 시, 설정. 0이면 Label Smoothing을 사용하지 않음 (수치가 커질 수록 더 많은 Label Smoothing을 적용); Default: 0
    'TRAINED_PATH': '', #이어서 Train을 진행할 때는 선택적으로 입력. #########Inference 시 필수 입력#############


    # Progressive Hard Negative Sampling 관련 인자 (Progressive Hard Negative Sampling 사용 시 필수 설정) INITIAL_RATIO == FINAL_RATIO거나 둘 중 하나라도 비어있다면 일반 Hard Negative Sampling Training으로 수행됨
    'INITIAL_RATIO': 0.2, # 처음 시작할 때 Hard Negative Sampling Ratio 설정 
    'FINAL_RATIO': 0.6, # 마지막 끝날 때 Hard Negative Sampling Ratio 설정
    'HARD_NEGATIVE_CURRICULUM_LEARNING_EPOCH': 10,
    'SCHEDULE_TYPE': 'cosine' # Ratio를 어떻게 전체 학습과정에 걸쳐 조절할 지 설정 (linear, exponential, cosine, step<25%, 50%, 75%에서 계단식으로 증가>)
}

if __name__ == "__main__":
    trainer = Trainer(config=CFG)
    trainer.train() # Train 시 주석 해제
    # trainer.predict() # Inference 시 주석 해제
