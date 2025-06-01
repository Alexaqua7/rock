# 🪨 건설용 자갈 암석 종류 분류 AI 경진대회

> 월간 데이콘 × 한국지능정보사회진흥원(NIA)

---

## 📌 대회 개요

건설용 자갈 이미지에서 **암석의 종류를 자동으로 분류**하는 AI 알고리즘을 개발합니다.  
자갈의 종류는 콘크리트와 아스팔트의 **강도, 내구성, 품질**에 직접적인 영향을 미치며,  
정확한 분류는 건설 현장의 품질 관리에 핵심적인 역할을 합니다.

기존의 수작업 검사 방식을 대체할 수 있는 **AI 기반 자동 분류 시스템**을 통해  
**건설 산업의 디지털 전환**에 기여하는 것이 본 대회의 목표입니다.

---
## 👥 팀원 소개

팀명: 건설용 자갈 암석

<table>
  <tr>
    <td align="center" style="padding: 12px;">
      <strong>DonghwanSeo</strong><br>
      <div style="height:1px; background-color:#ddd; width:60%; margin:6px auto;"></div>
      <img src="https://github.com/user-attachments/assets/f1a3b705-6e42-433e-9e00-9f9243d00c07" width="80"/>
    </td>
    <td align="center" style="padding: 12px;">
      <strong>aqua3g</strong><br>
      <div style="height:1px; background-color:#ddd; width:60%; margin:6px auto;"></div>
      <img src="https://github.com/user-attachments/assets/3d0a8319-2e5d-4add-93d9-131d6f2f9d97" width="80"/>
    </td>
  </tr>
</table>


---
## 🗂️ 데이터 소개

- **이미지 수**: 총 약 `475,026`장  
  - train: 380,020장  
  - test: 95,006장  
- **클래스 수**: `7`개  
  - Andesite, Basalt, Etc, Gneiss, Granite, Mud_Sandstone, Weathered_Rock  
- **해상도**: 다양함

### 데이터 구성
<pre>
├── train/
│ ├── Andesite/
│ ├── Basalt/
│ ├── Etc/
│ ├── Gneiss/
│ ├── Granite/
│ ├── Mud_Sandstone/
│ └── Weathered_Rock/
├── test/
├── sample_submission.csv # submission template
└── test.csv # 테스트 파일 명과 파일 경로
</pre>
---

## 🧠 모델 아키텍처

다양한 CNN 및 Transformer 기반 아키텍처를 실험하였습니다.

- **Baseline**: `Inception ResNet`
- **최종 사용**: `InternImage-XL`, `DaViT_base`, `Mamba_out`
- **Augmentations**: `RandomCenterCrop`, `HorizontalFlip`, `VerticalFlip`, `CLAHE`, `Gaussian Noise`
- **Loss Function**: `CrossEntropyLoss`, `LabelSmoothing`
- **Optimizer**: `AdamW`, `Adam`
- **Scheduler**: `CosineAnnealingLR`, `StepLR`

---
## 🏆 성능 요약

| 모델명                          | Macro-F1 (Validation 기준) | 
|--------------------------------|-----------------------------|
| InternImage-XL                 | 91.68%                      |
| DaViT_base                     | 90.77%                      | 
| Mamba_out                      | 90.89%                      | 
| 🧪 Ensemble                    | **93.29% (Public 기준)**     | 

<details>
  <summary>자세한 설명 보기</summary>

- **InternImage-XL**  
  ImageNet pretrained 모델이며 Hard Negative Sample을 이용하여 학습되었습니다.

- **DaViT_base**  
  ImageNet pretrained 기반이며 Hard Negative Sample 전략을 적용하였습니다.

- **Mamba_out**  
  ImageNet pretrained 모델로, 소수 클래스 비율을 보완하기 위해 Oversample을 사용했습니다.

- **Ensemble**  
  위 세 모델에 대해 Test Time Augmentation(TTA) 및 Soft Voting 방식의 앙상블을 수행했습니다.
</details>

---



## 프로젝트 구조

<pre>

ROCK/
├── experiments/
├── models/
│   └── internimage/
│       ├── hardnegative_internimage.py
│       ├── inference_internimage.py
│       └── internimage.py
├── test/
├── train/
├── utils/
│   ├── custom_decorator.py
│   ├── dataset.py
│   ├── loss.py
│   ├── sampling.py
│   ├── trainer.py
│   ├── training_function.py
│   ├── transforms.py
│   └── utils.py
├── weights/
│   └── OpenCVLab/
│       └── internimage_xl_22kto1k_384/
├── download.py 
├── logit_ensemble.py
├── README.md
├── requirements.txt
├── run.py
├── sample_submission.csv
├── soft_ensemble.py
└── test.csv

</pre>
  
### 주요 디렉토리 및 파일 설명

* **`experiments/`**: 다양한 실험 설정 및 결과가 저장되는 디렉토리
* **`models/`**: TIMM외의 모델 관련 코드가 포함되어 있습니다 (InternImage)
* **`utils/`**: 데이터셋 처리, 손실 함수, 트레이너 등 유틸리티 함수 및 클래스들이 모여 있습니다.
* **`weights/`**: 미리 학습된 모델 가중치 파일들이 저장됩니다. (InternImage를 돌리기 위해서 사전에 HuggingFace에서 이 경로에 모델을 저장해주세요)
* **`download.py`**: weights 폴더에 InternImage 모델을 다운로드하기 위한 파일입니다.
* **`requirements.txt`**: 프로젝트 실행에 필요한 Python 패키지 목록을 포함합니다.
* **`run.py`**: 프로젝트의 메인 실행 스크립트입니다. (TIMM 모델 Only)
* **`inference_*.py` 파일들**: 추론(inference)을 수행하는 스크립트들입니다.
* **`soft_ensemble.py`**: 소프트 앙상블 관련 스크립트입니다.
* **`logit_ensemble.py`**: 추론 시 logit을 저장하여 weighted ensemble을 수행할 수 있는 코드입니다.


---

## ⚙️ 실험 실행 방법

본 프로젝트는 세 가지 실험 방식(기본 학습, Hard Negative 학습, Oversampling 학습)과 추론 방식을 포함합니다.

### 1. 학습 (Training)

### TIMM 모델

<details>

##### 1-1. Hard Negative Sample Training

- 어려운 샘플(hard samples)에 집중하여 학습 성능을 향상시키는 방법입니다.
- Hard Negative Sample은 최근 학습 중 오분류된 데이터들 중 높은 손실을 기록한 샘플들로 구성되며, `HARD_NEGATIVE_MEMORY_SIZE`만큼 저장됩니다.
- 전체 배치에서 `HARD_NEGATIVE_RATIO` 만큼의 샘플을 Hard Negative에서 선택하며, 나머지는 클래스 균등 샘플링으로 채워집니다.
- 전체 배치에서 `HARD_NEGATIVE_RATIO`에 따른 Hard Negative 샘플과, 그 외 클래스 균등 샘플링이 채워지는 기준은 `BATCH_SIZE` × `ACCUMULATION_STEPS` 크기를 기준으로 채워집니다.
- `trainer.predict()`을 **주석 처리** 후에, `trainer.train()`를 **주석 해제**한 상태로 실행합니다.

**🔧 주요 설정 인자**

| 인자명                     | 설명                                                                 |
|---------------------------|----------------------------------------------------------------------|
| `TRAIN_TRANSFORM` | Train 시 사용 될 Transforms (`Albumentations.Compose` 활용)       |
| `TEST_TRANSFORM` | Validation 시 사용 될 Transforms (`Albumentations.Compose` 활용)       |
| `BALANCED_BATCH`                               | `True`로 설정                  |
| `HARD_NEGATIVE_MEMORY_SIZE` | Hard Negative Pool의 최대 크기 (예: 1000)                             |
| `HARD_NEGATIVE_RATIO`       | 전체 배치 중 Hard Negative로 채울 비율 (예: 0.2이면 전체 배치 중 20%)   |
| `ACCUMULATION_STEPS`        | 배치 크기 누적을 위한 step 수 (`BATCH_SIZE` × `ACCUMULATION_STEPS`) |

**💻 실행 방법**

```bash
python run.py
```



##### 1-2. Oversampling Training

클래스 불균형 문제를 해결하기 위한 방식입니다.

- `BALANCED_BATCH=True`로 설정하면, 각 클래스에서 균등하게 샘플을 뽑아 배치를 구성합니다.
- `HARD_NEGATIVE_MEMORY_SIZE`와 `HARD_NEGATIVE_RATIO`를 0으로 설정해야 Oversampling만 적용됩니다.
- `trainer.predict()`을 **주석 처리** 후에, `trainer.train()`를 **주석 해제**한 상태로 실행합니다.

###### 🔧 주요 설정 인자

| 인자명                                         | 설명                                               |
|-----------------------------------------------|----------------------------------------------------|
| `TRAIN_TRANSFORM` | Train 시 사용 될 Transforms (`Albumentations.Compose` 활용)       |
| `TEST_TRANSFORM` | Validation 시 사용 될 Transforms (`Albumentations.Compose` 활용)       |
| `BALANCED_BATCH`                               | `True`로 설정 시 Oversampling 사용                  |
| `HARD_NEGATIVE_MEMORY_SIZE`, `HARD_NEGATIVE_RATIO` | 둘 다 `0`으로 설정해야 Oversampling만 적용됩니다     |

###### 🧪 실행 예시

```bash
python run.py
```

</details>

### InternImage 모델

<details>

##### 1-1. Hard Negative Sample Training

- 어려운 샘플(hard samples)에 집중하여 학습 성능을 향상시키는 방법입니다.
- Hard Negative Sample은 최근 학습 중 오분류된 데이터들 중 높은 손실을 기록한 샘플들로 구성되며, `HARD_NEGATIVE_MEMORY_SIZE`만큼 저장됩니다.
- 전체 배치에서 `HARD_NEGATIVE_RATIO` 만큼의 샘플을 Hard Negative에서 선택하며, 나머지는 클래스 균등 샘플링으로 채워집니다.
- 전체 배치에서 `HARD_NEGATIVE_RATIO`에 따른 Hard Negative 샘플과, 그 외 클래스 균등 샘플링이 채워지는 기준은 `BATCH_SIZE` × `ACCUMULATION_STEPS` 크기를 기준으로 채워집니다.

**🔧 주요 설정 인자 (코드 내부에서 수정)**

| 인자명                     | 설명                                                                 |
|---------------------------|----------------------------------------------------------------------|
| `TRAIN_TRANSFORM` | Train 시 사용 될 Transforms (`Albumentations.Compose` 활용)       |
| `TEST_TRANSFORM` | Validation 시 사용 될 Transforms (`Albumentations.Compose` 활용)       |
| `HARD_NEGATIVE_MEMORY_SIZE` | Hard Negative Pool의 최대 크기 (예: 1000)                             |
| `HARD_NEGATIVE_RATIO`       | 전체 배치 중 Hard Negative로 채울 비율 (예: 0.2이면 전체 배치 중 20%)   |
| `ACCUMULATION_STEPS`        | 배치 크기 누적을 위한 step 수 (`BATCH_SIZE` × `ACCUMULATION_STEPS`) |

**💻 실행 방법**

models/internimage 폴더로 이동 후, 
```bash
python hardnegative_internimage.py
```

</details>

---

### 2. 추론 (Inference)
학습한 모델을 이용하여 테스트 데이터를 예측합니다. 

※ Test 시에는 TEST_TRANSFORM에 RandomCenterCrop이 포함되지 않도록 반드시 주석 처리해야 합니다.

#### TIMM 모델

<details>

- `trainer.train()`을 **주석 처리** 후에, `trainer.predict()`를 **주석 해제**한 상태로 실행합니다.

##### ✅ 필수 설정

| 인자명         | 설명                                                                 |
|----------------|----------------------------------------------------------------------|
| `TEST_TRANSFORM` | Inference 시 사용 될 Transforms (`Albumentations.Compose` 활용)       |
| `TRAINED_PATH` | 학습된 모델이 저장된 경로 (예: `./experiments/your_model.pth`)       |

##### 🧪 실행 예시

```bash
python run.py
```

</details>



#### InternImage 모델
<details>
  
- `trainer.train()`을 **주석 처리** 후에, `trainer.predict()`를 **주석 해제**한 상태로 실행합니다.

##### ✅ 필수 설정 (코드 내부에서 설정)

| 인자명         | 설명                                                                 |
|----------------|----------------------------------------------------------------------|
| `TEST_TRANSFORM` | Inference 시 사용 될 Transforms (`Albumentations.Compose` 활용)       |
| `TRAINED_PATH` | 학습된 모델이 저장된 경로 (예: `./experiments/your_model.pth`)       |

##### 🧪 실행 예시

models/internimage 폴더로 이동 후, 
```bash
python inference_internimage.py
```

</details>
