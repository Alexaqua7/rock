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

| 팀원                         | 역할               |
|------------------------------|--------------------|
| DonghwanSeo <br> <img src="https://github.com/user-attachments/assets/f1a3b705-6e42-433e-9e00-9f9243d00c07" width="80"/> | TTA, 앙상블, 모델 실험 |
| aqua3g <br> <img src="https://github.com/user-attachments/assets/3f9bb821-d6bc-47e0-9cb8-d175bbfd107a" width="80"/> | Hard Negative Sampling, Oversampling, 모델 실험 |


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
- **Augmentations**: `RandomCenterCrop`, `HorizontalFlip`, `VerticalFlip`, `CLAHE`, 'Gaussian Noise'
- **Loss Function**: `CrossEntropyLoss`, `LabelSmoothing`
- **Optimizer**: `AdamW`, `Adam`
- **Scheduler**: `CosineAnnealingLR`, `StepLR`

---

## 🏆 성능 요약

| 모델명                          | Macro-F1 (Validation 기준) | 특징 요약                         |
|--------------------------------|-----------------------------|------------------------------------|
| InternImage-XL                 | 91.68%                      | Hard Negative 사용                |
| DaViT_base                     | 90.77%                      | Hard Negative 사용                |
| Mamba_out                      | 90.89%                      | Oversampling 사용                 |
| 🧪 Ensemble                    | **93.29% (Public 기준)**     | TTA + Soft-Ensemble               |

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
