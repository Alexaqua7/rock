# 상세 설정 인자 설명서

본 문서에서는 프로젝트의 모든 설정 인자를 상세히 설명합니다.

---

## 1. 기본 설정

| 인자명            | 설명                                                         | 기본값  |
|------------------|-------------------------------------------------------------|--------|
| `SEED`           | 랜덤 시드 설정                                              | 41     |
| `TEST_SIZE`      | 학습/검증 분할 시 테스트 데이터 비율                        | 0.2    |
| `MODEL_NAME`     | 사용할 모델 이름으로 반드시 TIMM 내에 존재하는 모델이어야 함 (예: 'davit_base', 'mambaout_base_plus_rw.sw_e150_r384_in12k_ft_in1k')                         | 'davit_base' |
| `IMG_SIZE`       | 이미지 크기 (가로/세로)                                     | 224    |
| `EPOCHS`         | 학습 Epoch 수                                               | 15      |
| `BATCH_SIZE`     | 배치 크기                                                  | 64     |
| `NUM_WORKERS`    | DataLoader의 worker 수                                     | 16     |
| `LEARNING_RATE`  | 학습률                                                    | 3e-4   |

---

## 2. 학습 관련 설정

| 인자명                     | 설명                                                                                   | 기본값    |
|---------------------------|---------------------------------------------------------------------------------------|----------|
| `HARD_NEGATIVE_MEMORY_SIZE` | Hard Negative 샘플 저장 용량 (0이면 사용 안함)                                         | 1000     |
| `HARD_NEGATIVE_RATIO`       | 배치 중 Hard Negative 샘플 비율                                                       | 0.2      |
| `BALANCED_BATCH`            | 클래스 균등 샘플링 여부 (True이면 Oversampling)                                       | True     |
| `ACCUMULATION_STEPS`        | Gradient accumulation step 수 (Effective Batch Size = BATCH_SIZE × ACCUMULATION_STEPS) | 8        |

---

## 3. 옵티마이저 및 스케줄러

| 인자명       | 설명                         | 기본값        |
|-------------|-----------------------------|--------------|
| `OPTIMIZER` | optimizer 종류 (adam, adamw) | 'adam'       |
| `WARM_UP`   | warm-up epoch 수 (0이면 사용 안함) | 0            |
| `SCHEDULER` | learning rate scheduler 종류   | 'cosineannealing' |
| `ETA_MIN`   | cosine annealing의 최소 학습률 | 1e-8         |

---

## 4. 추론 관련

| 인자명       | 설명                          | 기본값        |
|-------------|------------------------------|--------------|
| `TRAINED_PATH` | 학습된 모델 파일 경로 (추론 시 필수) | '' (빈값)    |

---

## 5. 실행 방법

- 학습 실행: `python run.py` (`trainer.train()` 활성화)  
- 추론 실행: `python run.py` (`trainer.predict()` 활성화 및 `TRAINED_PATH` 설정 필요, TEST_TRANSFORM에 RandomCenterCrop 반드시 주석 처리 후 실행)

---

필요에 따라 각 인자의 설명을 더 추가하거나 예시 값을 갱신하여 사용하세요.
