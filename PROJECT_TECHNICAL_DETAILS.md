# 🛍️ 패션 멀티모달 평점 예측 프로젝트 기술 리포트 (Technical Report)

이 문서는 본 프로젝트에서 구현된 인공지능 모델의 구조와 학습 전략, 그리고 고도화 과정을 팀원들과 공유하기 위해 작성되었습니다.

---

## 1. 프로젝트 개요 (Overview)
*   **목표**: 아마존 패션 데이터(3.8만 건)를 활용하여 상품의 사진, 리뷰 텍스트, 정형 데이터(가격, 카테고리)를 결합해 평점(1~5점)을 예측함.
*   **핵심 기술**: 멀티모달 딥러닝(Multimodal Deep Learning), 전이학습(Transfer Learning), 게이트 메커니즘(GMU).

---

## 2. 모델 아키텍처 (Model Architecture)

본 모델은 3가지 서로 다른 형태의 데이터를 동시에 처리하는 **'3-Way 퓨전 아키텍처'**를 가집니다.

1.  **텍스트 인코더 (Text Encoder)**:
    *   **모델**: RoBERTa-base (허깅페이스 사전학습 모델)
    *   **역할**: 상품 리뷰 텍스트의 문맥적 의미를 추출하여 768차원의 벡터로 변환.
2.  **이미지 인코더 (Image Encoder)**:
    *   **모델**: EfficientNet-B0 (구글 사전학습 모델)
    *   **역할**: 상품 사진으로부터 질감, 색상, 형태 등의 시각적 특징을 1280차원의 벡터로 변환.
3.  **정형 데이터 인코더 (Tabular Encoder)**:
    *   **구성**: 가격 데이터 + 가격 누락 여부 + 카테고리 임베딩(Category Embedding).
    *   **역할**: 상품의 객관적 스펙 정보를 수치화함.
4.  **융합 모듈 (Fusion - 3-Way GMU)**:
    *   **기술**: Gated Multimodal Unit (GMU).
    *   **역할**: 텍스트, 이미지, 정형 데이터 중 어떤 정보가 평점 예측에 더 중요한지 스스로 판단하여 비중을 조절함 (Attention 메커니즘의 일종).

---

## 3. 학습 전략 (Training Strategy)

### Phase 1: Feature Extraction (전이학습 기초)
*   강의(BPM L08)에서 배운 **백본 동결(Backbone Freezing)** 기법 적용.
*   이미 검증된 거대 모델(RoBERTa, EfficientNet)의 지식은 그대로 두고, 상단의 융합 레이어만 빠르게 학습시켜 초기 오차를 줄임.

### Phase 2: Full Fine-tuning (미세 조정)
*   모든 레이어의 잠금을 해제하고 아주 낮은 학습률(1e-5)로 전체를 다시 학습.
*   패션 데이터셋의 미세한 특징(예: 리뷰의 뉘앙스, 사진의 스타일)을 모델이 깨닫게 만듦.

---

## 4. 고도화 과정 (V1 vs V2 비교)

| 구분 | V1: Full Fine-tuning (Baseline) | V2: Multitask Learning (Enhanced) |
| :--- | :--- | :--- |
| **적용 파일** | `multimodal_full_finetuning.py` | `multimodal_multitask_finetuning.py` |
| **핵심 기법** | 단일 손실 함수 (Single Loss) | **멀티태스크 손실 함수 (Multi-task Loss)** |
| **동작 원리** | 최종 결과물만 보고 학습함 | **독립 시험지(텍스트 전용, 이미지 전용)**를 추가하여 개별 공부를 강제함 |
| **해결 과제** | 정형 데이터에만 의존하는 '게으른 학습' 방지 | 이미지와 텍스트의 특징 추출 능력을 극대화하여 MAE 성능 개선 |
| **Dropout** | 15% (일반 수준) | **30% (가혹한 조건)** - 정형 데이터 없이도 예측하도록 훈련 |

---

## 5. 강의(BPM)와의 연결성

*   **L01~L03**: 파이토치 기반의 효율적인 데이터 파이프라인(`Dataset`, `DataLoader`) 구축.
*   **L04~L06**: CNN의 특징 추출 원리를 `EfficientNet`을 통해 심화 구현.
*   **L07**: `CosineAnnealingLR` 스케줄러와 `Weight Decay`를 통한 오버피팅 방지 및 최적화.
*   **L08**: 사전학습 모델 로드 및 2단계 파인튜닝 전략의 완벽한 실전 적용.

---

## 6. 향후 계획 (Next Steps)
현재 구현된 **Multitask Learning** 모델의 학습이 완료되면, 각 모달리티(텍스트/이미지)가 예측에 기여하는 비중이 얼마나 늘어났는지 분석하고, 최종적으로 가장 낮은 MAE를 기록한 모델을 서빙(Inference) 단계로 이행할 예정입니다.
