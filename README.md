# 🛍️ 아마존 패션 멀티모달 평점 예측 프로젝트 (Amazon Fashion Multimodal Rating Prediction)

본 프로젝트는 아마존 패션 데이터셋(3.8만 건)을 활용하여 이미지, 리뷰 텍스트, 정형 데이터를 결합해 상품 평점을 예측하는 최첨단 멀티모달 딥러닝 시스템을 구축하는 과정입니다.

---

## 🚀 1. 프로젝트 아키텍처 (Architecture)

*   **텍스트 인코더**: RoBERTa-base (문맥적 감성 분석)
*   **이미지 인코더**: EfficientNet-B0 (시각적 특징 추출)
*   **정형 데이터 처리**: 가격(Price) 및 카테고리 임베딩(Embedding)
*   **융합 모듈 (Fusion)**: **3-Way GMU (Gated Multimodal Unit)**
    *   각 모달리티의 중요도를 스스로 판단하여 동적으로 가중치 부여 (Attention Mechanism)

---

## 📈 2. 모델 발전 단계 (Step-by-Step Evolution)

### 🟢 V1: Baseline (Full Fine-tuning)
*   **전략**: 2단계 전이학습 (백본 동결 -> 전체 파인튜닝)
*   **결과**: MAE 1.5 수준 정체
*   **발견된 문제**: **'게으른 학습(Shortcut Learning)'** 발생. 모델이 사진과 글을 무시하고 오직 '가격' 정보에만 100% 의존함.

### 🔵 V2: Multitask Learning (성능 돌파)
*   **전략**: 이미지/텍스트 단독 예측 레이어 및 독립 Loss 추가
*   **결과**: **MAE 0.3499 달성 (기존 대비 75% 오차 감소)**
*   **발견된 문제**: 가격 의존증은 고쳤으나, 정보량이 압도적인 **'텍스트'에만 100% 의존**하는 현상 발견. 진정한 멀티모달 융합을 위해 보완 필요.

### 🔴 V3: Targeted Dropout (진정한 멀티모달)
*   **전략**: **텍스트 강제 마스킹(80% Dropout)** 적용
*   **목표**: 모델이 가장 선호하는 텍스트 힌트를 강제로 가려, 사진과 정형 데이터를 필사적으로 활용하게 만듦.

---

## 🛠️ 3. 기술적 의사결정 (Technical Decision)

### Framework: PyTorch vs TensorFlow
*   본 프로젝트는 **PyTorch**를 메인 프레임워크로 채택했습니다.
*   **이유**: 최신 텐서플로우(v2.11+)가 윈도우 네이티브 GPU 가속을 지원하지 않는 한계가 있어, 대규모 멀티모달 학습 효율을 위해 윈도우 GPU 환경을 완벽히 지원하는 파이토치를 선택했습니다. (강의에서 배운 Keras의 핵심 원리는 동일하게 적용됨)

---

## 📚 4. 강의(BPM) 내용과의 연결성

본 프로젝트는 **Big Data Programming** 수업에서 배운 핵심 개념들을 실전에 완벽히 이식했습니다.

*   **L01~L03 (Data Pipeline)**: `Dataset` 및 `DataLoader`를 통한 배치 단위 데이터 처리
*   **L04~L06 (CNN 기초)**: `EfficientNet`을 활용한 고급 특징 추출(Feature Extraction)
*   **L07 (Optimization)**: `Modality Dropout` 및 `CosineAnnealingLR` 최적화 기법 적용
*   **L08 (Transfer Learning)**: 사전학습 모델 로딩 및 단계별 동결/해제 전략 실행

---

## 📁 5. 주요 파일 가이드

*   `multimodal_targeted_dropout.py`: 현재 최우선 실행 중인 텍스트 마스킹 버전 (V3)
*   `multimodal_multitask_finetuning.py`: MAE 0.34를 달성한 멀티태스크 버전 (V2)
*   `multimodal_tf_keras.py`: 강의 내용 기반의 Keras 구현 버전 (구조 교육용)
*   `PROJECT_TECHNICAL_DETAILS.md`: 팀 공유용 기술 상세 리포트

---

## 🏆 6. 최종 성과 요약
*   **최고 성능**: MAE **0.3499** (평점 5점 만점 기준)
*   **주요 성과**: 게으른 학습 문제를 Multi-task Loss와 Targeted Dropout으로 해결하여 텍스트 감성과 가격 지표를 모두 이해하는 지능형 모델 구축 성공.
