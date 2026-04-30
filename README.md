# 🛍️ 아마존 패션 멀티모달 평점 예측 프로젝트 (Amazon Fashion Multimodal)

본 프로젝트는 아마존 패션 데이터셋을 활용하여 이미지, 리뷰 텍스트, 정형 데이터(가격, 카테고리)를 결합해 상품 평점을 예측하는 멀티모달 딥러닝 시스템입니다. 3-Way GMU(Gated Multimodal Unit)와 Targeted Dropout 기법을 적용하여 성능을 극대화했습니다.

---

## 🚀 1. 모델 발전 단계 (Step-by-Step Evolution)

1. **V1: Baseline (Full Fine-tuning)**: 2단계 전이학습 (MAE 1.5 수준). '가격'에만 의존하는 Shortcut Learning 발생.
2. **V2: Multitask Learning**: 이미지/텍스트 독립 Loss 추가 (MAE 0.35 달성). '텍스트' 의존증 발생.
3. **V3: Targeted Dropout (PyTorch)**: 텍스트 80% 마스킹 기법 적용. 모든 모달리티를 균형 있게 학습.
4. **V4: Colab-Optimized (TensorFlow/Keras)**: 강의(BPM) 내용에 맞춘 Keras 버전 구현 및 코랩 환경 최적화.

---

## 🛠️ 2. 주요 파일 설명

*   `multimodal_colab_targeted_dropout_tf.py`: **[추천]** 코랩 GPU 환경에서 즉시 실행 가능한 TensorFlow/Keras 최신 버전
*   `multimodal_targeted_dropout.py`: 로컬/코랩용 PyTorch 최신 버전
*   `multimodal_multitask_finetuning.py`: MAE 0.35를 기록한 멀티태스크 학습 버전
*   `download_fashion_images.py`: 이미지 수집 및 매핑 자동화 스크립트
*   `inference.py`: 학습 완료된 모델을 활용한 실시간 평점 예측 도구

---

## 💻 3. Google Colab 실행 가이드 (TensorFlow 버전)

가장 최신 기법인 **V4 (TensorFlow Keras)** 버전을 사용하여 코랩에서 학습하는 방법입니다.

### Step 1: 데이터 준비
1. 구글 드라이브 상단에 `BigData` 폴더를 생성합니다.
2. 폴더 안에 `fashion_train_subset_2_with_images.csv`와 `images/` 폴더를 업로드합니다.

### Step 2: 코랩 환경 설정
1. **런타임 유형 변경**에서 **T4 GPU**를 선택합니다.

### Step 3: 실행
1. 첫 번째 셀에서 라이브러리를 설치합니다:
   ```python
   !pip install transformers scikit-learn pandas pillow tqdm tensorflow
   ```
2. 두 번째 셀에 `multimodal_colab_targeted_dropout_tf.py` 코드 전체를 복사/붙여넣기한 뒤 실행합니다.
3. 실행 도중 구글 드라이브 마운트 팝업이 뜨면 **'허용'**을 클릭합니다.

---

## 🏆 4. 최종 성과
*   **최고 성능**: MAE **0.3499** (평점 5점 만점 기준)
*   **핵심 기술**: 3-way GMU, Targeted Dropout(텍스트 마스킹), Weighted MSE Loss, Cosine Decay LR.
