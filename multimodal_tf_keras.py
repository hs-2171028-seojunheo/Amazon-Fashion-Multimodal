import os
import pandas as pd
import numpy as np
import cv2

# TensorFlow Hub 모델 다운로드 경로 오류 해결을 위해 프로젝트 내 폴더로 지정
os.environ["TFHUB_CACHE_DIR"] = os.path.join(os.getcwd(), "tfhub_modules")
import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. 하이퍼파라미터 및 설정
# ==========================================
CSV_FILE = 'fashion_train_subset_2_with_images.csv'
BATCH_SIZE = 16
EPOCHS_PHASE1 = 2
EPOCHS_PHASE2 = 5
LR_PHASE1 = 1e-3
LR_PHASE2 = 1e-5

print(f"TensorFlow Version: {tf.__version__}")
print(f"Eager Execution: {tf.executing_eagerly()}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# ==========================================
# 2. 데이터 제너레이터 (Keras Sequence)
# 파이토치의 DataLoader 역할을 완벽히 대체합니다.
# ==========================================
class FashionDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size=32, shuffle=True):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)
            
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
        
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
            
    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        batch_df = self.df.iloc[batch_indices]
        
        texts = batch_df['input_text'].astype(str).values
        prices = batch_df['price_clean'].values.reshape(-1, 1).astype(np.float32)
        misses = batch_df['price_missing'].values.reshape(-1, 1).astype(np.float32)
        cats = batch_df['category_id'].values.reshape(-1, 1).astype(np.int32)
        targets = batch_df['target'].values.astype(np.float32)
        
        images = []
        for path in batch_df['image_path']:
            try:
                img = cv2.imread(str(path).replace("\\", "/"))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                images.append(img / 255.0)
            except:
                images.append(np.zeros((224, 224, 3), dtype=np.float32))
        images = np.array(images, dtype=np.float32)
        
        # 다중 입력 (Multi-Input)
        X = {
            'text': texts,
            'image': images,
            'price': prices,
            'miss': misses,
            'category': cats
        }
        # 다중 출력 (Multi-Task: 통홥, 텍스트 단독, 이미지 단독 시험지)
        Y = {
            'fused_out': targets,
            'text_out': targets,
            'image_out': targets
        }
        return X, Y

# ==========================================
# 3. 커스텀 레이어 (GMU 및 모달리티 드롭아웃)
# ==========================================
class ModalityDropout(layers.Layer):
    def __init__(self, p=0.3, **kwargs):
        super().__init__(**kwargs)
        self.p = p

    def call(self, inputs, training=None):
        t, i, tab = inputs
        if not training:
            return t, i, tab
        # 훈련 중에 30% 확률로 특정 모달리티 정보를 차단 (게으른 학습 방지)
        batch_size = tf.shape(t)[0]
        rand = tf.random.uniform((batch_size, 3))
        mask = tf.cast(rand >= self.p, tf.float32)
        return t * mask[:, 0:1], i * mask[:, 1:2], tab * mask[:, 2:3]

class ThreeWayGMU(layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = layers.Dense(dim, activation='relu')
        self.dense2 = layers.Dense(3, activation='softmax')

    def call(self, inputs):
        t, i, tab = inputs
        concat = tf.concat([t, i, tab], axis=-1)
        gates = self.dense2(self.dense1(concat)) # 비중(가중치) 계산
        fused = (t * gates[:, 0:1]) + (i * gates[:, 1:2]) + (tab * gates[:, 2:3])
        return fused, gates

# ==========================================
# 4. Keras 다중 입출력 모델 조립 (Functional API)
# ==========================================
def build_keras_model(num_cat, hidden_dim=256):
    # 1) 입력층 정의
    text_input = layers.Input(shape=(), dtype=tf.string, name='text')
    image_input = layers.Input(shape=(224, 224, 3), dtype=tf.float32, name='image')
    price_input = layers.Input(shape=(1,), dtype=tf.float32, name='price')
    miss_input = layers.Input(shape=(1,), dtype=tf.float32, name='miss')
    cat_input = layers.Input(shape=(1,), dtype=tf.int32, name='category')

    # 2) 전이학습 백본 (BPM L08 강의의 TF Hub 적용)
    text_hub = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4", trainable=False, name="text_hub")
    image_hub = hub.KerasLayer("https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1", trainable=False, name="image_hub")

    # 3) 특징 추출
    t_feat_raw = text_hub(text_input)
    i_feat_raw = image_hub(image_input)
    
    cat_emb = layers.Embedding(input_dim=num_cat, output_dim=32)(cat_input)
    cat_emb = layers.Flatten()(cat_emb)
    tab_concat = tf.concat([price_input, miss_input, cat_emb], axis=1)
    
    # 4) 차원 맞추기
    t_feat = layers.Dense(hidden_dim, activation='relu')(t_feat_raw)
    i_feat = layers.Dense(hidden_dim, activation='relu')(i_feat_raw)
    tab_feat = layers.Dense(hidden_dim, activation='relu')(tab_concat)

    # 5) 멀티태스크: 독립 시험지 (이미지 전용, 텍스트 전용)
    out_text = layers.Dense(1, activation='sigmoid')(t_feat) * 4.0 + 1.0
    out_text = layers.Layer(name='text_out')(out_text) # 이름 지정
    
    out_image = layers.Dense(1, activation='sigmoid')(i_feat) * 4.0 + 1.0
    out_image = layers.Layer(name='image_out')(out_image)

    # 6) 융합 및 최종 점수 (GMU)
    t_drop, i_drop, tab_drop = ModalityDropout(0.3)([t_feat, i_feat, tab_feat])
    fused, gates = ThreeWayGMU(hidden_dim)([t_drop, i_drop, tab_drop])
    
    out_fused = layers.Dense(1, activation='sigmoid')(fused) * 4.0 + 1.0
    out_fused = layers.Layer(name='fused_out')(out_fused)

    # 모델 생성
    model = Model(
        inputs=[text_input, image_input, price_input, miss_input, cat_input],
        outputs=[out_fused, out_text, out_image]
    )
    return model

# ==========================================
# 5. 실행부
# ==========================================
def main():
    print("1. Data loading...")
    df = pd.read_csv(CSV_FILE).fillna({"input_text": "No review"})
    le = LabelEncoder()
    df["category_id"] = le.fit_transform(df["sub_category"].astype(str))
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    train_gen = FashionDataGenerator(train_df, batch_size=BATCH_SIZE, shuffle=True)
    val_gen = FashionDataGenerator(val_df, batch_size=BATCH_SIZE, shuffle=False)

    print("2. Building Keras Model...")
    model = build_keras_model(num_cat=df["category_id"].nunique())
    
    # 콜백 설정 (최고 성능 모델 자동 저장)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "best_keras_multitask.h5", 
        monitor="val_fused_out_mae", # 최종 통합 점수의 오차 기준
        save_best_only=True, 
        mode="min",
        verbose=1
    )

    # ================================
    # PHASE 1: 백본 동결 학습 (Feature Extraction)
    # ================================
    print("\nPHASE 1: Feature Extraction (Backbone Frozen)")
    # 컴파일 (Multi-task Loss 적용: 통합점수 1.0, 개별점수 0.5 비중)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR_PHASE1),
        loss={'fused_out': 'mse', 'text_out': 'mse', 'image_out': 'mse'},
        loss_weights={'fused_out': 1.0, 'text_out': 0.5, 'image_out': 0.5},
        metrics={'fused_out': 'mae'}
    )
    
    model.fit(
        train_gen, 
        validation_data=val_gen, 
        epochs=EPOCHS_PHASE1, 
        callbacks=[checkpoint]
    )

    # ================================
    # PHASE 2: 전체 파인튜닝 (Full Fine-Tuning)
    # ================================
    print("\nPHASE 2: Full Fine-tuning (All Layers Unfrozen)")
    # 백본 잠금 해제
    model.get_layer("text_hub").trainable = True
    model.get_layer("image_hub").trainable = True
    
    # 미세 조정을 위해 아주 작은 학습률로 재컴파일
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR_PHASE2),
        loss={'fused_out': 'mse', 'text_out': 'mse', 'image_out': 'mse'},
        loss_weights={'fused_out': 1.0, 'text_out': 0.5, 'image_out': 0.5},
        metrics={'fused_out': 'mae'}
    )
    
    model.fit(
        train_gen, 
        validation_data=val_gen, 
        epochs=EPOCHS_PHASE2, 
        callbacks=[checkpoint]
    )
    
    print("\nTraining Complete! The best model has been saved as 'best_keras_multitask.h5'.")

if __name__ == "__main__":
    main()
