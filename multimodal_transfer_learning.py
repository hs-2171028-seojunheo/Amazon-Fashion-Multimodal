import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoTokenizer, RobertaModel, AutoImageProcessor, EfficientNetModel
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm

# ==========================================
# 1. 하이퍼파라미터 및 설정
# ==========================================
CSV_FILE = 'subset_100_with_images.csv'
BATCH_SIZE = 8
EPOCHS_PHASE1 = 3  # Feature Extraction (Base frozen)
EPOCHS_PHASE2 = 5  # Fine-tuning (Base unfrozen)
LR_PHASE1 = 1e-3   # High LR for the new head
LR_PHASE2 = 1e-5   # Very low LR for fine-tuning

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 기기: {DEVICE}")

# ==========================================
# 2. 데이터셋 클래스 (기존과 동일)
# ==========================================
class FashionMultimodalDataset(Dataset):
    def __init__(self, df, text_tokenizer, image_processor, max_len=128):
        self.df = df.reset_index(drop=True)
        self.text_tokenizer = text_tokenizer
        self.image_processor = image_processor
        self.max_len = max_len
        
        # 정형 데이터 전처리
        self.prices = torch.tensor(self.df['price_clean'].values, dtype=torch.float32)
        self.categories = torch.tensor(self.df['sub_category_encoded'].values, dtype=torch.long)
        self.ratings = torch.tensor(self.df['target'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. 텍스트 처리
        text = str(row['input_text'])
        encoding = self.text_tokenizer(
            text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt'
        )
        
        # 2. 이미지 처리
        img_path = row['image_path']
        try:
            image = Image.open(img_path).convert("RGB")
            # Hugging Face 전처리기로 변환 (내부적으로 Rescaling 등 자동 적용)
            pixel_values = self.image_processor(images=image, return_tensors='pt')['pixel_values'].squeeze(0)
        except:
            pixel_values = torch.zeros((3, 224, 224))
            
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'pixel_values': pixel_values,
            'price': self.prices[idx],
            'category': self.categories[idx],
            'rating': self.ratings[idx]
        }

# ==========================================
# 3. Transfer Learning 기반 모델 구조
# ==========================================
class FashionTransferModel(nn.Module):
    def __init__(self, num_categories, hidden_dim=512):
        super(FashionTransferModel, self).__init__()
        
        # [Part 1] Base Models (Backbones)
        # 1. 텍스트: RoBERTa
        self.text_backbone = RobertaModel.from_pretrained('roberta-base')
        
        # 2. 이미지: EfficientNet-B0 (강의 Part 2 Hugging Face 방식)
        self.image_backbone = EfficientNetModel.from_pretrained("google/efficientnet-b0")
        
        # 3. 메타데이터 (Tabular)
        self.category_emb = nn.Embedding(num_categories, 16)
        
        # [Part 2] New Head (Top Layers) - 강의 내용의 Dense 층에 해당
        # GMU or Simple Concat
        self.fusion_layer = nn.Linear(768 + 1280 + 17, hidden_dim) 
        
        self.prediction_head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # 평점 예측 (Regression)
        )

    def freeze_backbones(self):
        """Part 3: 가중치 동결 (base_model.trainable = False 에 해당)"""
        print(">>> Backbones Frozen (Phase 1)")
        for param in self.text_backbone.parameters():
            param.requires_grad = False
        for param in self.image_backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbones(self):
        """Part 3: 가중치 해제 (Fine-tuning 준비)"""
        print(">>> Backbones Unfrozen (Phase 2)")
        for param in self.text_backbone.parameters():
            param.requires_grad = True
        for param in self.image_backbone.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask, pixel_values, price, category):
        # Features extraction
        t_feat = self.text_backbone(input_ids, attention_mask).pooler_output
        # Hugging Face 모델은 pooler_output (GAP 적용됨) 반환
        i_feat = self.image_backbone(pixel_values).pooler_output.flatten(1)
        c_feat = self.category_emb(category)
        p_feat = price.unsqueeze(1)
        
        # Concat & Prediction
        combined = torch.cat([t_feat, i_feat, c_feat, p_feat], dim=1)
        fused = self.fusion_layer(combined)
        return self.prediction_head(fused).squeeze()

# ==========================================
# 4. 학습 실행
# ==========================================
def train_model():
    # 데이터 로드 및 전처리
    df = pd.read_csv(CSV_FILE)
    df = df.dropna(subset=['image_path'])
    le = LabelEncoder()
    df['sub_category_encoded'] = le.fit_transform(df['sub_category'].astype(str))
    
    # [강의 방식] Hugging Face 전처리기 사용 (Rescaling, Resize 자동 처리)
    image_processor = AutoImageProcessor.from_pretrained("google/efficientnet-b0")
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(FashionMultimodalDataset(train_df, tokenizer, image_processor), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(FashionMultimodalDataset(val_df, tokenizer, image_processor), batch_size=BATCH_SIZE)

    model = FashionTransferModel(num_categories=len(le.classes_)).to(DEVICE)
    criterion = nn.MSELoss()

    # --- PHASE 1: Feature Extraction (Freeze) ---
    model.freeze_backbones()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_PHASE1)
    
    print("\n[Phase 1] Training Top Layers Only...")
    for epoch in range(EPOCHS_PHASE1):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"P1 Epoch {epoch+1}"):
            optimizer.zero_grad()
            out = model(batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), 
                        batch['pixel_values'].to(DEVICE), batch['price'].to(DEVICE), batch['category'].to(DEVICE))
            loss = criterion(out, batch['rating'].to(DEVICE))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"P1 Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

    # --- PHASE 2: Fine-tuning (Unfreeze) ---
    model.unfreeze_backbones()
    # 낮은 학습률로 모든 파라미터 업데이트
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_PHASE2)
    
    print("\n[Phase 2] Fine-tuning All Layers...")
    for epoch in range(EPOCHS_PHASE2):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"P2 Epoch {epoch+1}"):
            optimizer.zero_grad()
            out = model(batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), 
                        batch['pixel_values'].to(DEVICE), batch['price'].to(DEVICE), batch['category'].to(DEVICE))
            loss = criterion(out, batch['rating'].to(DEVICE))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"P2 Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

if __name__ == "__main__":
    train_model()
