import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoTokenizer, RobertaModel
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import random
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings("ignore")


# ==========================================
# 1. 하이퍼파라미터 및 RunPod 환경 설정
# ==========================================
TEAM_NAME = "FromBottom" # ★ TODO: 본인 팀명으로 변경하세요
WORKSPACE_DIR = f"/workspace/{TEAM_NAME}"

# 로컬(Windows) 환경에서 개발/테스트 중인 경우, 안전을 위해 현재 디렉토리 사용
if os.name == 'nt':
    WORKSPACE_DIR = os.getcwd()
else:
    # RunPod 환경에서 타 팀 디렉토리 접근 차단용
    if not os.path.exists(WORKSPACE_DIR):
        os.makedirs(WORKSPACE_DIR, exist_ok=True)
    # 보안 조치: 허용된 WORKSPACE_DIR 밖으로 경로 이탈 금지
    os.chdir(WORKSPACE_DIR)

CSV_FILE = os.path.join(WORKSPACE_DIR, 'fashion_train_subset_2_with_images.csv')
IMAGE_DIR = os.path.join(WORKSPACE_DIR, 'images')

# ★ 파이프라인 사전 동작 확인용 테스트 모드 (GPU 할당 전 필수 과정)
# 조교 승인 및 RunPod GPU 할당 완료 후, 실제 학습 시 False로 변경하세요!
MOCK_TEST_MODE = False 

BATCH_SIZE = 16 if not MOCK_TEST_MODE else 4
ACCUMULATION_STEPS = 1 if not MOCK_TEST_MODE else 1
EPOCHS = 5 if not MOCK_TEST_MODE else 2
PHASE_1_EPOCHS = 2 if not MOCK_TEST_MODE else 1

DEVICE = torch.device('cuda' if torch.cuda.is_available() and not MOCK_TEST_MODE else 'cpu')
print(f"-> 현재 구동 기기(DEVICE): {DEVICE}")
print(f"-> 데이터 저장 및 작업 경로: {WORKSPACE_DIR}")
if MOCK_TEST_MODE:
    print("\n[안내] MOCK_TEST_MODE가 켜져 있습니다.")
    print("       소량의 샘플 데이터(100개)로 파이프라인(로딩->전처리->학습->저장) 전체 흐름을 테스트합니다.")
    print("       이 테스트가 오류 없이 정상 동작해야 GPU 할당을 요청할 수 있습니다.\n")

# ==========================================
# 2. Tokenizer & Image Transform
# ==========================================
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

v3_weights = models.MobileNet_V3_Large_Weights.DEFAULT
base_transform = v3_weights.transforms()

train_transform = transforms.Compose([
    transforms.TrivialAugmentWide(),
    transforms.RandomHorizontalFlip(),
    base_transform
])
val_transform = base_transform

# ==========================================
# 3. 데이터셋 클래스
# ==========================================
class AmazonFashionFullDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform if transform else val_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Text
        text = str(row["input_text"])
        enc = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        
        # Image (RunPod 내 상대경로 호환성 강화)
        img_filename = os.path.basename(str(row["image_path"]).replace("\\", "/"))
        img_path = os.path.join(IMAGE_DIR, img_filename)
        
        try:
            img = Image.open(img_path).convert("RGB")
            pixel_values = self.transform(img)
        except:
            pixel_values = torch.zeros(3, 224, 224)

        # Tabular
        price = torch.tensor([row["price_clean"]], dtype=torch.float32)
        price_missing = torch.tensor([row["price_missing"]], dtype=torch.float32)
        category = torch.tensor(row["category_id"], dtype=torch.long)
        
        # Target
        target = torch.tensor(row["target"], dtype=torch.float32)

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "pixel_values": pixel_values,
            "price": price,
            "price_missing": price_missing,
            "category_id": category,
            "target": target
        }

# ==========================================
# 4. 모델 아키텍처 (MobileNet-V3 하이브리드)
# ==========================================
class TargetedModalityDropout(nn.Module):
    def __init__(self, text_drop_p=0.8, general_drop_p=0.2): 
        super().__init__()
        self.text_drop_p = text_drop_p
        self.general_drop_p = general_drop_p

    def forward(self, t, i, tab):
        if not self.training: return t, i, tab
        mask = torch.ones((t.size(0), 3), device=t.device)
        for idx in range(t.size(0)):
            if random.random() < self.text_drop_p:
                mask[idx, 0] = 0 
            elif random.random() < self.general_drop_p:
                mask[idx, random.randint(1, 2)] = 0
        return t * mask[:, 0].unsqueeze(1), i * mask[:, 1].unsqueeze(1), tab * mask[:, 2].unsqueeze(1)

class ThreeWayGMU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim * 3, dim), nn.ReLU(), nn.Linear(dim, 3), nn.Softmax(dim=1))
    def forward(self, t, i, tab):
        weights = self.gate(torch.cat([t, i, tab], dim=1))
        fused = weights[:, 0].unsqueeze(1)*t + weights[:, 1].unsqueeze(1)*i + weights[:, 2].unsqueeze(1)*tab
        return fused, weights

class MultitaskFashionModelV3(nn.Module):
    def __init__(self, num_cat, hidden_dim=256):
        super().__init__()
        self.text_encoder = RobertaModel.from_pretrained("roberta-base")
        self.text_fc = nn.Linear(768, hidden_dim)
        
        mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.image_encoder = mobilenet.features
        self.image_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.image_fc = nn.Linear(960, hidden_dim) 
        
        self.cat_emb = nn.Embedding(num_cat, 32)
        self.tab_fc = nn.Sequential(nn.Linear(1 + 1 + 32, hidden_dim), nn.ReLU())
        
        self.modality_dropout = TargetedModalityDropout(0.8, 0.2)
        self.gmu = ThreeWayGMU(hidden_dim)
        
        self.text_regressor = nn.Linear(hidden_dim, 1)
        self.image_regressor = nn.Linear(hidden_dim, 1)
        self.fused_regressor = nn.Linear(hidden_dim, 1)

    def forward(self, ids, mask, pixels, price, miss, cat):
        t_feat_raw = self.text_fc(self.text_encoder(ids, mask).pooler_output)
        
        img_feat = self.image_encoder(pixels)
        img_feat = self.image_pool(img_feat).view(pixels.size(0), -1)
        i_feat_raw = self.image_fc(img_feat)
        
        tab_feat_raw = self.tab_fc(torch.cat([price, miss, self.cat_emb(cat)], dim=1))
        
        out_text = torch.sigmoid(self.text_regressor(t_feat_raw)).squeeze() * 4 + 1
        out_image = torch.sigmoid(self.image_regressor(i_feat_raw)).squeeze() * 4 + 1
        
        t_feat, i_feat, tab_feat = self.modality_dropout(t_feat_raw, i_feat_raw, tab_feat_raw)
        fused, gates = self.gmu(t_feat, i_feat, tab_feat)
        out_fused = torch.sigmoid(self.fused_regressor(fused)).squeeze() * 4 + 1
        
        return out_fused, out_text, out_image, gates

# ==========================================
# 5. 유틸리티 (학습 및 평가)
# ==========================================
def weighted_mse_loss(pred, target):
    weight_map = {1.0: 4.0, 2.0: 4.0, 3.0: 3.0, 4.0: 2.0, 5.0: 1.0}
    target_rounded = torch.round(target).clamp(1.0, 5.0)
    weights = torch.tensor([weight_map[t.item()] for t in target_rounded], device=target.device)
    return (weights * (pred - target)**2).mean()

def train_epoch(model, loader, optimizer, scheduler, device, acc_steps, epoch):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"Epoch {epoch} Train")
    for i, batch in enumerate(pbar):
        target = batch["target"].to(device)
        
        out_fused, out_text, out_image, _ = model(
            batch["input_ids"].to(device), batch["attention_mask"].to(device), 
            batch["pixel_values"].to(device), batch["price"].to(device), 
            batch["price_missing"].to(device), batch["category_id"].to(device)
        )
        
        loss_fused = weighted_mse_loss(out_fused, target)
        loss_text = weighted_mse_loss(out_text, target)
        loss_image = weighted_mse_loss(out_image, target)
        
        loss = (1.0 * loss_fused + 0.2 * loss_text + 1.2 * loss_image) / acc_steps
        loss.backward()
        
        if (i+1) % acc_steps == 0 or (i+1) == len(loader):
            optimizer.step()
            if scheduler: scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * acc_steps
        pbar.set_postfix({'loss': loss.item() * acc_steps})
    return total_loss / len(loader)

def evaluate(model, loader, device, epoch):
    model.eval()
    total_loss = 0
    preds, targets, gates = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Epoch {epoch} Eval"):
            target = batch["target"].to(device)
            out_fused, out_text, out_image, gate = model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device), 
                batch["pixel_values"].to(device), batch["price"].to(device), 
                batch["price_missing"].to(device), batch["category_id"].to(device)
            )
            
            loss = weighted_mse_loss(out_fused, target)
            total_loss += loss.item()
            preds.extend(out_fused.cpu().numpy())
            targets.extend(batch["target"].numpy())
            gates.extend(gate.cpu().numpy())
            
    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)
    avg_gate = np.mean(gates, axis=0)
    return total_loss / len(loader), mse, mae, avg_gate

# ==========================================
# 6. 메인 실행부
# ==========================================
def main():
    if not os.path.exists(CSV_FILE):
        print(f"오류: {CSV_FILE} 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    print("1. Data loading...")
    df = pd.read_csv(CSV_FILE)
    
    # Mock Test 모드일 경우 100건만 샘플링하여 빠르게 파이프라인 무결성 확인
    if MOCK_TEST_MODE:
        df = df.sample(n=min(100, len(df)), random_state=42)
        
    df = df.fillna({"input_text": "No review"})
    le = LabelEncoder()
    df["category_id"] = le.fit_transform(df["sub_category"].astype(str))
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    train_loader = DataLoader(AmazonFashionFullDataset(train_df, transform=train_transform), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(AmazonFashionFullDataset(val_df, transform=val_transform), batch_size=BATCH_SIZE)
    
    model = MultitaskFashionModelV3(num_cat=df["category_id"].nunique()).to(DEVICE)
    best_mae = float('inf')

    # [Phase 1]
    print("\n--- [Phase 1] Text Encoder Frozen (Image-Driven Learning) ---")
    for param in model.text_encoder.parameters():
        param.requires_grad = False
        
    optimizer_p1 = torch.optim.AdamW([
        {"params": model.image_encoder.parameters(), "lr": 2e-5},
        {"params": [p for n, p in model.named_parameters() if "encoder" not in n and p.requires_grad], "lr": 1e-4}
    ])
    scheduler_p1 = CosineAnnealingLR(optimizer_p1, T_max=PHASE_1_EPOCHS * (len(train_loader) // ACCUMULATION_STEPS))

    for epoch in range(1, PHASE_1_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer_p1, scheduler_p1, DEVICE, ACCUMULATION_STEPS, epoch)
        val_loss, val_mse, val_mae, avg_gate = evaluate(model, val_loader, DEVICE, epoch)
        
        print(f"\n[Phase 1 - Epoch {epoch}] Train Loss: {train_loss:.4f} | Val MAE: {val_mae:.4f}")
        print(f"GMU Gate -> Text: {avg_gate[0]:.2f}, Image: {avg_gate[1]:.2f}, Tabular: {avg_gate[2]:.2f}\n")
        
        # 12시간 제한 규정 대비: 매 에폭마다 주기적 체크포인트 저장
        ckpt_path = os.path.join(WORKSPACE_DIR, f"checkpoint_epoch_{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"✅ 시간 초과 대비 주기적 체크포인트 저장 완료: {ckpt_path}\n")

    # [Phase 2]
    print("\n--- [Phase 2] Text Encoder Unfrozen (Differential LR) ---")
    for param in model.text_encoder.parameters():
        param.requires_grad = True

    optimizer_p2 = torch.optim.AdamW([
        {"params": model.text_encoder.parameters(), "lr": 1e-6},
        {"params": model.image_encoder.parameters(), "lr": 1e-5},
        {"params": [p for n, p in model.named_parameters() if "encoder" not in n], "lr": 1e-4}
    ])
    scheduler_p2 = CosineAnnealingLR(optimizer_p2, T_max=(EPOCHS - PHASE_1_EPOCHS) * (len(train_loader) // ACCUMULATION_STEPS))

    for epoch in range(PHASE_1_EPOCHS + 1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer_p2, scheduler_p2, DEVICE, ACCUMULATION_STEPS, epoch)
        val_loss, val_mse, val_mae, avg_gate = evaluate(model, val_loader, DEVICE, epoch)
        
        print(f"\n[Phase 2 - Epoch {epoch}] Train Loss: {train_loss:.4f} | Val MAE: {val_mae:.4f}")
        print(f"GMU Gate -> Text: {avg_gate[0]:.2f}, Image: {avg_gate[1]:.2f}, Tabular: {avg_gate[2]:.2f}\n")
        
        # 체크포인트 저장
        ckpt_path = os.path.join(WORKSPACE_DIR, f"checkpoint_epoch_{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"✅ 시간 초과 대비 주기적 체크포인트 저장 완료: {ckpt_path}\n")
        
        if val_mae < best_mae:
            best_mae = val_mae
            best_path = os.path.join(WORKSPACE_DIR, "best_runpod_v3_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"🌟 New Best RunPod V3 Model Saved (MAE: {best_mae:.4f})\n")

    print("==================================================================")
    print("🎯 학습이 성공적으로 완료되었습니다!")
    print("규정에 따라 본 코드는 별도의 추론(Inference) 로직을 포함하지 않습니다.")
    print("작업이 끝났으므로 과금을 막기 위해 즉시 RunPod를 Terminated 하시기 바랍니다.")
    print("==================================================================")

if __name__ == "__main__":
    main()
