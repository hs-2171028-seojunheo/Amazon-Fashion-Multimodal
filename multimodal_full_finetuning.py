import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoTokenizer, RobertaModel, AutoImageProcessor, EfficientNetModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import random
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings("ignore")

# ==========================================
# 1. 하이퍼파라미터 및 경로 설정
# ==========================================
CSV_FILE = 'fashion_train_subset_2_with_images.csv'
BATCH_SIZE = 4
ACCUMULATION_STEPS = 4

EPOCHS_PHASE1 = 2
EPOCHS_PHASE2 = 5

LR_PHASE1 = 1e-3
LR_PHASE2 = 1e-5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ==========================================
# 2. 데이터셋 클래스
# ==========================================
class AmazonFashionFullDataset(Dataset):
    def __init__(self, df, tokenizer, image_processor):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["input_text"])
        enc = self.tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        
        img_path = str(row["image_path"]).replace("\\", "/")
        try:
            img = Image.open(img_path).convert("RGB")
            pixel_values = self.image_processor(images=img, return_tensors='pt')['pixel_values'].squeeze(0)
        except:
            pixel_values = torch.zeros(3, 224, 224)

        price = torch.tensor([row["price_clean"]], dtype=torch.float32)
        price_missing = torch.tensor([row["price_missing"]], dtype=torch.float32)
        category = torch.tensor(row["category_id"], dtype=torch.long)
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
# 3. 모델 아키텍처
# ==========================================
class ModalityDropout(nn.Module):
    def __init__(self, p=0.15):
        super().__init__()
        self.p = p

    def forward(self, t, i, tab):
        if not self.training: return t, i, tab
        mask = torch.ones((t.size(0), 3), device=t.device)
        for idx in range(t.size(0)):
            if random.random() < self.p:
                mask[idx, random.randint(0, 2)] = 0
        return t * mask[:, 0].unsqueeze(1), i * mask[:, 1].unsqueeze(1), tab * mask[:, 2].unsqueeze(1)

class ThreeWayGMU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim * 3, dim), nn.ReLU(), nn.Linear(dim, 3), nn.Softmax(dim=1))
    def forward(self, t, i, tab):
        weights = self.gate(torch.cat([t, i, tab], dim=1))
        fused = weights[:, 0].unsqueeze(1)*t + weights[:, 1].unsqueeze(1)*i + weights[:, 2].unsqueeze(1)*tab
        return fused, weights

class UltimateFashionModel(nn.Module):
    def __init__(self, num_cat, hidden_dim=256):
        super().__init__()
        self.text_encoder = RobertaModel.from_pretrained("roberta-base", use_safetensors=True)
        self.image_encoder = EfficientNetModel.from_pretrained("google/efficientnet-b0", use_safetensors=True)
        self.text_fc = nn.Linear(768, hidden_dim)
        self.image_fc = nn.Linear(1280, hidden_dim)
        self.cat_emb = nn.Embedding(num_cat, 32)
        self.tab_fc = nn.Sequential(nn.Linear(1 + 1 + 32, hidden_dim), nn.ReLU())
        self.modality_dropout = ModalityDropout(0.15)
        self.gmu = ThreeWayGMU(hidden_dim)
        self.regressor = nn.Linear(hidden_dim, 1)

    def freeze_backbones(self):
        for p in self.text_encoder.parameters(): p.requires_grad = False
        for p in self.image_encoder.parameters(): p.requires_grad = False
    
    def unfreeze_backbones(self):
        for p in self.text_encoder.parameters(): p.requires_grad = True
        for p in self.image_encoder.parameters(): p.requires_grad = True

    def forward(self, ids, mask, pixels, price, miss, cat):
        t_feat = self.text_fc(self.text_encoder(ids, mask).pooler_output)
        i_feat = self.image_fc(self.image_encoder(pixels).pooler_output.flatten(1))
        tab_feat = self.tab_fc(torch.cat([price, miss, self.cat_emb(cat)], dim=1))
        t_feat, i_feat, tab_feat = self.modality_dropout(t_feat, i_feat, tab_feat)
        fused, gates = self.gmu(t_feat, i_feat, tab_feat)
        out = torch.sigmoid(self.regressor(fused)).squeeze() * 4 + 1
        return out, gates

# ==========================================
# 4. 유틸리티
# ==========================================
def weighted_mse_loss(pred, target):
    weight_map = {1.0: 4.0, 2.0: 4.0, 3.0: 3.0, 4.0: 2.0, 5.0: 1.0}
    target_rounded = torch.round(target).clamp(1.0, 5.0)
    weights = torch.tensor([weight_map[t.item()] for t in target_rounded], device=target.device)
    return (weights * (pred - target)**2).mean()

def train_epoch(model, loader, optimizer, scheduler, device, acc_steps, epoch, phase_name):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"[{phase_name}] Epoch {epoch} Train")
    for i, batch in enumerate(pbar):
        out, _ = model(batch["input_ids"].to(device), batch["attention_mask"].to(device), 
                       batch["pixel_values"].to(device), batch["price"].to(device), 
                       batch["price_missing"].to(device), batch["category_id"].to(device))
        loss = weighted_mse_loss(out, batch["target"].to(device)) / acc_steps
        loss.backward()
        if (i+1) % acc_steps == 0 or (i+1) == len(loader):
            optimizer.step()
            if scheduler: scheduler.step()
            optimizer.zero_grad()
        total_loss += loss.item() * acc_steps
        pbar.set_postfix({'loss': loss.item() * acc_steps})
    return total_loss / len(loader)

def evaluate(model, loader, device, epoch, phase_name):
    model.eval()
    total_loss = 0
    preds, targets, gates = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"[{phase_name}] Epoch {epoch} Eval"):
            out, gate = model(batch["input_ids"].to(device), batch["attention_mask"].to(device), 
                              batch["pixel_values"].to(device), batch["price"].to(device), 
                              batch["price_missing"].to(device), batch["category_id"].to(device))
            loss = weighted_mse_loss(out, batch["target"].to(device))
            total_loss += loss.item()
            preds.extend(out.cpu().numpy())
            targets.extend(batch["target"].numpy())
            gates.extend(gate.cpu().numpy())
    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)
    avg_gate = np.mean(gates, axis=0)
    return total_loss / len(loader), mse, mae, avg_gate

# ==========================================
# 5. 실행
# ==========================================
def main():
    print("1. Data loading...")
    df = pd.read_csv(CSV_FILE).fillna({"input_text": "No review"})
    le = LabelEncoder()
    df["category_id"] = le.fit_transform(df["sub_category"].astype(str))
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    image_processor = AutoImageProcessor.from_pretrained("google/efficientnet-b0")
    train_loader = DataLoader(AmazonFashionFullDataset(train_df, tokenizer, image_processor), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(AmazonFashionFullDataset(val_df, tokenizer, image_processor), batch_size=BATCH_SIZE)
    model = UltimateFashionModel(num_cat=df["category_id"].nunique()).to(DEVICE)
    best_mae = float('inf')

    print("\nPHASE 1: Feature Extraction (Backbone Frozen)")
    model.freeze_backbones()
    optimizer_p1 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_PHASE1)
    for epoch in range(1, EPOCHS_PHASE1 + 1):
        train_loss = train_epoch(model, train_loader, optimizer_p1, None, DEVICE, ACCUMULATION_STEPS, epoch, "Phase 1")
        val_loss, val_mse, val_mae, avg_gate = evaluate(model, val_loader, DEVICE, epoch, "Phase 1")
        print(f"[Phase 1 - Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val MAE: {val_mae:.4f} | Val MSE: {val_mse:.4f}")
        print(f"GMU Gate -> Text: {avg_gate[0]:.2f}, Image: {avg_gate[1]:.2f}, Tabular: {avg_gate[2]:.2f}\n")

    print("\nPHASE 2: Full Fine-tuning (All Layers Unfrozen)")
    model.unfreeze_backbones()
    optimizer_p2 = torch.optim.AdamW(model.parameters(), lr=LR_PHASE2)
    scheduler_p2 = CosineAnnealingLR(optimizer_p2, T_max=EPOCHS_PHASE2 * (len(train_loader) // ACCUMULATION_STEPS))
    for epoch in range(1, EPOCHS_PHASE2 + 1):
        train_loss = train_epoch(model, train_loader, optimizer_p2, scheduler_p2, DEVICE, ACCUMULATION_STEPS, epoch, "Phase 2")
        val_loss, val_mse, val_mae, avg_gate = evaluate(model, val_loader, DEVICE, epoch, "Phase 2")
        print(f"[Phase 2 - Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val MAE: {val_mae:.4f} | Val MSE: {val_mse:.4f}")
        print(f"GMU Gate -> Text: {avg_gate[0]:.2f}, Image: {avg_gate[1]:.2f}, Tabular: {avg_gate[2]:.2f}")
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), "best_ultimate_model.pth")
            print(f"New Best Model Saved (MAE: {best_mae:.4f})\n")
        else:
            print()

if __name__ == "__main__":
    main()
