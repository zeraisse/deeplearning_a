import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import math
import os
from model_scratch import ExtractiveTransformer, MAX_LEN

# todo beam search f1 value
# integrer le monitoring ?

# --- CONFIGURATION FROM SCRATCH ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 384       
VOCAB_SIZE = 30522  
EMBED_DIM = 256     
NUM_HEADS = 8
FF_DIM = 1024
NUM_LAYERS = 6      
BATCH_SIZE = 32     
EPOCHS = 50         
LR = 3e-4           

print(f"🔥 Mode: FROM SCRATCH (Extractive) sur {DEVICE}")

# --- 1. TOKENIZER & DATA ---
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
ds_train = load_dataset("rajpurkar/squad", split='train')

class SquadExtractiveDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        context = item['context']
        question = item['question']
        try:
            answer_text = item['answers']['text'][0]
            answer_start = item['answers']['answer_start'][0]
        except IndexError:
            answer_text = ""
            answer_start = 0

        inputs = self.tokenizer(
            question,
            context,
            max_length=MAX_LEN,
            padding="max_length",
            truncation="only_second",
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        
        input_ids = inputs["input_ids"].squeeze(0)
        mask = inputs["attention_mask"].squeeze(0) 
        offsets = inputs["offset_mapping"].squeeze(0)
        
        answer_end = answer_start + len(answer_text)
        start_token_idx = 0
        end_token_idx = 0
        
        found_start = False
        for i, (o_start, o_end) in enumerate(offsets):
            if o_start <= answer_start and o_end >= answer_start:
                start_token_idx = i
                found_start = True
            if o_start <= answer_end and o_end >= answer_end:
                end_token_idx = i
                break
        
        if end_token_idx < start_token_idx:
            end_token_idx = start_token_idx

        return input_ids, mask, torch.tensor(start_token_idx), torch.tensor(end_token_idx)

train_dataset = SquadExtractiveDataset(ds_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 2. MODÈLE & OPTIM ---
model = ExtractiveTransformer().to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# --- 3. ENTRAÎNEMENT ---
best_loss = float('inf')

print("🚀 Démarrage Entraînement FROM SCRATCH (Extractive)...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for input_ids, mask, start_idx, end_idx in loop:
        input_ids, mask = input_ids.to(DEVICE), mask.to(DEVICE)
        start_idx, end_idx = start_idx.to(DEVICE), end_idx.to(DEVICE)
        
        optimizer.zero_grad()
        
        start_logits, end_logits = model(input_ids, mask)
        
        loss_start = criterion(start_logits, start_idx)
        loss_end = criterion(end_logits, end_idx)
        loss = (loss_start + loss_end) / 2
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    avg_loss = total_loss / len(train_loader)
    
    print(f"Epoch {epoch+1} terminée. Loss Moyenne: {avg_loss:.4f}")

    # SAUVEGARDE
    torch.save(model.state_dict(), "squad_scratch_extractive_last.pth")
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "squad_scratch_extractive_best.pth")
        print(f"🌟 Nouveau record ! Modèle sauvegardé (Loss: {best_loss:.4f})")
    else:
        print("💾 Sauvegarde étape effectuée.")

print("🏆 Terminé.")