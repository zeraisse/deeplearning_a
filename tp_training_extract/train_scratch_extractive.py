import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from model_scratch import (
    ExtractiveTransformer, 
    DEVICE, MAX_LEN, BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY
)

os.makedirs("checkpoints", exist_ok=True)

print(f"Mode:  {DEVICE}")

# --- 1. FONCTIONS INTELLIGENTES (BEAM SEARCH) ---

def get_best_span(start_logits, end_logits, n_best=5, max_len=30):
    """
    C'est ici que se cache le Beam Search simplifié.
    Au lieu de prendre argmax(start) et argmax(end), on cherche le meilleur couple.
    """
    # On prend les N meilleurs indices de début et de fin
    start_probs, start_indices = torch.topk(start_logits, n_best)
    end_probs, end_indices = torch.topk(end_logits, n_best)
    
    start_indices = start_indices.tolist()
    end_indices = end_indices.tolist()
    start_probs = start_probs.tolist()
    end_probs = end_probs.tolist()
    
    best_score = -float('inf')
    best_start = 0
    best_end = 0
    
    # On teste toutes les combinaisons valides
    for i in range(len(start_indices)):
        for j in range(len(end_indices)):
            start_idx = start_indices[i]
            end_idx = end_indices[j]
            score = start_probs[i] + end_probs[j]
            
            # Filtre 1 : Fin avant début -> Impossible
            if end_idx < start_idx:
                continue
            # Filtre 2 : Réponse trop longue -> Peu probable
            if end_idx - start_idx + 1 > max_len:
                continue
                
            if score > best_score:
                best_score = score
                best_start = start_idx
                best_end = end_idx
                
    return best_start, best_end

def calculate_f1(pred_start, pred_end, true_start, true_end):
    # Calcul précis de la F1 (chevauchement des mots)
    if pred_end < pred_start: return 0.0
    
    pred_tokens = set(range(pred_start, pred_end + 1))
    true_tokens = set(range(true_start, true_end + 1))
    
    if len(pred_tokens) == 0 or len(true_tokens) == 0: return 0.0
    common = len(pred_tokens.intersection(true_tokens))
    if common == 0: return 0.0
    
    prec = common / len(pred_tokens)
    rec = common / len(true_tokens)
    return 2 * (prec * rec) / (prec + rec)

# --- 2. DATASETS ---
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
data = load_dataset("rajpurkar/squad")
ds_train = data['train']
ds_val = data['validation']

class SquadDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        context = item['context']
        question = item['question']
        try:
            ans_text = item['answers']['text'][0]
            ans_start = item['answers']['answer_start'][0]
        except:
            ans_text = ""
            ans_start = 0
            
        inputs = self.tokenizer(question, context, max_length=MAX_LEN, padding="max_length", 
                                truncation="only_second", return_offsets_mapping=True, return_tensors="pt")
        
        input_ids = inputs["input_ids"].squeeze(0)
        mask = inputs["attention_mask"].squeeze(0)
        offsets = inputs["offset_mapping"].squeeze(0)
        
        ans_end = ans_start + len(ans_text)
        start_idx, end_idx = 0, 0
        
        if ans_text:
            for i, (o_start, o_end) in enumerate(offsets):
                if o_start <= ans_start and o_end >= ans_start: start_idx = i
                if o_start <= ans_end and o_end >= ans_end: 
                    end_idx = i
                    break
            if end_idx < start_idx: end_idx = start_idx
            
        return input_ids, mask, torch.tensor(start_idx), torch.tensor(end_idx)

# --- 3. MODEL & OPTIM ---
train_loader = DataLoader(SquadDataset(ds_train, tokenizer), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(SquadDataset(ds_val, tokenizer), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = ExtractiveTransformer().to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY) 
criterion = nn.CrossEntropyLoss()

# --- 4. BOUCLE DE TRAINING INTELLIGENTE ---
best_f1 = 0.0  # On ne tracke plus la loss, mais la F1 !
history_loss = []
history_f1 = []

print("Lancement (Monitoring basé sur F1 + Beam Search)...")

for epoch in range(EPOCHS):
    model.train()
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    total_loss = 0
    
    for input_ids, mask, start_idx, end_idx in loop:
        input_ids, mask = input_ids.to(DEVICE), mask.to(DEVICE)
        start_idx, end_idx = start_idx.to(DEVICE), end_idx.to(DEVICE)
        
        optimizer.zero_grad()
        s_logits, e_logits = model(input_ids, mask)
        loss = (criterion(s_logits, start_idx) + criterion(e_logits, end_idx)) / 2
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    # --- VALIDATION AVEC BEAM SEARCH ---
    model.eval()
    val_f1_accum = 0
    
    # On utilise tqdm aussi pour la validation car le Beam Search prend un peu de temps
    with torch.no_grad():
        for input_ids, mask, start_idx, end_idx in tqdm(val_loader, desc="Validation (Beam Search)"):
            input_ids, mask = input_ids.to(DEVICE), mask.to(DEVICE)
            
            s_logits, e_logits = model(input_ids, mask)
            
            # Pour chaque exemple du batch, on applique le Beam Search
            for i in range(input_ids.size(0)):
                # Récupérer les logits d'un seul exemple
                sl = s_logits[i]
                el = e_logits[i]
                
                # --- BEAM SEARCH ICI ---
                pred_s, pred_e = get_best_span(sl, el, n_best=3) # On regarde les 5 meilleures options
                
                # Calcul F1
                f1 = calculate_f1(pred_s, pred_e, start_idx[i].item(), end_idx[i].item())
                val_f1_accum += f1

    avg_f1 = val_f1_accum / len(ds_val)
    history_f1.append(avg_f1)
    
    print(f"📊 Epoch {epoch+1} | Moyenne F1 (Beam Search): {avg_f1:.4f}")

    # --- SAUVEGARDE BASÉE SUR L'INTELLIGENCE (F1) ---
    if avg_f1 > best_f1:
        best_f1 = avg_f1
        torch.save(model.state_dict(), "checkpoints/best_model_smart.pth")
        print(f"NOUVEAU RECORD D'INTELLIGENCE ! Modèle sauvegardé (F1: {best_f1:.4f})")
    
    torch.save(model.state_dict(), "checkpoints/last_model.pth")

    # Graphique
    plt.figure()
    plt.plot(history_f1, label='Val F1 Score')
    plt.title('Progression de l\'Intelligence du Modèle')
    plt.legend()
    plt.grid(True)
    plt.savefig('smart_monitoring.png')
    plt.close()

print("Terminé.")