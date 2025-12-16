import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
import math
from tqdm import tqdm
from model import TransformerModel, MAX_LEN, VOCAB_SIZE, BATCH_SIZE, LEARNING_RATE, EPOCHS

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation du périphérique : {DEVICE}")


# --- 1. TOKENIZER & DATA ---
print("Chargement Tokenizer & Data...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
ds_train = load_dataset("rajpurkar/squad", split='train') 
ds_val = load_dataset("rajpurkar/squad", split='validation')

class SquadDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        context = item['context']
        question = item['question']
        answer = item['answers']['text'][0]
        
        # Input: Question + Context
        input_text = f"{question} [SEP] {context}"
        enc_tokens = self.tokenizer(input_text, max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
        
        # Target: Answer
        ans_tokens = self.tokenizer(answer, max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
        
        src_ids = enc_tokens['input_ids'].squeeze(0)
        tgt_ids = ans_tokens['input_ids'].squeeze(0)
        
        # On shift les labels (Teacher Forcing manuel)
        # Note: BERT tokenizer gère déjà CLS/SEP, on simplifie ici pour l'exemple
        dec_input = tgt_ids.clone() # Input du decoder
        label = tgt_ids.clone()     # Ce qu'on doit prédire
        
        # On remplace le padding par -100 pour que la Loss l'ignore (Standard PyTorch)
        label[label == 0] = -100
        
        return src_ids, dec_input, label

train_dataset = SquadDataset(ds_train)
val_dataset = SquadDataset(ds_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)



# --- 3. ENTRAÎNEMENT ---
model = TransformerModel().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=-100) # Ignore le padding

print("🔥 Démarrage de l'entraînement PyTorch...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    # Barre de progression
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for src, tgt_in, tgt_out in progress_bar:
        src, tgt_in, tgt_out = src.to(DEVICE), tgt_in.to(DEVICE), tgt_out.to(DEVICE)
        
        optimizer.zero_grad()
        
        # Le décodeur prend la réponse décalée
        output = model(src, tgt_in)
        
        # Reshape pour la Loss : [Batch * Seq, Vocab]
        output = output.reshape(-1, VOCAB_SIZE)
        tgt_out = tgt_out.reshape(-1)
        
        loss = criterion(output, tgt_out)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} terminée. Loss moyenne: {avg_loss:.4f}")
    
    # Sauvegarde
    torch.save(model.state_dict(), "squad_pytorch_model.pth")

print("Modèle sauvegardé : squad_pytorch_model.pth")