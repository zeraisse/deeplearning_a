import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import math

# --- CONFIGURATION FROM SCRATCH ---
# On reste raisonnable car on part de zéro
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 384       # Taille standard pour le QA
VOCAB_SIZE = 30522  # Taille du vocabulaire BERT
EMBED_DIM = 256     # Dimension des vecteurs
NUM_HEADS = 8
FF_DIM = 1024
NUM_LAYERS = 6      # 6 couches d'Encodeur
BATCH_SIZE = 32     # Ajuste selon ta VRAM (16 ou 32 sur 5070 Ti)
EPOCHS = 20         # Il faudra du temps pour apprendre
LR = 3e-4           # Learning rate un peu plus élevé pour le scratch

print(f"🔥 Mode: FROM SCRATCH (Extractive) sur {DEVICE}")

# --- 1. TOKENIZER & DATA ---
# On utilise le tokenizer juste pour couper les mots, mais AUCUN POIDS n'est chargé
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
        answer_text = item['answers']['text'][0]
        answer_start = item['answers']['answer_start'][0]

        # On tokenise Question + Contexte
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
        # Le masque sert à ignorer le padding
        mask = inputs["attention_mask"].squeeze(0) 
        
        # --- CALCUL DES CIBLES (START / END) ---
        offsets = inputs["offset_mapping"].squeeze(0)
        answer_end = answer_start + len(answer_text)
        
        start_token_idx = 0
        end_token_idx = 0
        
        # On cherche les tokens qui correspondent aux positions des caractères
        # C'est de la mathématique d'alignement
        found_start = False
        for i, (o_start, o_end) in enumerate(offsets):
            if o_start <= answer_start and o_end >= answer_start:
                start_token_idx = i
                found_start = True
            if o_start <= answer_end and o_end >= answer_end:
                end_token_idx = i
                break
        
        # Sécurité si la réponse a été tronquée
        if end_token_idx < start_token_idx:
            end_token_idx = start_token_idx

        return input_ids, mask, torch.tensor(start_token_idx), torch.tensor(end_token_idx)

train_dataset = SquadExtractiveDataset(ds_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 2. MODÈLE EXTRACTIF (Custom Architecture) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ExtractiveTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialisation aléatoire des Embeddings (From Scratch !)
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos_encoder = PositionalEncoding(EMBED_DIM, MAX_LEN)
        
        # Uniquement l'Encodeur (Pas de décodeur = plus facile à apprendre)
        encoder_layer = nn.TransformerEncoderLayer(d_model=EMBED_DIM, nhead=NUM_HEADS, dim_feedforward=FF_DIM, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        
        # Tête de sortie : On veut 2 chiffres pour chaque mot (Probabilité Start, Probabilité End)
        self.qa_outputs = nn.Linear(EMBED_DIM, 2)

    def forward(self, input_ids, attention_mask=None):
        # 1. Embedding
        x = self.embedding(input_ids)
        x = self.pos_encoder(x)
        
        # 2. Transformer (Encodeur)
        # On inverse le masque d'attention pour PyTorch (True = ignorer)
        src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # 3. Prédiction (Batch, Seq_Len, 2)
        logits = self.qa_outputs(x)
        
        # On sépare Start et End logits
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        return start_logits, end_logits

# --- 3. ENTRAÎNEMENT ---
model = ExtractiveTransformer().to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

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
        
        # On calcule l'erreur sur le début ET la fin
        loss_start = criterion(start_logits, start_idx)
        loss_end = criterion(end_logits, end_idx)
        loss = (loss_start + loss_end) / 2
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"✅ Epoch {epoch+1} terminée. Loss Moyenne: {total_loss / len(train_loader):.4f}")
    
    # Sauvegarde
    torch.save(model.state_dict(), "squad_scratch_extractive.pth")
    print("💾 Modèle sauvegardé.")

print("Terminé.")