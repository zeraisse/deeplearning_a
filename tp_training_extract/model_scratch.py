import torch
import torch.nn as nn
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Data & Tokenizer
MAX_LEN = 384           # Taille max d'une séquence
VOCAB_SIZE = 30522      # Taille vocabulaire BERT
# Architecture Modèle
EMBED_DIM = 256         # Largeur du modèle
NUM_HEADS = 8           # Têtes d'attention
FF_DIM = 1024           # Largeur interne FeedForward
NUM_LAYERS = 8          # Profondeur (Augmenté à 8 pour plus d'intelligence)
DROPOUT = 0.3           # Freinage fort pour éviter le "par cœur" (Overfitting)
# Entraînement
BATCH_SIZE = 32         # Si Erreur "OOM" (Out of Memory), passe à 16
EPOCHS = 50             # Longue durée
LR = 3e-4               # Learning Rate
WEIGHT_DECAY = 0.01     # Régularisation supplémentaire

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
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos_encoder = PositionalEncoding(EMBED_DIM, MAX_LEN)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM, 
            nhead=NUM_HEADS, 
            dim_feedforward=FF_DIM, 
            dropout=0.3, # <--- AUGMENTE ÇA (C'est le frein)
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        
        self.qa_outputs = nn.Linear(EMBED_DIM, 2)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.pos_encoder(x)
        
        src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        logits = self.qa_outputs(x)
        start_logits, end_logits = logits.split(1, dim=-1)
        
        return start_logits.squeeze(-1), end_logits.squeeze(-1)