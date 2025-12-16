import torch
import torch.nn as nn
import math

# --- HYPERPARAMÈTRES ---
MAX_LEN = 128
VOCAB_SIZE = 30522
EMBED_DIM = 256
NUM_HEADS = 4
FF_DIM = 1024
NUM_LAYERS = 2
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 10

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

class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos_encoder = PositionalEncoding(EMBED_DIM, MAX_LEN)
        
        self.transformer = nn.Transformer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            num_encoder_layers=NUM_LAYERS,
            num_decoder_layers=NUM_LAYERS,
            dim_feedforward=FF_DIM,
            batch_first=True,
            dropout=0.1
        )
        
        self.fc_out = nn.Linear(EMBED_DIM, VOCAB_SIZE)
        
    def forward(self, src, tgt):
        # On récupère le device automatiquement depuis les données d'entrée
        device = src.device
        
        src_emb = self.pos_encoder(self.embedding(src))
        tgt_emb = self.pos_encoder(self.embedding(tgt))
        
        tgt_seq_len = tgt.size(1)
        # On génère le masque sur le même device que les données
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).to(device)
        
        src_pad_mask = (src == 0)
        tgt_pad_mask = (tgt == 0)

        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask
        )
        
        return self.fc_out(out)