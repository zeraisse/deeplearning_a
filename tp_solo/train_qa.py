import os
import glob
import math
import collections
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import CSVLogger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# --- HYPERPARAMETERS ---
MAX_LEN = 384
VOCAB_SIZE = 30522
EMBED_DIM = 256
NUM_HEADS = 4
FF_DIM = 1024
NUM_LAYERS = 4
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 50
BEAM_SIZE = 3

# --- UTILS: F1 SCORE ---
def compute_f1(prediction, truth):
    pred_tokens = prediction.split()
    truth_tokens = truth.split()
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common_tokens.values())
    
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    
    return (2 * precision * recall) / (precision + recall)

# --- MODEL: TRANSFORMER + BEAM SEARCH ---
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
        src_emb = self.pos_encoder(self.embedding(src))
        tgt_emb = self.pos_encoder(self.embedding(tgt))
        
        tgt_seq_len = tgt.size(1)
        device = src.device
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).to(device)
        
        src_pad_mask = (src == 0)
        tgt_pad_mask = (tgt == 0)

        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask
        )
        return self.fc_out(out)

    def beam_search(self, src, tokenizer, beam_width=3, max_gen_len=50):
        device = src.device
        self.eval()
        
        src_emb = self.pos_encoder(self.embedding(src))
        memory = self.transformer.encoder(src_emb)
        
        # (score, sequence_tensor)
        beams = [(0.0, torch.tensor([[101]], device=device))] # 101 = [CLS]/[BOS]
        
        for _ in range(max_gen_len):
            new_beams = []
            for score, seq in beams:
                if seq[0, -1].item() == 102: # 102 = [SEP]/[EOS]
                    new_beams.append((score, seq))
                    continue
                
                tgt_emb = self.pos_encoder(self.embedding(seq))
                tgt_mask = self.transformer.generate_square_subsequent_mask(seq.size(1)).to(device)
                
                out = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                logits = self.fc_out(out[:, -1, :])
                log_probs = torch.log_softmax(logits, dim=-1)
                
                top_scores, top_indices = torch.topk(log_probs, beam_width)
                
                for i in range(beam_width):
                    new_score = score + top_scores[0, i].item()
                    new_seq = torch.cat([seq, top_indices[0, i].unsqueeze(0).unsqueeze(0)], dim=1)
                    new_beams.append((new_score, new_seq))
            
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]
            
            if all(b[1][0, -1].item() == 102 for b in beams):
                break
                
        return beams[0][1]

# --- LIGHTNING MODULE ---
class SquadLightningModule(pl.LightningModule):
    def __init__(self, tokenizer):
        super().__init__()
        self.model = TransformerModel()
        self.tokenizer = tokenizer
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.save_hyperparameters(ignore=['tokenizer'])

    def forward(self, src, tgt):
        return self.model(src, tgt)

    def training_step(self, batch, batch_idx):
        src, tgt_in, tgt_out = batch
        logits = self(src, tgt_in)
        loss = self.criterion(logits.reshape(-1, VOCAB_SIZE), tgt_out.reshape(-1))
        
        # Token Accuracy (Teacher Forcing)
        preds = torch.argmax(logits, dim=-1)
        mask = tgt_out != -100
        correct = (preds == tgt_out) & mask
        accuracy = correct.sum().float() / mask.sum().float()
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt_in, tgt_out = batch
        logits = self(src, tgt_in)
        loss = self.criterion(logits.reshape(-1, VOCAB_SIZE), tgt_out.reshape(-1))
        
        # 1. Validation Token Accuracy
        preds_tf = torch.argmax(logits, dim=-1)
        mask = tgt_out != -100
        correct = (preds_tf == tgt_out) & mask
        val_acc = correct.sum().float() / mask.sum().float()
        
        # 2. Generative F1 Score (Greedy for speed monitoring, Beam Search is too slow for full val)
        # On utilise une génération simple pour monitorer la F1
        total_f1 = 0
        if batch_idx < 5: # Limit generation to first 5 batches to save time
            for i in range(min(4, src.size(0))): # Check first 4 samples
                # Greedy generation manually for speed
                with torch.no_grad():
                    gen_seq = self.model.beam_search(src[i].unsqueeze(0), self.tokenizer, beam_width=1, max_gen_len=20)
                
                pred_text = self.tokenizer.decode(gen_seq.squeeze(), skip_special_tokens=True)
                
                # Reconstruct Truth Text
                truth_ids = tgt_out[i].cpu().numpy()
                truth_text = self.tokenizer.decode(truth_ids[truth_ids != -100], skip_special_tokens=True)
                
                total_f1 += compute_f1(pred_text, truth_text)
            
            avg_f1 = total_f1 / min(4, src.size(0))
            self.log("val_f1", avg_f1, prog_bar=True)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=LEARNING_RATE)

# --- CALLBACKS ---
class PlottingCallback(Callback):
    def __init__(self):
        self.metrics = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}
        self.epochs = []

    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        
        # --- FIX: Sécurité anti-crash ---
        # Pendant le "Sanity Check", train_loss n'existe pas encore.
        if "train_loss" not in m:
            return 
        # --------------------------------

        self.epochs.append(trainer.current_epoch)
        
        # On récupère les valeurs proprement
        self.metrics["train_loss"].append(m["train_loss"].item())
        self.metrics["val_loss"].append(m.get("val_loss", torch.tensor(0.0)).item())
        self.metrics["val_acc"].append(m.get("val_acc", torch.tensor(0.0)).item())
        self.metrics["val_f1"].append(m.get("val_f1", torch.tensor(0.0)).item())

        # Création du graphique
        try:
            fig, ax = plt.subplots(1, 3, figsize=(18, 5))
            
            # Loss
            ax[0].plot(self.epochs, self.metrics["train_loss"], label="Train", marker='.')
            ax[0].plot(self.epochs, self.metrics["val_loss"], label="Val", marker='.')
            ax[0].set_title("Loss")
            ax[0].legend()
            ax[0].grid(True)
            
            # Accuracy
            ax[1].plot(self.epochs, self.metrics["val_acc"], color="green", marker='.')
            ax[1].set_title("Token Accuracy")
            ax[1].grid(True)
            
            # F1 Score
            ax[2].plot(self.epochs, self.metrics["val_f1"], color="orange", marker='.')
            ax[2].set_title("Generative F1 Score")
            ax[2].grid(True)
            
            plt.tight_layout()
            plt.savefig("monitoring_metrics.png")
            plt.close()
        except Exception as e:
            print(f"Erreur lors du plot: {e}")
            
# --- DATASET ---
class SquadDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        try: answer = item['answers']['text'][0]
        except IndexError: answer = ""
        
        # Format: Question [SEP] Context
        input_text = f"{item['question']} [SEP] {item['context']}"
        
        enc = self.tokenizer(input_text, max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
        # Target: [CLS] Answer [SEP]
        ans = self.tokenizer(answer, max_length=50, padding="max_length", truncation=True, return_tensors="pt")
        
        src_ids = enc['input_ids'].squeeze(0)
        
        # Shift targets for teacher forcing
        # tgt_in: [CLS] Answer ...
        # tgt_out: Answer ... [SEP]
        full_tgt = ans['input_ids'].squeeze(0)
        tgt_in = full_tgt[:-1]
        tgt_out = full_tgt[1:].clone()
        tgt_out[tgt_out == 0] = -100 # Ignore padding in loss
        
        return src_ids, tgt_in, tgt_out

# --- MAIN ---
if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    data = load_dataset("rajpurkar/squad", split="train[:5%]+validation[:5%]") # Small subset for demo
    split = data.train_test_split(test_size=0.1, seed=42)
    
    train_dl = DataLoader(SquadDataset(split['train'], tokenizer), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_dl = DataLoader(SquadDataset(split['test'], tokenizer), batch_size=BATCH_SIZE, num_workers=2)
    
    model_module = SquadLightningModule(tokenizer)
    
    logger = CSVLogger("logs", name="squad_experiment")
    
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        devices=1,
        logger=logger,
        log_every_n_steps=10,
        callbacks=[
            ModelCheckpoint(filename="best_model", monitor="val_loss"),
            LearningRateMonitor(logging_interval='step'),
            PlottingCallback()
        ]
    )
    
    print("Training Started...")
    trainer.fit(model_module, train_dl, val_dl)