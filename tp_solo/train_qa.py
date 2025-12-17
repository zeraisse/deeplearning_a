import os
import warnings
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import CSVLogger
from model import TransformerModel, MAX_LEN, VOCAB_SIZE, BATCH_SIZE, LEARNING_RATE, EPOCHS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# --- 1. PRÉPARATION DES DONNÉES (90% / 10%) ---
print("📊 Chargement et découpage des données...")
full_dataset = load_dataset("rajpurkar/squad", split="train+validation")
split_data = full_dataset.train_test_split(test_size=0.1, seed=42)
ds_train = split_data['train']
ds_val = split_data['test']
print(f"✅ Données : {len(ds_train)} Train / {len(ds_val)} Val")

class SquadDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        try: answer = item['answers']['text'][0]
        except IndexError: answer = ""
        
        input_text = f"{item['question']} [SEP] {item['context']}"
        enc_tokens = self.tokenizer(input_text, max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
        ans_tokens = self.tokenizer(answer, max_length=MAX_LEN + 1, padding="max_length", truncation=True, return_tensors="pt")
        
        src_ids = enc_tokens['input_ids'].squeeze(0)
        ans_ids = ans_tokens['input_ids'].squeeze(0)
        
        label = ans_ids[1:].clone()
        label[label == 0] = -100 
        
        return src_ids, ans_ids[:-1], label

# --- 2. CALLBACK TEXTE (CELUI QUE TU AVAIS DÉJÀ) ---
class PredictionLogger(Callback):
    def __init__(self, tokenizer, num_samples=2, log_file="training_logs.txt"):
        super().__init__()
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.log_file = log_file
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("=== LOGS D'ENTRAÎNEMENT ===\n")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx != 0: return
        src, tgt_in, tgt_out = batch
        with torch.no_grad():
            preds = torch.argmax(pl_module(src, tgt_in), dim=-1)

        with open(self.log_file, "a", encoding="utf-8") as f:
            header = f"\n--- EPOCH {trainer.current_epoch} ---\n"
            f.write(header)
            # On affiche aussi dans le terminal pour que tu saches que ça tourne
            print(f"📝 Ecriture des logs dans {self.log_file}...") 
            
            for i in range(min(self.num_samples, src.size(0))):
                src_txt = self.tokenizer.decode(src[i], skip_special_tokens=True)[:100]
                truth_ids = tgt_out[i].cpu().numpy()
                truth_txt = self.tokenizer.decode(truth_ids[truth_ids != -100], skip_special_tokens=True)
                pred_txt = self.tokenizer.decode(preds[i], skip_special_tokens=True)
                
                msg = f"Q: {src_txt}...\nRef: {truth_txt}\nPred: {pred_txt}\n{'-'*20}\n"
                f.write(msg)

# --- 3. NOUVEAU CALLBACK GRAPHIQUE (POUR L'IMAGE PNG) ---
class PlottingCallback(Callback):
    def __init__(self):
        super().__init__()
        self.train_loss = []
        self.val_loss = []
        self.epochs = []

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        t_loss = metrics.get("train_loss")
        v_loss = metrics.get("val_loss")

        if t_loss is not None and v_loss is not None:
            self.train_loss.append(t_loss.item())
            self.val_loss.append(v_loss.item())
            self.epochs.append(trainer.current_epoch)
            
            plt.figure(figsize=(10, 6))
            plt.plot(self.epochs, self.train_loss, label="Train Loss", color='blue', marker='o')
            plt.plot(self.epochs, self.val_loss, label="Val Loss", color='red', linestyle='--', marker='x')
            plt.title(f"Training Progress - Epoch {trainer.current_epoch}")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig("courbe_apprentissage.png") # Sauvegarde l'image
            plt.close()

class SquadLightningModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = TransformerModel()
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.save_hyperparameters()

    def forward(self, src, tgt): return self.model(src, tgt)

    def training_step(self, batch, batch_idx):
        loss = self.criterion(self(batch[0], batch[1]).reshape(-1, VOCAB_SIZE), batch[2].reshape(-1))
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.criterion(self(batch[0], batch[1]).reshape(-1, VOCAB_SIZE), batch[2].reshape(-1))
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self): return optim.AdamW(self.parameters(), lr=LEARNING_RATE)

# --- 4. LANCEMENT ---
if __name__ == '__main__':
    train_dl = DataLoader(SquadDataset(ds_train, tokenizer), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl = DataLoader(SquadDataset(ds_val, tokenizer), batch_size=BATCH_SIZE, num_workers=0)
    
    logger = CSVLogger("logs_csv", name="squad_history")
    # ON AJOUTE NOTRE CALLBACK GRAPHIQUE ICI
    callbacks = [
        ModelCheckpoint(dirpath="checkpoints", filename="model-{epoch:02d}-{val_loss:.2f}", monitor="val_loss"),
        LearningRateMonitor(logging_interval='step'),
        PredictionLogger(tokenizer), # Ton logger texte
        PlottingCallback()           # Le logger graphique PNG
    ]

    torch.set_float32_matmul_precision('medium')

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        accumulate_grad_batches=4,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10
    )

    ckpt_path = None
    list_of_checkpoints = glob.glob('checkpoints/*.ckpt')
    if list_of_checkpoints:
        ckpt_path = max(list_of_checkpoints, key=os.path.getctime)
        print(f"♻️ Reprise depuis : {ckpt_path}")

    print("🚀 Entraînement lancé...")
    trainer.fit(SquadLightningModule(), train_dl, val_dl, ckpt_path=ckpt_path)