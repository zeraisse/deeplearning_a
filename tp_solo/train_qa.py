import os
import warnings
import glob
import torch
import torch.nn as nn
import torch.optim as optim
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

# --- 1. PRÉPARATION DES DONNÉES (MODIFIÉ POUR 90% / 10%) ---
full_dataset = load_dataset("rajpurkar/squad", split="train+validation")

# On coupe : 10% pour le test (validation), le reste (90%) pour l'entrainement
# seed=42 permet d'avoir toujours le même découpage si tu relances
split_data = full_dataset.train_test_split(test_size=0.1, seed=42)

ds_train = split_data['train']
ds_val = split_data['test']

print(f"📊 Données préparées : {len(ds_train)} entraînement (90%) / {len(ds_val)} validation (10%)")

class SquadDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        context = item['context']
        question = item['question']
        try:
            # On prend la première réponse disponible
            answer = item['answers']['text'][0]
        except IndexError:
            answer = ""
        
        input_text = f"{question} [SEP] {context}"
        enc_tokens = self.tokenizer(input_text, max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
        
        ans_tokens = self.tokenizer(answer, max_length=MAX_LEN + 1, padding="max_length", truncation=True, return_tensors="pt")
        
        src_ids = enc_tokens['input_ids'].squeeze(0)
        ans_ids = ans_tokens['input_ids'].squeeze(0)
        
        dec_input = ans_ids[:-1].clone()
        label = ans_ids[1:].clone()
        
        label[label == 0] = -100
        
        return src_ids, dec_input, label

class PredictionLogger(Callback):
    def __init__(self, tokenizer, num_samples=2, log_file="training_logs.txt"):
        super().__init__()
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.log_file = log_file
        
        # On vide le fichier au démarrage
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("=== LOGS D'ENTRAÎNEMENT ===\n")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx != 0: return

        src, tgt_in, tgt_out = batch
        
        with torch.no_grad():
            logits = pl_module(src, tgt_in)
            preds = torch.argmax(logits, dim=-1)

        # On ouvre le fichier en mode "append" (ajout)
        with open(self.log_file, "a", encoding="utf-8") as f:
            header = f"\n--- EPOCH {trainer.current_epoch} ---\n"
            print(header) # Affichage Terminal
            f.write(header)
            
            for i in range(min(self.num_samples, src.size(0))):
                src_text = self.tokenizer.decode(src[i], skip_special_tokens=True)[:100]
                
                truth_ids = tgt_out[i].cpu().numpy()
                truth_text = self.tokenizer.decode(truth_ids[truth_ids != -100], skip_special_tokens=True)
                
                pred_text = self.tokenizer.decode(preds[i], skip_special_tokens=True)

                # Formatage du message
                msg = (f"Q:    {src_text}...\n"
                       f"Ref:  {truth_text}\n"
                       f"Pred: {pred_text}\n"
                       f"{'-'*20}\n")
                
                print(msg) # Affichage Terminal
                f.write(msg) # Ecriture Fichier

class SquadLightningModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = TransformerModel()
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.save_hyperparameters()

    def forward(self, src, tgt):
        return self.model(src, tgt)

    def training_step(self, batch, batch_idx):
        src, tgt_in, tgt_out = batch
        output = self(src, tgt_in)
        
        output = output.reshape(-1, VOCAB_SIZE)
        tgt_out = tgt_out.reshape(-1)
        
        loss = self.criterion(output, tgt_out)
        
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt_in, tgt_out = batch
        output = self(src, tgt_in)
        
        output = output.reshape(-1, VOCAB_SIZE)
        tgt_out = tgt_out.reshape(-1)
        
        loss = self.criterion(output, tgt_out)
        self.log("val_loss", loss, prog_bar=True, logger=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=LEARNING_RATE)

    def on_before_optimizer_step(self, optimizer):
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        self.log("grad_norm", total_norm, prog_bar=True)

if __name__ == '__main__':
    # Initialisation des datasets avec les variables définies plus haut
    train_dataset = SquadDataset(ds_train, tokenizer)
    val_dataset = SquadDataset(ds_val, tokenizer)
    
    logger = CSVLogger("logs_csv", name="squad_history")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="squad-transformer-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    pred_logger = PredictionLogger(tokenizer=tokenizer)

    torch.set_float32_matmul_precision('medium')

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        accumulate_grad_batches=4,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor, pred_logger], 
        log_every_n_steps=10,
        profiler="simple"
    )

    list_of_checkpoints = glob.glob('checkpoints/*.ckpt')
    if list_of_checkpoints:
        latest_checkpoint = max(list_of_checkpoints, key=os.path.getctime)
        model_lightning = SquadLightningModule.load_from_checkpoint(latest_checkpoint)
    else:
        model_lightning = SquadLightningModule()

    trainer.fit(model_lightning, train_loader, val_loader)