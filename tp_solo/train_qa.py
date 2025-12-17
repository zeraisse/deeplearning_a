import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

# On importe l'architecture depuis model.py
from model import TransformerModel, MAX_LEN, VOCAB_SIZE, BATCH_SIZE, LEARNING_RATE, EPOCHS

# --- 1. PRÉPARATION DES DONNÉES ---
print("📥 Chargement du Tokenizer et des Données...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
ds_train = load_dataset("rajpurkar/squad", split='train') 
ds_val = load_dataset("rajpurkar/squad", split='validation')

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
            answer = item['answers']['text'][0]
        except IndexError:
            answer = "" # Gestion des cas rares sans réponse
        
        # Encodage Encodeur : Question + Contexte
        input_text = f"{question} [SEP] {context}"
        enc_tokens = self.tokenizer(input_text, max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
        
        # Encodage Décodeur : Réponse
        ans_tokens = self.tokenizer(answer, max_length=MAX_LEN + 1, padding="max_length", truncation=True, return_tensors="pt")
        
        src_ids = enc_tokens['input_ids'].squeeze(0)
        ans_ids = ans_tokens['input_ids'].squeeze(0)
        
        # Décalage pour le Teacher Forcing
        dec_input = ans_ids[:-1].clone() # Entrée : [START, mot1, mot2]
        label = ans_ids[1:].clone()      # Sortie attendue : [mot1, mot2, END]
        
        # On remplace les 0 (padding) par -100 pour que la Loss les ignore
        label[label == 0] = -100 
        
        return src_ids, dec_input, label

# --- 2. LE MODULE LIGHTNING (Le Cerveau) ---
class SquadLightningModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = TransformerModel() # Appelle ton architecture model.py
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.save_hyperparameters() # Sauvegarde la config

    def forward(self, src, tgt):
        return self.model(src, tgt)

    def training_step(self, batch, batch_idx):
        src, tgt_in, tgt_out = batch
        output = self(src, tgt_in)
        
        # Aplatir les dimensions pour la Loss
        # Output: (Batch * Seq_Len, Vocab_Size)
        output = output.reshape(-1, VOCAB_SIZE)
        # Target: (Batch * Seq_Len)
        tgt_out = tgt_out.reshape(-1)
        
        loss = self.criterion(output, tgt_out)
        
        # Logs pour voir la progression en direct
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
        # AdamW est souvent meilleur pour les Transformers que Adam classique
        return optim.AdamW(self.parameters(), lr=LEARNING_RATE)

# --- 3. LANCEMENT ---
if __name__ == '__main__':
    # Initialisation des Datasets
    train_dataset = SquadDataset(ds_train, tokenizer)
    val_dataset = SquadDataset(ds_val, tokenizer)

    # DataLoaders (Optimisés pour Windows: num_workers=0 évite les bugs, monte à 2 ou 4 si Linux)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0)

    # CALLBACKS
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="squad-transformer-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        verbose=True
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,        # Arrête si pas d'amélioration pendant 5 époques
        verbose=True,
        mode="min"
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # LE TRAINER (Configuration RTX 5070 Ti)
    print("Démarrage de l'entraînement PyTorch Lightning...")
    
    # Activation du Float32 Matmul Precision pour cartes NVIDIA récentes (série 30/40/50)
    torch.set_float32_matmul_precision('medium')

    trainer = pl.Trainer(
        max_epochs=EPOCHS ,                  # Le EarlyStopping coupera avant si besoin
        accelerator="gpu",
        devices=1,
        precision="16-mixed",           # INDISPENSABLE pour 16 layers (économise la VRAM)
        accumulate_grad_batches=4,      # Astuce : Simule un batch 4x plus grand
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        log_every_n_steps=50
    )

    # Création du modèle
    model_lightning = SquadLightningModule()

    # C'est parti !
    trainer.fit(model_lightning, train_loader, val_loader)

    print(f"Terminé ! Meilleur modèle : {checkpoint_callback.best_model_path}")