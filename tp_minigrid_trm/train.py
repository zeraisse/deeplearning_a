import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from gridEnv import gridEnv, get_expert_action

from config import (
    DEVICE, INPUT_DIM, D_MODEL, NUM_HEADS, NUM_ACTIONS, 
    SEQ_LEN, N_STEPS, BATCH_SIZE, LEARNING_RATE, EPOCHS, EPISODES,
    GRID_SIZE, CHECKPOINT_FILE, BEST_MODEL_FILE, RESUME_TRAINING, MAX_STEPS,
    DATASET_FILE, FORCE_NEW_DATASET
)

class TRMBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        return x

class TRMAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_steps = N_STEPS
        
        self.x_emb = nn.Linear(INPUT_DIM, D_MODEL)
        self.pos_emb = nn.Parameter(torch.randn(1, SEQ_LEN, D_MODEL))
        
        # Vecteurs récurrents y et z
        self.y0 = nn.Parameter(torch.zeros(1, SEQ_LEN, D_MODEL))
        self.z0 = nn.Parameter(torch.zeros(1, SEQ_LEN, D_MODEL))

        self.block = TRMBlock(D_MODEL, NUM_HEADS)
        self.head = nn.Linear(D_MODEL, NUM_ACTIONS)

    def forward(self, x_pixels):
        b, h, w, c = x_pixels.shape
        x = x_pixels.view(b, -1, c)
        x = self.x_emb(x) + self.pos_emb

        y = self.y0.expand(b, -1, -1)
        z = self.z0.expand(b, -1, -1)

        for _ in range(self.n_steps):
            u_z = x + y + z
            z = self.block(u_z)
            u_y = y + z
            y = self.block(u_y)

        y_summary = y.mean(dim=1)
        return self.head(y_summary)


def generate_dataset(episodes=EPISODES):
    # 1. EST-CE QUE LE FICHIER EXISTE DEJA ?
    if os.path.exists(DATASET_FILE) and not FORCE_NEW_DATASET:
        print(f"Dataset trouvé : '{DATASET_FILE}'. Chargement immédiat...")
        # On charge le fichier (weights_only=False pour éviter l'erreur de sécurité)
        try:
            return torch.load(DATASET_FILE, weights_only=False)
        except Exception as e:
            print(f"Erreur chargement ({e}). On régénère tout.")

    # 2. SINON, ON LE GENERE
    print(f"Génération du dataset ({episodes} épisodes)... Cela peut prendre du temps.")
    env = gridEnv(size=GRID_SIZE, render_mode="rgb_array")
    X_data, y_data = [], []

    for _ in tqdm(range(episodes), desc="Simulation"):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < MAX_STEPS:
            action = get_expert_action(env)
            
            # --- HUD LOGIC (Pixel Blanc) ---
            img = obs['image'].astype(np.float32) / 255.0
            if env.unwrapped.carrying:
                img[0, 0, :] = 1.0 
            # -------------------------------

            X_data.append(img)
            y_data.append(action)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

    # Conversion Tensor
    X_tensor = torch.tensor(np.array(X_data))
    y_tensor = torch.tensor(np.array(y_data), dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    
    print(f"Génération terminée ({len(X_data)} images).")
    
    # 3. ON SAUVEGARDE POUR LA PROCHAINE FOIS !
    print(f"Sauvegarde dans '{DATASET_FILE}'...")
    torch.save(dataset, DATASET_FILE)
    
    return dataset

def plot_metrics(history):
    epochs_range = range(1, len(history['loss']) + 1)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, history['loss'], 'r-o', label='Loss')
    plt.title('Loss')
    plt.grid(True)
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, history['acc'], 'b-o', label='Accuracy')
    plt.title('Accuracy (%)')
    plt.grid(True)
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, history['f1'], 'g-o', label='F1 Score')
    plt.title('F1 Score')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

def train_model(dataset, epochs=EPOCHS):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = TRMAgent().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print(f"Start TRM Training on {DEVICE} | Actions={NUM_ACTIONS}")
    
    # Variables d'état
    start_epoch = 0
    best_f1 = 0.0
    history = {'loss': [], 'acc': [], 'f1': []}

    # --- 1. CHARGEMENT DU BACKUP (Si demandé et si existe) ---
    if RESUME_TRAINING and os.path.exists(CHECKPOINT_FILE):
        print(f"Checkpoint trouvé : '{CHECKPOINT_FILE}'. Reprise de l'entraînement...")
        checkpoint = torch.load(CHECKPOINT_FILE, weights_only=False)  

        # On recharge tout
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1 # On reprend à l'époque suivante
        best_f1 = checkpoint['best_f1']
        history = checkpoint['history']
        
        print(f"   -> Reprise à l'époque {start_epoch + 1}/{epochs}. Best F1 actuel: {best_f1:.4f}")
    else:
        print("Nouvel entraînement démarré.")

    model.train()
    
    # --- 2. BOUCLE D'ENTRAINEMENT ---
    for epoch in range(start_epoch, epochs):
        total_loss = 0
        all_preds, all_labels = [], []
        
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            # --- GRADIENT CLIPPING ---
            # coupe tout mouvement trop violent (> 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # ------
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
        epoch_loss = total_loss / len(loader)
        epoch_acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        history['loss'].append(epoch_loss)
        history['acc'].append(epoch_acc)
        history['f1'].append(epoch_f1)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}% | F1: {epoch_f1:.4f}")
        
        # --- 3. SAUVEGARDE DU CHECKPOINT (Systématique) ---
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_f1': best_f1,
            'history': history
        }, CHECKPOINT_FILE)
        
        # --- 4. SAUVEGARDE DU MEILLEUR MODELE (Conditionnelle) ---
        if epoch_f1 > best_f1:
            best_f1 = epoch_f1
            torch.save(model.state_dict(), BEST_MODEL_FILE)
            print(f"   ★ Nouveau Record F1 ! Modèle sauvegardé dans '{BEST_MODEL_FILE}'")
        
    plot_metrics(history)
    
    # A la fin, on charge les poids du MEILLEUR modèle (pas forcément le dernier) pour la vidéo
    if os.path.exists(BEST_MODEL_FILE):
        print("Chargement du meilleur modèle pour l'évaluation...")
        model.load_state_dict(torch.load(BEST_MODEL_FILE, weights_only=False))
        
    return model