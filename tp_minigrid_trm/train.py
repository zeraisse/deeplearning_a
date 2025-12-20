import os
import minigrid
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
from minigrid.wrappers import FullyObsWrapper

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
    env = FullyObsWrapper(gridEnv(size=GRID_SIZE, render_mode="rgb_array"))
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
    print("🚀 Optimisation RTX 5070 Ti : Chargement VRAM (Version Stable 10x10)")
    
    # --- 1. FORCE VRAM ---
    # Avec GRID_SIZE=10, ça passe large !
    inputs, targets = dataset.tensors
    inputs = inputs.to(DEVICE)
    targets = targets.to(DEVICE)
    
    gpu_dataset = TensorDataset(inputs, targets)
    # num_workers=0 obligatoire une fois sur GPU
    loader = DataLoader(gpu_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    print("✅ Données chargées sur le GPU.")

    model = TRMAgent().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    weights = torch.tensor([1.0, 1.0, 0.3, 10.0, 1.0, 10.0, 1.0]).to(DEVICE)
    print("⚖️ Activation des poids correctifs : Punition x10 sur les oublis de clé/porte !")
    criterion = nn.CrossEntropyLoss(weight=weights)

    print(f"Start TRM Training on {DEVICE}")
    
    best_loss = float('inf') 
    history = {'loss': []}

    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        
        for imgs, labels in loop:
            # Pas de .to(DEVICE), c'est déjà dessus !
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        epoch_loss = total_loss / len(loader)
        history['loss'].append(epoch_loss)
        
        # Un print simple pour voir que ça avance
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f}")
        
        # Sauvegarde
        torch.save({'epoch': epoch, 'model': model.state_dict(), 'history': history}, CHECKPOINT_FILE)
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), BEST_MODEL_FILE)
            print(f"   ★ Record Loss ! ({best_loss:.4f})")
        
    return model