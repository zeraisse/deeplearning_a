import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import minigrid # Important pour enregistrer les envs
from sklearn.metrics import f1_score
from torch.utils.data import TensorDataset, DataLoader

# Fix pour serveur sans écran
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from gridEnv import get_expert_action

# --- HYPERPARAMETERS ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENV_ID = "MiniGrid-Empty-8x8-v0"
INPUT_DIM = 3       # RGB
D_MODEL = 64        # Latent dim
NUM_HEADS = 2       # Attention heads
NUM_ACTIONS = 3     # Gauche, Droite, Avancer (MiniGrid a 7 actions, mais on en utilise 3)
SEQ_LEN = 49        # Vue 7x7 = 49 pixels (Standard Minigrid)
N_STEPS = 3         # TRM Thinking steps

BATCH_SIZE = 64     # Augmenté pour stabilité
LEARNING_RATE = 0.0005
EPOCHS = 25         # Suffisant avec un bon expert
EPISODES = 1000     # Plus d'épisodes car 16x16 est plus grand

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
    print(f"Generation dataset on {ENV_ID} ({episodes} eps)...")
    # render_mode='rgb_array' est obligatoire pour avoir l'image dans l'observation
    env = gym.make(ENV_ID, render_mode="rgb_array")
    
    X_data, y_data = [], []

    for i in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            # L'expert calcule l'action optimale
            action = get_expert_action(env)
            
            # Gymnasium Minigrid renvoie un dict. On veut 'image'.
            # Normalisation 0-255 -> 0-1
            img = obs['image'].astype(np.float32) / 255.0
            
            X_data.append(img)
            y_data.append(action)
            
            # Step
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    env.close()
    
    # Conversion Tensor
    X_tensor = torch.tensor(np.array(X_data))
    y_tensor = torch.tensor(np.array(y_data), dtype=torch.long)
    print(f"Dataset size: {len(X_data)} frames")
    return TensorDataset(X_tensor, y_tensor)

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
    print("Graphiques sauvegardés sous 'training_metrics.png'")
    plt.close()

def train_model(dataset, epochs=EPOCHS):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = TRMAgent().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print(f"Start TRM Training | Epochs={epochs}")
    history = {'loss': [], 'acc': [], 'f1': []}
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        all_preds, all_labels = [], []
        
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
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
        
        print(f"Ep {epoch+1} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.1f}% | F1: {epoch_f1:.3f}")
        
    plot_metrics(history)
    return model