import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from torch.utils.data import TensorDataset, DataLoader
from gridEnv import gridEnv, get_expert_action

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. ARCHITECTURE TRM (Identique) ---
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
    def __init__(self, input_dim=3, d_model=64, num_heads=2, num_actions=3, seq_len=49, n_steps=3):
        super().__init__()
        self.n_steps = n_steps
        
        self.x_emb = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model))
        self.y0 = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.z0 = nn.Parameter(torch.zeros(1, seq_len, d_model))

        self.block = TRMBlock(d_model, num_heads)
        self.head = nn.Linear(d_model, num_actions)

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

# --- 2. DATASET ---
def generate_dataset(episodes=500):
    print("Generation dataset via Expert...")
    env = gridEnv(size=6, render_mode="rgb_array")
    X_data, y_data = [], []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = get_expert_action(env)
            img = obs['image'].astype(np.float32) / 255.0
            X_data.append(img)
            y_data.append(action)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    X_tensor = torch.tensor(np.array(X_data))
    y_tensor = torch.tensor(np.array(y_data), dtype=torch.long)
    return TensorDataset(X_tensor, y_tensor)

# --- 3. FONCTION DE PLOT (NOUVEAU) ---
def plot_metrics(history):
    epochs = range(1, len(history['loss']) + 1)
    
    plt.figure(figsize=(15, 5))
    
    # Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['loss'], 'r-o', label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['acc'], 'b-o', label='Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)

    # F1 Score
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['f1'], 'g-o', label='F1 Score')
    plt.title('Training F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    print("📊 Graphiques sauvegardés sous 'training_metrics.png'")
    plt.close()

# --- 4. ENTRAINEMENT AVEC METRIQUES ---
def train_model(dataset, epochs=5):
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = TRMAgent(n_steps=3).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()

    print(f"Start TRM Training on {DEVICE}...")
    
    # Dictionnaire pour stocker l'historique
    history = {'loss': [], 'acc': [], 'f1': []}
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Stockage pour calculs metrics
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
        # Calcul des métriques de l'époque
        epoch_loss = total_loss / len(loader)
        epoch_acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
        # F1 score 'weighted' gère bien le déséquilibre des classes
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted') 
        
        history['loss'].append(epoch_loss)
        history['acc'].append(epoch_acc)
        history['f1'].append(epoch_f1)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}% | F1: {epoch_f1:.4f}")
        
    # Génération du graphique à la fin
    plot_metrics(history)
        
    return model