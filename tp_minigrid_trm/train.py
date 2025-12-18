import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from gridEnv import gridEnv, get_expert_action

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TRMBlock(nn.Module):
    # Bloc partagé qui contient l'Attention et le MLP
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
        # Self-Attention + Residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP + Residual
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        return x

class TRMAgent(nn.Module):
    def __init__(self, input_dim=3, d_model=64, num_heads=2, num_actions=3, seq_len=49, n_steps=3):
        super().__init__()
        self.n_steps = n_steps
        self.seq_len = seq_len
        self.d_model = d_model

        # Embeddings
        self.x_emb = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model))

        # Etats initiaux apprenables (y0 et z0)
        self.y0 = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.z0 = nn.Parameter(torch.zeros(1, seq_len, d_model))

        # Le Cerveau partagé (block1 et block2 du schéma/code Keras)
        self.block = TRMBlock(d_model, num_heads)
        
        # Tête de sortie (Reverse Embedding / Prediction)
        self.head = nn.Linear(d_model, num_actions)

    def forward(self, x_pixels):
        b, h, w, c = x_pixels.shape
        # 1. Préparation de x (Input)
        x = x_pixels.view(b, -1, c) # (Batch, 49, 3)
        x = self.x_emb(x) + self.pos_emb

        # 2. Initialisation de y et z
        y = self.y0.expand(b, -1, -1)
        z = self.z0.expand(b, -1, -1)

        # 3. Boucle Récurrente (Thinking Process)
        for _ in range(self.n_steps):
            
            # Mise à jour de z : Input = x + y + z
            u_z = x + y + z
            z = self.block(u_z) # block joue le rôle de tiny_net

            # Mise à jour de y : Input = y + z
            u_y = y + z
            y = self.block(u_y)

        # 4. Prédiction finale
        # On moyenne les vecteurs y de la séquence pour avoir une décision unique
        y_summary = y.mean(dim=1)
        return self.head(y_summary)

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

def train_model(dataset, epochs=5):
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    # Instanciation du TRM avec 3 pas de réflexion
    model = TRMAgent(n_steps=3).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()

    print(f"Start TRM Training on {DEVICE}...")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f} | Acc: {100 * correct / total:.2f}%")
        
    return model