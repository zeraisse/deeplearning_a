import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from MiniGridEnv import gridEnv, get_expert_action # <-- Import mis à jour

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerAgent(nn.Module):
    def __init__(self, input_dim=3, d_model=64, num_heads=2, num_layers=2, num_actions=3, seq_len=49):
        super().__init__()
        # [Input x] : Projection linéaire des pixels
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        # [Latent z] : Raisonnement via Self-Attention
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # [Prediction y] : Décision finale
        self.head = nn.Linear(d_model, num_actions)

    def forward(self, x):
        b, h, w, c = x.shape
        x = x.view(b, -1, c) # Aplatissement (Batch, 49, 3)
        
        z = self.embedding(x) + self.pos_emb 
        z = self.transformer(z) # Le modèle réfléchit
        z_summary = z.mean(dim=1) # Synthèse globale
        
        return self.head(z_summary) # Prédiction

def generate_dataset(episodes=500):
    print("🤖 Génération des données via l'Expert...")
    env = gridEnv(size=6, render_mode="rgb_array") # <-- Utilisation de gridEnv
    X_data, y_data = [], []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = get_expert_action(env)
            
            # Normalisation (0-1)
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
    model = TransformerAgent().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print(f"🔥 Démarrage de l'entraînement sur {DEVICE}...")
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