from utils import make_env
from model import DARQN
import torch

# 1. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = make_env("ALE/Pacman-v5")
obs_shape = env.observation_space.shape # (4, 84, 84)
n_actions = env.action_space.n

print(f"🏗️ Construction du DARQN sur {device}...")
model = DARQN(input_shape=obs_shape, num_actions=n_actions).to(device)

# 2. Simulation d'un Batch (Show-off: Batch size 32)
# PyTorch attend (Batch, Channels, Height, Width)
dummy_input = torch.randn(32, *obs_shape).to(device)

# 3. Initialisation Mémoire LSTM
hidden = model.init_hidden(batch_size=32, device=device)

# 4. Forward Pass
print("⚡ Test du Forward Pass...")
q_values, next_hidden, attn_map = model(dummy_input, hidden)

print(f"✅ Output Q-Values Shape: {q_values.shape} (Doit être [32, {n_actions}])")
print(f"✅ Attention Map Shape: {attn_map.shape} (Doit être [32, 64, 1])")
print("🚀 Modèle prêt pour l'entraînement !")