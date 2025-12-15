import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import glob
from utils import make_env
from agent import DARQNAgent

# --- CONFIGURATION ---
ENV_NAME = "ALE/Pong-v5"
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "runs/pong_darqn_sm120"
MAX_EPISODES = 800
SAVE_INTERVAL = 50     # Sauvegarde tous les 50 épisodes
TARGET_UPDATE = 100    # Mise à jour du target net
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 50000 # Nombre de frames pour la décroissance


BATCH_SIZE = 128       # (Au lieu de 32) On sature la mémoire du GPU
LR = 0.00025           # (Au lieu de 1e-4) Apprentissage 2.5x plus rapide (plus risqué mais ça passe)
GAMMA = 0.99
MEMORY_SIZE = 50000    # Mémoire plus grande
MIN_MEMORY = 2000

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Démarrage sur {torch.cuda.get_device_name(0)} (Blackwell Engine)")

env = make_env(ENV_NAME)
obs_shape = env.observation_space.shape
n_actions = env.action_space.n

agent = DARQNAgent(obs_shape, n_actions, device, batch_size=BATCH_SIZE)
writer = SummaryWriter(LOG_DIR)

# --- GESTION DE REPRISE (BACKUP) ---
start_episode = 0
frame_idx = 0
checkpoints = sorted(glob.glob(f"{CHECKPOINT_DIR}/*.pth"))

if checkpoints:
    latest_ckpt = checkpoints[-1]
    print(f"🔄 Restauration du backup : {latest_ckpt}")
    checkpoint = torch.load(latest_ckpt, weights_only=False) # weights_only=False pour charger optimizer
    agent.policy_net.load_state_dict(checkpoint['model_state'])
    agent.target_net.load_state_dict(checkpoint['model_state'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state'])
    start_episode = checkpoint['episode']
    frame_idx = checkpoint['frame_idx']
    # On recalcule epsilon
    epsilon = max(EPSILON_END, EPSILON_START - (frame_idx / EPSILON_DECAY))
    print(f"✅ Reprise à l'épisode {start_episode} (Epsilon: {epsilon:.3f})")
else:
    print("✨ Nouvel entraînement démarré.")
    epsilon = EPSILON_START

# --- BOUCLE D'ENTRAÎNEMENT ---
pbar = tqdm(range(start_episode, MAX_EPISODES), desc="Training")

for episode in pbar:
    state, info = env.reset()
    state = np.array(state)
    
    # Reset du Hidden State du LSTM pour le nouvel épisode
    hidden = agent.policy_net.init_hidden(1, device)
    
    total_reward = 0
    done = False
    episode_loss = []
    
    while not done:
        frame_idx += 1
        
        # 1. Action
        epsilon = max(EPSILON_END, EPSILON_START - (frame_idx / EPSILON_DECAY) * (EPSILON_START - EPSILON_END))
        action, next_hidden = agent.select_action(state, hidden, epsilon)
        
        # 2. Step Environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = np.array(next_state)
        done = terminated or truncated
        
        # 3. Stockage (Attention : on stocke des transitions simples, le buffer fera les séquences)
        agent.memory.push(state, action, reward, next_state, done)
        
        state = next_state
        hidden = next_hidden # On propage la mémoire LSTM
        total_reward += reward
        
        # 4. Learning (Mixed Precision Inside)
        loss = agent.train_step()
        if loss is not None:
            episode_loss.append(loss)
        
        # 5. Update Target Net
        if frame_idx % TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

    # --- LOGGING & BACKUP ---
    avg_loss = np.mean(episode_loss) if episode_loss else 0
    
    # TensorBoard
    writer.add_scalar('Reward/Episode', total_reward, episode)
    writer.add_scalar('Loss/Avg', avg_loss, episode)
    writer.add_scalar('Epsilon', epsilon, episode)
    
    # Barre de progression
    pbar.set_postfix({'Reward': f'{total_reward:.1f}', 'Eps': f'{epsilon:.2f}'})
    
    # Sauvegarde périodique
    if episode > 0 and episode % SAVE_INTERVAL == 0:
        save_path = f"{CHECKPOINT_DIR}/ckpt_ep_{episode}.pth"
        torch.save({
            'episode': episode,
            'frame_idx': frame_idx,
            'model_state': agent.policy_net.state_dict(),
            'optimizer_state': agent.optimizer.state_dict(),
        }, save_path)
        # Nettoyage des vieux checkpoints (optionnel)

env.close()
writer.close()
print("🏆 Entraînement terminé !")