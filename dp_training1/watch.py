import torch
import numpy as np
from utils import make_env
from agent import DARQNAgent
import glob
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

# --- CONFIG ---
ENV_NAME = "ALE/Pong-v5"  # <--- Bien vérifier que c'est PONG
CHECKPOINT_DIR = "checkpoints"
VIDEO_FOLDER = "videos"   # Dossier où la vidéo sera sauvegardée

def load_latest_checkpoint(agent):
    checkpoints = sorted(glob.glob(f"{CHECKPOINT_DIR}/ckpt_ep_450.pth"))
    if not checkpoints:
        print("❌ Aucun checkpoint trouvé !")
        return None, 0
    latest_ckpt = checkpoints[-1]
    print(f"🍿 Chargement du cerveau : {latest_ckpt}")
    checkpoint = torch.load(latest_ckpt, map_location=agent.device, weights_only=False)
    agent.policy_net.load_state_dict(checkpoint['model_state'])
    agent.policy_net.eval()
    return latest_ckpt, checkpoint['episode']

def record_game():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. On utilise render_mode="rgb_array" (INDISPENSABLE pour enregistrer sans écran)
    env = make_env(ENV_NAME, render_mode="rgb_array")
    
    # 2. On ajoute le wrapper Caméra
    env = RecordVideo(
        env, 
        video_folder=VIDEO_FOLDER, 
        name_prefix="replay_pong",
        episode_trigger=lambda x: True, # Enregistre tous les épisodes
        disable_logger=True
    )

    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    agent = DARQNAgent(obs_shape, n_actions, device)

    # Chargement
    last_ckpt, episode = load_latest_checkpoint(agent)
    if not last_ckpt: return

    print(f"🎥 Enregistrement en cours (Modèle Ep {episode})...")
    print("⏳ Patiente quelques secondes, une partie de Pong dure un peu...")

    # Une seule partie suffit pour la démo
    state, info = env.reset()
    state = np.array(state)
    hidden = agent.policy_net.init_hidden(1, device)
    done = False
    total_reward = 0
    
    while not done:
        # Jeu sérieux (Epsilon bas)
        action, next_hidden = agent.select_action(state, hidden, epsilon=0.01)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        state = np.array(next_state)
        hidden = next_hidden

    env.close() # Important pour finaliser le fichier vidéo
    print(f"🏁 Partie terminée ! Score : {total_reward}")
    print(f"✅ Vidéo sauvegardée dans le dossier : {VIDEO_FOLDER}/")

if __name__ == "__main__":
    record_game()