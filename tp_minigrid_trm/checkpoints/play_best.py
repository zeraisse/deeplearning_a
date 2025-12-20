import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
# ----------------------------------------

import torch
import numpy as np
import imageio
import cv2
from gridEnv import gridEnv
from train import TRMAgent
from config import (
    DEVICE, GRID_SIZE, VIDEO_FPS, MAX_VIDEO_STEPS,
    # Attention: config.py définit le nom du fichier, mais pas son chemin complet.
    # On va utiliser le fichier local au sous-dossier.
    BEST_MODEL_FILE 
)
from minigrid.wrappers import FullyObsWrapper

# On force le chemin du modèle pour qu'il cherche dans le dossier COURANT (le sous-dossier)
# et pas à la racine, puisque tu as dit que le modèle était avec le script.
LOCAL_MODEL_PATH = os.path.join(current_dir, os.path.basename(BEST_MODEL_FILE))
OUTPUT_VIDEO = os.path.join(current_dir, "gameplay_demonstration.mp4")

def run_demo():
    print(f"🎬 Chargement du champion depuis : {LOCAL_MODEL_PATH}")
    
    # 1. Charger l'environnement
    env = FullyObsWrapper(gridEnv(size=GRID_SIZE, render_mode="rgb_array"))
    
    # 2. Charger le modèle
    model = TRMAgent().to(DEVICE)
    
    if not os.path.exists(LOCAL_MODEL_PATH):
        print(f"❌ ERREUR : Le fichier '{LOCAL_MODEL_PATH}' est introuvable !")
        print("Vérifie qu'il est bien dans le même dossier que ce script.")
        return

    try:
        model.load_state_dict(torch.load(LOCAL_MODEL_PATH, weights_only=False))
    except Exception as e:
        print(f"⚠️ Erreur chargement ({e}), tentative classique...")
        model.load_state_dict(torch.load(LOCAL_MODEL_PATH))

    model.eval()

    # 3. Préparer la vidéo
    print(f"🎥 Enregistrement dans '{OUTPUT_VIDEO}'...")
    writer = imageio.get_writer(OUTPUT_VIDEO, fps=VIDEO_FPS)
    
    obs, _ = env.reset()
    done = False
    step = 0
    
    frame = env.render()
    frame_big = cv2.resize(frame, (600, 600), interpolation=cv2.INTER_NEAREST)
    writer.append_data(frame_big)

    print("▶️ Simulation lancée...")

    while not done and step < MAX_VIDEO_STEPS:
        img = obs['image'].astype(np.float32) / 255.0
        img_tensor = torch.tensor(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logits = model(img_tensor)
            action = torch.argmax(logits).item()
        
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step += 1
        
        frame = env.render()
        frame_big = cv2.resize(frame, (600, 600), interpolation=cv2.INTER_NEAREST)
        writer.append_data(frame_big)
        
        if step % 10 == 0:
            print(f"\rStep {step}/{MAX_VIDEO_STEPS} | Action: {action}", end="")

    writer.close()
    env.close()
    print(f"\n✅ Terminé ! Vidéo : {OUTPUT_VIDEO}")
    
    if terminated and reward > 0:
        print("🏆 VICTOIRE !")
    elif truncated:
        print("⏱️ TIMEOUT")
    else:
        print("☠️ ECHEC")

if __name__ == "__main__":
    run_demo()