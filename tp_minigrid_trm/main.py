import torch
import numpy as np
import imageio
from gridEnv import gridEnv, GRID_SIZE
from train import generate_dataset, train_model, DEVICE
from minigrid.wrappers import FullyObsWrapper

from config import VIDEO_FILENAME, VIDEO_FPS, MAX_VIDEO_STEPS, GRID_SIZE, DEVICE

def generate_video(model, filename=VIDEO_FILENAME):
    print(f"Generating video...")
    env = FullyObsWrapper(gridEnv(size=GRID_SIZE, render_mode="rgb_array"))
    writer = imageio.get_writer(filename, fps=VIDEO_FPS)
    
    obs, _ = env.reset()
    done = False
    
    # 1ère frame
    frame = env.render()
    writer.append_data(frame)
    
    step_count = 0
    while not done and step_count < MAX_VIDEO_STEPS:
        # --- HUD LOGIC ---
        img = obs['image'].astype(np.float32) / 255.0
        if env.unwrapped.carrying:
            img[0, 0, :] = 1.0 
        # -----------------

        img_tensor = torch.tensor(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logits = model(img_tensor)
            action = torch.argmax(logits).item()
            
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        frame = env.render()
        writer.append_data(frame)
        step_count += 1

    writer.close()
    env.close()
    print(f"Video saved to {filename}")

if __name__ == "__main__":
    # 1. Dataset (utilise EPISODES de train.py par défaut)
    dataset = generate_dataset()
    
    # 2. Entraînement (utilise EPOCHS de train.py par défaut)
    model = train_model(dataset)
    
    # 3. Génération Vidéo
    model.eval()
    generate_video(model)