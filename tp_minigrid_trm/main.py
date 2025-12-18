import torch
import numpy as np
import imageio
from gridEnv import gridEnv, GRID_SIZE
from train import generate_dataset, train_model, DEVICE

# --- HYPERPARAMETERS VIDEO ---
VIDEO_FILENAME = "trm_result.mp4"
VIDEO_FPS = 5
MAX_VIDEO_STEPS = 60

def generate_video(model, filename=VIDEO_FILENAME):
    print(f"Generating video to {filename}...")
    env = gridEnv(size=GRID_SIZE, render_mode="rgb_array")
    writer = imageio.get_writer(filename, fps=VIDEO_FPS)
    
    obs, _ = env.reset()
    done = False
    
    # Première frame
    writer.append_data(env.render())
    
    step_count = 0
    while not done and step_count < MAX_VIDEO_STEPS:
        img = obs['image'].astype(np.float32) / 255.0
        img_tensor = torch.tensor(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logits = model(img_tensor)
            action = torch.argmax(logits).item()
            
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        writer.append_data(env.render())
        step_count += 1
        
        if terminated and reward > 0:
            print("VICTORY detected in video generation!")

    writer.close()
    env.close()
    print("Video generation complete.")

if __name__ == "__main__":
    # 1. Dataset (utilise EPISODES de train.py par défaut)
    dataset = generate_dataset()
    
    # 2. Entraînement (utilise EPOCHS de train.py par défaut)
    model = train_model(dataset)
    
    # 3. Génération Vidéo
    model.eval()
    generate_video(model)