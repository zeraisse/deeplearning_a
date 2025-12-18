import torch
import numpy as np
import imageio
import gymnasium as gym
import minigrid
from train import generate_dataset, train_model, DEVICE, ENV_ID

VIDEO_FILENAME = "trm_gym_result.mp4"
VIDEO_FPS = 8  # Un peu plus rapide car 16x16 c'est grand

def generate_video(model, filename=VIDEO_FILENAME):
    print(f"Generating video for {ENV_ID}...")
    env = gym.make(ENV_ID, render_mode="rgb_array")
    writer = imageio.get_writer(filename, fps=VIDEO_FPS)
    
    obs, _ = env.reset()
    done = False
    
    # Enregistrement première frame
    writer.append_data(env.render())
    
    step_count = 0
    while not done and step_count < 200: # Max 200 steps pour éviter les fichiers énormes
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
            print("VICTOIRE !")

    writer.close()
    env.close()
    print(f"Video saved to {filename}")

if __name__ == "__main__":
    # 1. Dataset Gymnasium
    dataset = generate_dataset()
    
    # 2. Entraînement
    model = train_model(dataset)
    
    # 3. Vidéo
    model.eval()
    generate_video(model)