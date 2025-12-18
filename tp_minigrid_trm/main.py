import torch
import numpy as np
import imageio
# Import de TON env
from gridEnv import gridEnv, GRID_SIZE
from train import generate_dataset, train_model, DEVICE

VIDEO_FILENAME = "custom_env_result.mp4"
VIDEO_FPS = 5

def generate_video(model, filename=VIDEO_FILENAME):
    print(f"Generating video for custom gridEnv (Size {GRID_SIZE})...")
    env = gridEnv(size=GRID_SIZE, render_mode="rgb_array")
    writer = imageio.get_writer(filename, fps=VIDEO_FPS)
    
    obs, _ = env.reset()
    done = False
    
    writer.append_data(env.render())
    
    step_count = 0
    while not done and step_count < 60:
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
    # 1. Dataset Custom
    dataset = generate_dataset()
    
    # 2. Entraînement
    model = train_model(dataset)
    
    # 3. Vidéo
    model.eval()
    generate_video(model)