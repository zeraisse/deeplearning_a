import torch
import numpy as np
import imageio
from gridEnv import gridEnv
from train import generate_dataset, train_model, DEVICE

def generate_video(model, filename="trm_result.mp4"):
    print(f"Generating video to {filename}...")
    env = gridEnv(size=6, render_mode="rgb_array")
    writer = imageio.get_writer(filename, fps=5)
    
    obs, _ = env.reset()
    done = False
    
    writer.append_data(env.render())
    
    step_count = 0
    while not done and step_count < 50:
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
            print("Victory in video!")

    writer.close()
    env.close()
    print("Video saved.")

if __name__ == "__main__":
    # 1. Dataset
    dataset = generate_dataset(episodes=600)
    
    # 2. Train TRM
    model = train_model(dataset, epochs=5)
    
    # 3. Video
    model.eval()
    generate_video(model)