import torch
import numpy as np
from gridEnv import gridEnv
from train import generate_dataset, train_model, DEVICE

def visual_test(model):
    test_env = gridEnv(size=6, render_mode="human")
    model.eval()

    for i in range(3): 
        obs, _ = test_env.reset()
        done = False
        print(f"Episode {i+1}...")
        
        while not done:
            test_env.render()
            
            img = obs['image'].astype(np.float32) / 255.0
            img_tensor = torch.tensor(img).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                logits = model(img_tensor)
                action = torch.argmax(logits).item()
                
            obs, reward, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated
            
            if terminated and reward > 0:
                print("VICTOIRE !")
                
    test_env.close()

if __name__ == "__main__":
    # 1. Création des données
    dataset = generate_dataset(episodes=600)
    
    # 2. Entraînement
    model = train_model(dataset, epochs=5)
    
    # 3. Test
    visual_test(model)