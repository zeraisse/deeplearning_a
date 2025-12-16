import torch
import deepinv
from torchvision.utils import save_image
import os
import glob

# --- CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_size = 64
channels = 3
timestamps = 1000

# 👇 MODIFIE CECI avec le dernier fichier .pth qui existe dans ton dossier !
model_path = "ddpm_ffhq_ep70.pth"

# --- 1. CHARGER LE MODÈLE ---
print(f"Chargement sur {device}...")
model = deepinv.models.DiffUNet(in_channels=channels, out_channels=channels, pretrained=None).to(device)

try:
    # map_location est important si tu as sauvé sur GPU et que tu charges sur CPU
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Modèle {model_path} chargé !")
except FileNotFoundError:
    print(f"Le fichier {model_path} n'existe pas encore.")
    print("Regarde dans ton dossier et change la variable 'model_path' avec un fichier 'epX.pth' existant.")
    exit()

# --- 2. PARAMÈTRES MATHÉMATIQUES ---
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, timestamps).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
sqrt_inv_alphas = torch.sqrt(1.0 / alphas)

# --- 3. GÉNÉRATION ---
def sample(num_images=16):
    print(f"Génération de {num_images} visages en cours...")
    
    x = torch.randn(num_images, channels, image_size, image_size).to(device)
    
    with torch.no_grad():
        for i in range(timestamps - 1, -1, -1):
            t = torch.tensor([i] * num_images, device=device)
            
            # Prédiction du modèle
            output = model(x, t, type_t='timestep')
            
            # --- CORRECTION CRITIQUE (Comme dans train.py) ---
            # Si le modèle sort 6 canaux (bruit + variance), on ne garde que les 3 premiers
            if output.shape[1] == 2 * channels:
                predicted_noise, _ = torch.split(output, channels, dim=1)
            else:
                predicted_noise = output
            # -------------------------------------------------

            beta_t = betas[i]
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[i]
            sqrt_inv_alpha_t = sqrt_inv_alphas[i]
            
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            x = sqrt_inv_alpha_t * (x - (beta_t / sqrt_one_minus_alpha_cumprod_t) * predicted_noise)
            x += torch.sqrt(beta_t) * noise
            
            if i % 100 == 0:
                print(f"Step {i}/{timestamps}...")

    x = (x + 1) / 2.0
    return x

# --- 4. SAUVEGARDE ---
images = sample(num_images=64) 
save_name = f"gen_{model_path.replace('.pth', '')}.png"
save_image(images, save_name, nrow=8)
print(f"Image générée : {save_name}")