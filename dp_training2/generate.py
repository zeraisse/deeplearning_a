import torch
import deepinv
from torchvision.utils import save_image
import os

# --- CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 64
channels = 3
timestamps = 1000
model_path = "ddpm_ffhq_final.pth" # Ou "ddpm_ffhq_ep10.pth" si tu veux tester en cours de route

# --- 1. CHARGER LE MODÈLE ---
model = deepinv.models.DiffUNet(in_channels=channels, out_channels=channels, pretrained=None).to(device)

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Modèle {model_path} chargé !")
except FileNotFoundError:
    print(f"❌ Pas de fichier {model_path}. Lance l'entraînement d'abord !")
    exit()

# --- 2. PARAMÈTRES MATHÉMATIQUES (IDENTIQUES AU TRAIN) ---
# Il est CRUCIAL d'avoir exactement les mêmes betas
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, timestamps).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
# Calculs pour le processus inverse (Reverse Diffusion)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
sqrt_inv_alphas = torch.sqrt(1.0 / alphas)
posterior_variance = betas * (1.0 - torch.cat([torch.tensor([1.0]).to(device), alphas_cumprod[:-1]])) / (1.0 - alphas_cumprod)

# --- 3. GÉNÉRATION (REVERSE PROCESS) ---
def sample(num_images=16):
    print(f"🎨 Génération de {num_images} visages...")
    
    # Étape 1 : On part du bruit pur (Pure Noise)
    x = torch.randn(num_images, channels, image_size, image_size).to(device)
    
    # Étape 2 : La boucle inverse (De 1000 à 0)
    # C'est là que la magie opère : on "sculpte" le bruit
    with torch.no_grad():
        for i in range(timestamps - 1, -1, -1):
            t = torch.tensor([i] * num_images, device=device)
            
            # Le modèle prédit le bruit à enlever
            predicted_noise = model(x, t, type_t='timestep')
            
            # Formule mathématique du DDPM (ne t'inquiète pas, c'est la formule standard)
            # x_{t-1} = 1/sqrt(alpha) * (x_t - (1-alpha)/sqrt(1-alpha_bar) * epsilon) + sigma * z
            
            beta_t = betas[i]
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[i]
            sqrt_inv_alpha_t = sqrt_inv_alphas[i]
            
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x) # Pas de bruit à la toute dernière étape
            
            x = sqrt_inv_alpha_t * (x - (beta_t / sqrt_one_minus_alpha_cumprod_t) * predicted_noise)
            x += torch.sqrt(beta_t) * noise # On rajoute un petit peu d'aléatoire (Langevin dynamics)
            
            if i % 100 == 0:
                print(f"⏳ Denoising step {i}/{timestamps}...")

    # Étape 3 : On remet les pixels entre 0 et 1 pour l'affichage
    x = (x + 1) / 2.0
    return x

# --- 4. SAUVEGARDE ---
images = sample(num_images=16)
save_image(images, "resultat_generation.png", nrow=4)
print("✨ Image générée : resultat_generation.png")