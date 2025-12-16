import torch
import deepinv
from torchvision.utils import save_image

# --- CONFIG SÉCURISÉE (CPU) ---
# On utilise le CPU pour ne pas faire planter l'entraînement en cours
device = torch.device("cpu") 
image_size = 64
channels = 3
timestamps = 1000

# On prend le dernier checkpoint disponible
model_path = "ddpm_ffhq_ep65.pth" 

print(f"🕵️ Test rapide du modèle {model_path} sur CPU...")

# 1. Chargement
try:
    model = deepinv.models.DiffUNet(in_channels=channels, out_channels=channels, pretrained=None).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print("✅ Modèle chargé !")
except Exception as e:
    print(f"❌ Erreur : {e}")
    exit()

# 2. Maths (Schedule)
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, timestamps).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
sqrt_inv_alphas = torch.sqrt(1.0 / alphas)

# 3. Génération rapide (Juste 8 images)
def sample(num_images=8):
    print("🎨 Génération en cours... (Patience, le CPU est plus lent)")
    x = torch.randn(num_images, channels, image_size, image_size).to(device)
    
    with torch.no_grad():
        for i in range(timestamps - 1, -1, -1):
            t = torch.tensor([i] * num_images, device=device)
            output = model(x, t, type_t='timestep')
            print("generation step", i)
            
            if output.shape[1] == 2 * channels:
                predicted_noise, _ = torch.split(output, channels, dim=1)
            else:
                predicted_noise = output

            beta_t = betas[i]
            sqrt_inv_alpha_t = sqrt_inv_alphas[i]
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[i]
            
            if i > 0: noise = torch.randn_like(x)
            else: noise = torch.zeros_like(x)
            
            x = sqrt_inv_alpha_t * (x - (beta_t / sqrt_one_minus_alpha_cumprod_t) * predicted_noise)
            x += torch.sqrt(beta_t) * noise
            
            if i % 200 == 0: print(f"Step {i}...")
            
    return (x.clamp(-1, 1) + 1) / 2.0

# 4. Sauvegarde
imgs = sample(8)
save_image(imgs, "test_ep65.png", nrow=4)
print("\n✨ Regarde l'image 'test_ep65.png' !")