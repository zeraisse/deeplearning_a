import torch
import deepinv
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import glob
import re  # Pour extraire le numéro de l'époque du nom de fichier

# --- CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64        
image_size = 64        
channels = 3           
lr = 1e-4
epochs = 100
timestamps = 1000      
save_interval = 5      # Sauvegarde tous les X epochs

# --- 1. DATASET ---
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

DATASET_PATH = "./ffhq_dataset" 

try:
    dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)
    print(f"✅ Dataset chargé : {len(dataset)} images trouvées.")
except Exception as e:
    print(f"⚠️ Erreur chargement FFHQ, fallback sur MNIST.")
    dataset = datasets.MNIST('./data', train=True, download=True, 
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,))
                           ]))
    channels = 1 

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# --- 2. MODÈLE ---
model = deepinv.models.DiffUNet(in_channels=channels, out_channels=channels, pretrained=None).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
mse = torch.nn.MSELoss() 

# --- 3. NOISE SCHEDULE ---
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, timestamps).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

# --- 4. LOGIQUE DE REPRISE (RESUME) ---
start_epoch = 0

# On cherche tous les fichiers qui ressemblent à 'ddpm_ffhq_ep*.pth'
checkpoints = glob.glob("ddpm_ffhq_ep*.pth")

if checkpoints:
    # On trie pour trouver le plus grand numéro (ex: ep100 > ep9)
    # On utilise une fonction lambda pour extraire le chiffre du nom de fichier
    latest_ckpt = max(checkpoints, key=lambda f: int(re.search(r'ep(\d+)', f).group(1)))
    
    print(f"🔄 Checkpoint trouvé : {latest_ckpt}")
    print("⏳ Chargement du modèle...")
    
    try:
        # On charge les poids
        state_dict = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(state_dict)
        
        # On met à jour l'époque de départ
        start_epoch = int(re.search(r'ep(\d+)', latest_ckpt).group(1))
        print(f"✅ Reprise confirmée à l'époque {start_epoch} !")
    except Exception as e:
        print(f"❌ Erreur lors du chargement du backup : {e}")
        print("⚠️ Démarrage à zéro par sécurité.")
        start_epoch = 0
else:
    print("✨ Aucun checkpoint trouvé. Démarrage d'un nouvel entraînement.")

print(f"🚀 Démarrage de l'entraînement sur {device} (Epoch {start_epoch} -> {epochs})...")

# --- 5. TRAINING LOOP ---
# On commence la boucle à 'start_epoch' au lieu de 0
for epoch in range(start_epoch, epochs):
    model.train()
    epoch_loss = 0
    
    for i, (imgs, _) in enumerate(train_loader):
        imgs = imgs.to(device)
        print(f"Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}", end='\r')
        
        noise = torch.randn_like(imgs)
        t = torch.randint(0, timestamps, (imgs.size(0),), device=device)
        
        sqrt_alpha_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        noised_imgs = sqrt_alpha_t * imgs + sqrt_one_minus_alpha_t * noise
        
        optimizer.zero_grad()
        output = model(noised_imgs, t, type_t='timestep')
        
        if output.shape[1] == 2 * channels:
            estimated_noise, estimated_variance = torch.split(output, channels, dim=1)
        else:
            estimated_noise = output

        loss = mse(estimated_noise, noise)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(train_loader):.4f}")
    
    # Sauvegarde périodique
    if (epoch+1) % save_interval == 0:
        save_name = f'ddpm_ffhq_ep{epoch+1}.pth'
        torch.save(model.state_dict(), save_name)
        print(f"💾 Sauvegarde : {save_name}")

# Sauvegarde finale
torch.save(model.state_dict(), 'ddpm_ffhq_final.pth')
print("✅ Modèle final sauvegardé.")