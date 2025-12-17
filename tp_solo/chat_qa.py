import torch
import glob
import os
from transformers import AutoTokenizer
# On importe l'architecture définie dans model.py
from model import TransformerModel, MAX_LEN, NUM_LAYERS

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. RECHERCHE AUTOMATIQUE DU DERNIER CHECKPOINT LIGHTNING ---
list_of_checkpoints = glob.glob('checkpoints/*.ckpt')

if not list_of_checkpoints:
    # Fallback sur l'ancien fichier si aucun checkpoint Lightning n'existe
    MODEL_PATH = "squad_pytorch_model.pth"
    print("Aucun checkpoint Lightning trouvé, recherche de l'ancien .pth...")
else:
    MODEL_PATH = max(list_of_checkpoints, key=os.path.getctime)
    print(f"Checkpoint Lightning trouvé : {MODEL_PATH}")

# --- 2. CHARGEMENT ---
print("Chargement du Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

print(f"Chargement de l'architecture...")
# Attention : Verifier que NUM_LAYERS dans model.py est bien le même 
# que celui utilisé pour l'entraînement (16) !
model = TransformerModel().to(DEVICE)

print("Chargement des poids...")
try:
    if ".ckpt" in MODEL_PATH:
        # --- SPÉCIAL LIGHTNING ---
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        state_dict = checkpoint['state_dict']
        
        # Lightning ajoute un préfixe "model." devant toutes les variables.
        # Nous devons l'enlever pour que ça rentre dans TransformerModel.
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                new_key = key.replace("model.", "") # On enlève 'model.'
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
                
        model.load_state_dict(new_state_dict)
        print("Poids Lightning chargés et adaptés avec succès !")
        
    else:
        # --- CLASSIQUE PYTORCH ---
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Poids standards chargés !")

    model.eval()

except FileNotFoundError:
    print(f"Erreur : Le fichier {MODEL_PATH} est introuvable.")
    exit()
except RuntimeError as e:
    print(f"Erreur d'architecture : {e}")
    print("Conseil : Vérifie que NUM_LAYERS dans model.py est identique à l'entraînement.")
    exit()

# --- 3. GÉNÉRATION (Inchangée, ta logique était bonne) ---
def generate_answer(question, context):
    input_text = f"{question} [SEP] {context}"
    enc_tokens = tokenizer(input_text, max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
    src = enc_tokens['input_ids'].to(DEVICE)
    
    # On commence la réponse par [CLS] (101)
    tgt_input = torch.tensor([[101]], device=DEVICE)

    with torch.no_grad():
        for _ in range(MAX_LEN):
            # Le modèle gère les masques en interne via model.py
            output = model(src, tgt_input)
            
            # On regarde le dernier token prédit
            next_token_logits = output[:, -1, :] 
            next_token_id = next_token_logits.argmax(dim=-1).unsqueeze(0)
            
            # Si c'est [SEP] (102), on arrête
            if next_token_id.item() == 102:
                break
            
            # Sinon on l'ajoute à la suite et on recommence
            tgt_input = torch.cat([tgt_input, next_token_id], dim=1)

    generated_ids = tgt_input[0, 1:] # On ignore le [CLS] de départ
    return tokenizer.decode(generated_ids, skip_special_tokens=True)

# --- 4. INTERFACE ---
print("\n" + "="*50)
print("🤖 ORACLE SQuAD (16 Layers)")
print("="*50)

while True:
    print("\n--- NOUVEAU CONTEXTE ---")
    context = input("📜 Texte : ")
    if not context.strip(): continue
    if context.lower() in ['exit', 'quit']: break
    
    while True:
        question = input("\n❓ Question : ")
        if question.lower() == 'new': break
        if question.lower() in ['exit', 'quit']: exit()
        
        try:
            print("🤔 Réflexion...")
            reponse = generate_answer(question, context)
            print(f"💡 Réponse : {reponse}")
        except Exception as e:
            print(f"❌ Erreur : {e}")