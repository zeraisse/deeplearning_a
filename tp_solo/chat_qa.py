import torch
import glob
import os
import warnings
from transformers import AutoTokenizer

# On importe la classe du modèle et les params depuis ton fichier d'entraînement
# Assure-toi que le fichier s'appelle bien 'train_qa.py'
try:
    from train_qa import TransformerModel, MAX_LEN
except ImportError:
    print("❌ Erreur : Impossible d'importer TransformerModel depuis train_qa.py")
    print("Vérifie que tu es dans le bon dossier.")
    exit()

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")

# --- 1. RECHERCHE AUTOMATIQUE DU DERNIER CHECKPOINT ---
# On cherche dans le dossier par défaut de Lightning
list_of_checkpoints = glob.glob('logs/squad_experiment/**/checkpoints/*.ckpt', recursive=True)

# Si vide, on cherche à la racine ou dans le dossier checkpoints simple
if not list_of_checkpoints:
    list_of_checkpoints = glob.glob('checkpoints/*.ckpt')

if not list_of_checkpoints:
    print("❌ Aucun checkpoint trouvé (ni dans logs/, ni dans checkpoints/).")
    print("Lance d'abord : python train_qa.py")
    exit()
else:
    # On prend le fichier le plus récent
    MODEL_PATH = max(list_of_checkpoints, key=os.path.getctime)
    print(f"📂 Checkpoint trouvé : {MODEL_PATH}")

# --- 2. CHARGEMENT ---
print("⏳ Chargement du Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

print(f"🏗️  Chargement de l'architecture...")
model = TransformerModel().to(DEVICE)

print("⚖️  Chargement des poids...")
try:
    # --- CHARGEMENT LIGHTNING ---
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # Si c'est un checkpoint Lightning, les clés commencent par "model."
    # Si c'est un save manuel, non. On gère les deux cas.
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    new_state_dict = {}
    for key, value in state_dict.items():
        # On nettoie le préfixe "model." ajouté par Lightning
        if key.startswith("model."):
            new_key = key.replace("model.", "") 
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
            
    model.load_state_dict(new_state_dict)
    print("✅ Poids chargés avec succès !")
    
    model.eval()

except RuntimeError as e:
    print(f"❌ Erreur d'architecture : {e}")
    print("Conseil : Vérifie que tes hyperparamètres (EMBED_DIM, LAYERS, etc.) dans train_qa.py sont les mêmes qu'à l'entraînement.")
    exit()

# --- 3. GÉNÉRATION (Via Beam Search) ---
def generate_answer(question, context):
    # Préparation
    input_text = f"{question} [SEP] {context}"
    enc_tokens = tokenizer(input_text, max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
    
    # Ajout de la dimension de batch (unsqueeze) car le modèle attend [Batch, Seq]
    src = enc_tokens['input_ids'].to(DEVICE)

    with torch.no_grad():
        # Appel direct à la fonction beam_search de ton modèle
        # Elle gère déjà le [CLS] de départ et la boucle
        generated_ids = model.beam_search(src, tokenizer, beam_width=3, max_gen_len=30)

    # Décodage
    return tokenizer.decode(generated_ids.squeeze(), skip_special_tokens=True)

# --- 4. INTERFACE ---
print("\n" + "="*50)
print(f"🤖 ORACLE SQuAD (Génératif)")
print("="*50)

while True:
    print("\n--- 📝 NOUVEAU CONTEXTE ---")
    context = input("Texte : ")
    if not context.strip(): continue
    if context.lower() in ['exit', 'quit', 'q']: break
    
    while True:
        question = input("\n❓ Question (ou 'new' pour changer de texte) : ")
        if question.lower() == 'new': break
        if question.lower() in ['exit', 'quit', 'q']: exit()
        
        try:
            print("🤔 Réflexion...")
            reponse = generate_answer(question, context)
            print(f"💡 Réponse : \033[1m{reponse}\033[0m") # En gras
        except Exception as e:
            print(f"❌ Erreur lors de la génération : {e}")