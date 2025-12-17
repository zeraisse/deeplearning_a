import torch
from transformers import AutoTokenizer
from model import TransformerModel, MAX_LEN

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "squad_pytorch_model.pth"

# --- CHARGEMENT ---
print("Chargement du Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

print(f"Chargement du Modèle depuis {MODEL_PATH}...")
model = TransformerModel().to(DEVICE)

try:
    # On charge les poids
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Modèle chargé et prêt !")
except FileNotFoundError:
    print(f"Erreur : Le fichier {MODEL_PATH} est introuvable.")
    exit()

# --- GÉNÉRATION ---
def generate_answer(question, context):
    input_text = f"{question} [SEP] {context}"
    enc_tokens = tokenizer(input_text, max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
    src = enc_tokens['input_ids'].to(DEVICE)
    
    tgt_input = torch.tensor([[101]], device=DEVICE) # [CLS]

    with torch.no_grad():
        for _ in range(MAX_LEN):
            # Le modèle gère le masque et le device automatiquement (voir model.py)
            output = model(src, tgt_input)
            
            next_token_logits = output[:, -1, :] 
            next_token_id = next_token_logits.argmax(dim=-1).unsqueeze(0)
            
            if next_token_id.item() == 102: # [SEP]
                break
            
            tgt_input = torch.cat([tgt_input, next_token_id], dim=1)

    generated_ids = tgt_input[0, 1:] 
    return tokenizer.decode(generated_ids, skip_special_tokens=True)

print("\n" + "="*50)
print("="*50)

while True:
    print("\n--- NOUVEAU CONTEXTE ---")
    context = input("Texte : ")
    if not context.strip(): continue
    if context.lower() in ['exit', 'quit']: break
    
    while True:
        question = input("\nQuestion : ")
        if question.lower() == 'new': break
        if question.lower() in ['exit', 'quit']: exit()
        
        try:
            print(f"💡 Réponse : {generate_answer(question, context)}")
        except Exception as e:
            print(f"Erreur : {e}")