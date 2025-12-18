import torch
from transformers import AutoTokenizer
from model_scratch import ExtractiveTransformer, MAX_LEN, VOCAB_SIZE, DEVICE

# Chargement
MODEL_PATH = "squad_scratch_extractive_best.pth"
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model = ExtractiveTransformer().to(DEVICE)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Modèle Extractive Loaded!")
except:
    print("Fichier non trouvé.")
    exit()

def get_answer(question, context):
    inputs = tokenizer(
        question,
        context,
        max_length=MAX_LEN,
        padding="max_length",
        truncation="only_second",
        return_tensors="pt"
    )
    
    input_ids = inputs["input_ids"].to(DEVICE)
    mask = inputs["attention_mask"].to(DEVICE)
    
    with torch.no_grad():
        start_logits, end_logits = model(input_ids, mask)
    
    # On prend les indices avec la plus haute probabilité
    start_idx = torch.argmax(start_logits)
    end_idx = torch.argmax(end_logits)
    
    # Si la fin est avant le début, c'est une erreur, on inverse ou on prend 1 mot
    if end_idx < start_idx:
        end_idx = start_idx + 10 # Fallback
        
    tokens = input_ids[0][start_idx : end_idx + 1]
    return tokenizer.decode(tokens, skip_special_tokens=True)

# Chat Loop
while True:
    context = input("\nContext: ")
    if not context: continue
    question = input("Question: ")
    
    try:
        print(f"💡 Réponse: {get_answer(question, context)}")
    except Exception as e:
        print(e)