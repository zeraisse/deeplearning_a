import tensorflow as tf
from transformers import AutoTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# --- CONFIGURATION (Doit être identique à l'entraînement) ---
maxlen = 128
model_path = "squad_model_best.keras"

# --- RE-DEFINITION DES CLASSES (Nécessaire pour le chargement) ---
# Copie-colle EXACTEMENT les mêmes classes PositionalEmbedding, TransformerEncoderBlock, TransformerDecoderBlock ici
# (Pour alléger la réponse, je te laisse recopier les 3 classes du fichier train_qa.py ici)
# ... [INSÉRER LES CLASSES ICI] ...

# Si tu ne veux pas recopier, tu peux mettre les classes dans un fichier `models.py` et faire `from models import ...`
# Mais pour faire simple, recolle les classes PositionalEmbedding, TransformerEncoderBlock, TransformerDecoderBlock juste là.

# --- CHARGEMENT ---
print("📥 Chargement du Tokenizer et du Modèle...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Dictionnaire des objets custom pour Keras
custom_objects = {
    "PositionalEmbedding": PositionalEmbedding,
    "TransformerEncoderBlock": TransformerEncoderBlock,
    "TransformerDecoderBlock": TransformerDecoderBlock
}

model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
print("✅ Modèle chargé !")

def generate_answer(question, context):
    # 1. Préparer l'entrée (Encoder)
    input_text = f"{question} [SEP] {context}"
    tokenized_in = tokenizer(input_text, max_length=maxlen, padding='max_length', truncation=True)
    encoder_input = np.array([tokenized_in['input_ids']])
    
    # 2. Initialiser le Décodeur avec le token de départ [CLS] (ID 101 pour BERT)
    # On crée une séquence vide remplie de padding
    decoder_input = np.zeros((1, maxlen), dtype="int32")
    decoder_input[0, 0] = 101 # Start Token
    
    # 3. Boucle de génération mot par mot
    for i in range(maxlen - 1):
        # On prédit
        predictions = model.predict([encoder_input, decoder_input], verbose=0)
        
        # On regarde quelle est la prédiction pour le token actuel (i)
        # predictions shape: (1, 128, 30522)
        predicted_id = np.argmax(predictions[0, i, :])
        
        # Si le modèle prédit [SEP] (102), c'est fini
        if predicted_id == 102:
            break
            
        # Sinon, on ajoute le mot à l'entrée du décodeur pour le tour suivant
        decoder_input[0, i+1] = predicted_id

    # 4. Décoder les IDs en texte
    # On récupère tous les tokens générés (en ignorant les 0 du début qui n'ont pas été remplis)
    generated_ids = [id for id in decoder_input[0] if id not in [0, 101, 102]]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)

# --- INTERFACE ---
print("\n" + "="*50)
print("🤖 ORACLE SQuAD - Pose tes questions sur un texte")
print("="*50)

while True:
    print("\n--- NOUVEAU CONTEXTE ---")
    context = input("📜 Copie-colle ton texte/paragraphe ici : ")
    if not context: continue
    
    while True:
        question = input("\n❓ Ta question (ou 'new' pour changer de texte) : ")
        if question.lower() == 'new': break
        if question.lower() in ['exit', 'quit']: exit()
        
        print("🤔 Réflexion en cours...")
        try:
            answer = generate_answer(question, context)
            print(f"💡 Réponse : {answer}")
        except Exception as e:
            print(f"❌ Erreur : {e}")