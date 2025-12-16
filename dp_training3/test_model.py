import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, Embedding, MultiHeadAttention
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import numpy as np

# --- CONFIGURATION (Doit être identique à l'entraînement) ---
maxlen = 200
vocab_size = 10000
embed_dim = 128
num_heads = 4
ff_dim = 512

# --- 1. DÉFINITION DES CLASSES (Obligatoire pour charger le modèle) ---
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    # Nécessaire pour la sauvegarde/chargement
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config

class Transformer(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.embedding = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.trans_block = TransformerBlock(embed_dim, num_heads, ff_dim, rate)
        self.final_layer = Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = x + positions
        x = self.trans_block(x, training=training)
        x = x[:, -1, :] 
        return self.final_layer(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config

# --- 2. CHARGEMENT DES FICHIERS ---
print("⚙️ Chargement du Tokenizer...")
try:
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    print("✅ Tokenizer chargé.")
except FileNotFoundError:
    print("❌ Erreur : 'tokenizer.pickle' introuvable. Lance l'entraînement d'abord !")
    exit()

print("⚙️ Chargement du Modèle...")
try:
    # On précise les classes personnalisées dans custom_objects
    model = tf.keras.models.load_model(
        'imdb_transformer.keras',
        custom_objects={'Transformer': Transformer, 'TransformerBlock': TransformerBlock}
    )
    print("✅ Modèle chargé.")
except Exception as e:
    print(f"❌ Erreur chargement modèle : {e}")
    exit()

# --- 3. FONCTIONS UTILITAIRES ---
def clean_text(text):
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

def predict_sentiment(text):
    # 1. Nettoyer
    cleaned_text = clean_text(text)
    # 2. Convertir en chiffres (Tokenize)
    seq = tokenizer.texts_to_sequences([cleaned_text])
    # 3. Padding (Ajouter des zéros pour avoir la taille 200)
    padded = pad_sequences(seq, maxlen=maxlen, padding='post', truncating='post')
    
    # 4. Prédiction
    prediction = model.predict(padded, verbose=0)
    score = prediction[0][0]
    
    return score

# --- 4. BOUCLE INTERACTIVE ---
print("\n" + "="*50)
print("🎬 TESTEUR DE CRITIQUES DE FILMS (TRANSFORMER)")
print("Tape 'exit' ou 'quit' pour quitter.")
print("="*50 + "\n")

while True:
    user_input = input("✍️  Écris une critique (en anglais) : ")
    
    if user_input.lower() in ['exit', 'quit']:
        print("Bye bye ! 👋")
        break
    
    if not user_input.strip():
        continue
        
    try:
        score = predict_sentiment(user_input)
        
        # Affichage du résultat
        print("-" * 30)
        if score > 0.5:
            # Plus le score est proche de 1, plus c'est positif
            confidence = (score - 0.5) * 2 * 100
            print(f"✅ POSITIF (Confiance: {confidence:.2f}%)")
            print(f"   Score brut : {score:.4f}")
        else:
            # Plus le score est proche de 0, plus c'est négatif
            confidence = (0.5 - score) * 2 * 100
            print(f"❌ NÉGATIF (Confiance: {confidence:.2f}%)")
            print(f"   Score brut : {score:.4f}")
        print("-" * 30 + "\n")
        
    except Exception as e:
        print(f"⚠️ Erreur lors de la prédiction : {e}")