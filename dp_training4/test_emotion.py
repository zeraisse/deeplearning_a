import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, Embedding, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from datasets import load_dataset
import numpy as np

# --- 1. CONFIGURATION ---
vocab_size = 10000
maxlen = 100
model_path = "transformer_emotion.keras"

# Dictionnaire pour traduire les chiffres (0-5) en mots
emotion_labels = {
    0: "😢 Tristesse (Sadness)",
    1: "😃 Joie (Joy)",
    2: "❤️ Amour (Love)",
    3: "😡 Colère (Anger)",
    4: "😱 Peur (Fear)",
    5: "😲 Surprise (Surprise)"
}

# --- 2. DÉFINITION DES CLASSES (Obligatoire pour charger le modèle) ---
# On copie-colle exactement les mêmes classes que pour l'entraînement
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([Dense(ff_dim, activation="relu"), Dense(embed_dim)])
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
    
    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads, "ff_dim": self.ff_dim, "rate": self.rate})
        return config

class Transformer(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_classes, rate=0.1, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_classes = num_classes
        self.rate = rate
        self.embedding = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.trans_block = TransformerBlock(embed_dim, num_heads, ff_dim, rate)
        self.pool = GlobalAveragePooling1D()
        self.final_layer = Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = x + positions
        x = self.trans_block(x, training=training)
        x = self.pool(x)
        return self.final_layer(x)

    def get_config(self):
        config = super().get_config()
        config.update({"vocab_size": self.vocab_size, "embed_dim": self.embed_dim, "num_heads": self.num_heads, "ff_dim": self.ff_dim, "num_classes": self.num_classes, "rate": self.rate})
        return config

# --- 3. PRÉPARATION (Tokenizer) ---
print("⚙️ Reconstruction du Tokenizer (Dictionnaire)...")
ds = load_dataset("dair-ai/emotion", "split")
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<UNK>")
tokenizer.fit_on_texts(ds['train']['text'])
print("✅ Tokenizer prêt.")

# --- 4. CHARGEMENT DU MODÈLE ---
print(f"📥 Chargement du modèle {model_path}...")
try:
    # On doit dire à Keras où trouver nos classes 'Transformer' et 'TransformerBlock'
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"Transformer": Transformer, "TransformerBlock": TransformerBlock}
    )
    print("✅ Modèle chargé avec succès !")
except Exception as e:
    print(f"❌ Erreur : {e}")
    print("Vérifie que le fichier .keras existe bien.")
    exit()

# --- 5. BOUCLE DE TEST ---
print("\n" + "="*50)
print("🧠 DÉTECTEUR D'ÉMOTIONS (6 CLASSES)")
print("Tape une phrase en anglais pour tester.")
print("Tape 'exit' pour quitter.")
print("="*50 + "\n")

while True:
    text = input("✍️  Ta phrase : ")
    
    if text.lower() in ["exit", "quit", "q"]:
        print("Bye bye ! 👋")
        break
        
    if not text.strip():
        continue

    # 1. Transformer le texte en chiffres
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=maxlen, padding='post', truncating='post')

    # 2. Prédiction
    prediction = model.predict(padded, verbose=0)[0] # On récupère le vecteur de probabilités

    # 3. Trouver le gagnant (argmax)
    best_class_index = np.argmax(prediction) # L'index du plus grand chiffre
    confidence = prediction[best_class_index] * 100 # Le pourcentage

    label_name = emotion_labels[best_class_index]

    # 4. Affichage
    print(f"--> Résultat : {label_name}")
    print(f"--> Confiance : {confidence:.2f}%")
    print("-" * 30)