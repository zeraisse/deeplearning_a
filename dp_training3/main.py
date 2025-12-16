import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, Embedding, MultiHeadAttention
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
import pickle # Important pour la sauvegarde

# --- CONFIGURATION ---
FILE_PATH = "dataset/IMDB_Dataset.csv"
embed_dim = 128
num_heads = 4
ff_dim = 512
maxlen = 200
vocab_size = 10000
batch_size = 32

# --- 1. CHARGEMENT & PRÉTRAITEMENT ---
print(f"📂 Chargement de {FILE_PATH}...")

try:
    df = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f"❌ Erreur : Le fichier {FILE_PATH} est introuvable.")
    exit()

def clean_text(text):
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

print("🧹 Nettoyage des textes...")
df['review'] = df['review'].apply(clean_text)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

print("🔢 Tokenization (Transformation en chiffres)...")
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<UNK>")
tokenizer.fit_on_texts(df['review'])

sequences = tokenizer.texts_to_sequences(df['review'])
padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')

X_train, X_test, y_train, y_test = train_test_split(padded_sequences, df['sentiment'].values, test_size=0.2, random_state=42)
print(f"✅ Prêt : {len(X_train)} exemples d'entraînement, {len(X_test)} de test.")

# --- 2. ARCHITECTURE TRANSFORMER (CORRIGÉE POUR SAUVEGARDE) ---
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        # On doit stocker les variables pour le get_config
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
    
    # INDISPENSABLE POUR SAUVEGARDER
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
        # On stocke les variables
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

    # INDISPENSABLE POUR SAUVEGARDER
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

# --- 3. ENTRAÎNEMENT ---
print("🚀 Démarrage de l'entraînement...")
# On ajoute name='transformer' pour éviter des bugs de nommage interne
model = Transformer(vocab_size, embed_dim, num_heads, ff_dim, name='transformer')
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Un petit build pour initialiser les poids avant
model.build(input_shape=(None, maxlen))

model.fit(X_train, y_train, batch_size=batch_size, epochs=10, validation_data=(X_test, y_test))

# --- 4. TEST RAPIDE ---
new_review = "I really wasted my time watching this movie, the plot was terrible."
print(f"\n📝 Test sur : '{new_review}'")
# Prédiction rapide
seq = tokenizer.texts_to_sequences([clean_text(new_review)])
padded = pad_sequences(seq, maxlen=maxlen, padding='post', truncating='post')
pred = model.predict(padded)
label = "POSITIVE" if pred[0][0] > 0.5 else "NEGATIVE"
print(f"Résultat : {label} ({pred[0][0]:.4f})")

# --- 5. SAUVEGARDE (Celle qui plantait) ---
print("💾 Sauvegarde du modèle...")
try:
    model.save("imdb_transformer.keras")
    print("✅ Modèle sauvegardé avec succès !")
except Exception as e:
    print(f"❌ Échec sauvegarde modèle : {e}")

print("💾 Sauvegarde du Tokenizer...")
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("✅ TOUT EST BON ! Tu peux lancer test_model.py maintenant.")