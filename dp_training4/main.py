import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, Embedding, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from datasets import load_dataset
import numpy as np

# --- CONFIGURATION ---
embed_dim = 128      # Réduit un peu pour que ça aille plus vite (512 c'est gros)
num_heads = 4        # 4 têtes suffisent pour ce problème
ff_dim = 512         # Taille couche interne
maxlen = 100         # Les phrases "Emotion" sont courtes (tweets)
vocab_size = 10000
batch_size = 32
num_classes = 6      # <--- IMPORTANT : 6 Émotions

# --- 1. CHARGEMENT DES DONNÉES ---
print("📥 Chargement du dataset...")
ds = load_dataset("dair-ai/emotion", "split")

# Extraction
raw_train_text = ds['train']['text']
y_train = np.array(ds['train']['label'])

raw_test_text = ds['test']['text']
y_test = np.array(ds['test']['label'])

raw_val_text = ds['validation']['text']
y_val = np.array(ds['validation']['label'])

# --- 2. PREPROCESSING ---
print("🔢 Tokenization...")
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<UNK>")
tokenizer.fit_on_texts(raw_train_text)

x_train = pad_sequences(tokenizer.texts_to_sequences(raw_train_text), maxlen=maxlen, padding='post', truncating='post')
x_test = pad_sequences(tokenizer.texts_to_sequences(raw_test_text), maxlen=maxlen, padding='post', truncating='post')
x_val = pad_sequences(tokenizer.texts_to_sequences(raw_val_text), maxlen=maxlen, padding='post', truncating='post')

# --- 3. ARCHITECTURE DU MODÈLE ---

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        # On sauvegarde les paramètres pour la sauvegarde du modèle
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
    
    # Indispensable pour model.save()
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
        
        # CORRECTION 1 : Pooling Global
        self.pool = GlobalAveragePooling1D()
        
        # CORRECTION 2 : 6 sorties + Softmax
        self.final_layer = Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = x + positions
        x = self.trans_block(x, training=training)
        
        # On utilise le pooling au lieu de prendre le dernier token
        x = self.pool(x)
        
        return self.final_layer(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "num_classes": self.num_classes, # On n'oublie pas num_classes
            "rate": self.rate,
        })
        return config

# --- 4. ENTRAÎNEMENT ---
print("🚀 Démarrage de l'entraînement...")

# Initialisation
model = Transformer(vocab_size, embed_dim, num_heads, ff_dim, num_classes)

# Compilation (CORRECTION 3 : sparse_categorical_crossentropy)
model.compile(optimizer="adam", 
              loss="sparse_categorical_crossentropy", 
              metrics=["accuracy"])

# Train
history = model.fit(
    x_train, y_train, 
    batch_size=batch_size, 
    epochs=20,
    validation_data=(x_val, y_val)
)

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.title("Courbes d'entraînement")
plt.xlabel("Époque")
plt.ylabel("Valeur")
plt.grid(True)
plt.show()

# Sauvegarde au format moderne .keras
print("💾 Sauvegarde...")
model.save("transformer_emotion.keras")
print("✅ Terminé !")