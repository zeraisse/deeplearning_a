import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, MultiHeadAttention, LayerNormalization, Dropout, Add
from tensorflow.keras.models import Model
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import os
import pickle

# --- OPTIMISATION GPU (RTX 5070 Ti) ---
# Le Mixed Precision permet d'aller 2x plus vite en utilisant moins de VRAM
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# --- CONFIGURATION ---
model_name = "squad_transformer"
maxlen = 128          # On limite la taille pour que ça rentre en mémoire et aille vite
vocab_size = 30522    # Taille vocabulaire BERT
embed_dim = 256       # Dimension des vecteurs (Plus grand = plus intelligent)
num_heads = 4         # Nombre de têtes d'attention
ff_dim = 1024         # Taille couche interne
num_layers = 2        # Nombre de couches (Encoder et Decoder). Tu peux monter à 4 si tu es joueur.
batch_size = 64       # Ta 5070 Ti devrait encaisser 64 ou 128
epochs = 20           # Laisse tourner !

# 1. TOKENIZER & DATASET
print("📥 Chargement du Tokenizer BERT...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

print("📥 Chargement de SQuAD (Complet)...")
ds = load_dataset("rajpurkar/squad", split='train') # Tout le dataset !
val_ds = load_dataset("rajpurkar/squad", split='validation')

# 2. PRÉTRAITEMENT DES DONNÉES
def preprocess_data(dataset, limit=None):
    if limit: dataset = dataset.select(range(limit))
    
    enc_inputs = []
    dec_inputs = []
    dec_outputs = []

    print(f"⚙️ Traitement de {len(dataset)} exemples...")
    
    for item in dataset:
        context = item['context']
        question = item['question']
        answer = item['answers']['text'][0]

        # Format Entrée Encodeur : "[CLS] question [SEP] context [SEP]"
        # On combine pour que le modèle ait tout le contexte
        input_text = f"{question} [SEP] {context}"
        
        # Tokenization
        tokenized_in = tokenizer(input_text, max_length=maxlen, padding='max_length', truncation=True)
        tokenized_ans = tokenizer(answer, max_length=maxlen+1, padding='max_length', truncation=True)['input_ids']
        
        # Préparation Décodeur (Teacher Forcing)
        # Input: [CLS] + Réponse (sans le dernier token)
        # Target: Réponse + [SEP] (décalé d'un cran)
        
        # BERT IDs: 101=[CLS], 102=[SEP], 0=[PAD]
        # On s'assure que la réponse commence par [CLS] (101) pour l'input
        ans_ids = [t for t in tokenized_ans if t != 0] # Enlever padding temporairement
        if ans_ids[0] != 101: ans_ids = [101] + ans_ids
        if ans_ids[-1] != 102: ans_ids = ans_ids + [102]
        
        # On refait le padding manuellement proprement
        curr_len = len(ans_ids)
        if curr_len > maxlen + 1:
            ans_ids = ans_ids[:maxlen+1] # Coupe si trop long
            ans_ids[-1] = 102 # Remet le SEP à la fin
        
        padded_ans = ans_ids + [0] * (maxlen + 1 - len(ans_ids))
        
        dec_in = padded_ans[:-1]  # Enlève le dernier
        dec_out = padded_ans[1:]  # Enlève le premier
        
        enc_inputs.append(tokenized_in['input_ids'])
        dec_inputs.append(dec_in)
        dec_outputs.append(dec_out)

    return np.array(enc_inputs), np.array(dec_inputs), np.array(dec_outputs)

# On prépare tout (Attention, ça peut prendre 1-2 minutes)
x_enc, x_dec, y_dec = preprocess_data(ds)
x_enc_val, x_dec_val, y_dec_val = preprocess_data(val_ds)

# 3. BRIQUES DU MODÈLE (AVEC POSITIONAL EMBEDDING)

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model, max_len):
        super().__init__()
        self.d_model = d_model
        self.embedding = Embedding(vocab_size, d_model)
        self.pos_encoding = Embedding(max_len, d_model)

    def call(self, x):
        length = tf.shape(x)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        # On additionne le sens du mot (embedding) + sa position (pos_encoding)
        return self.embedding(x) + self.pos_encoding(positions)

class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([Dense(ff_dim, "relu"), Dense(embed_dim)])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=True):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TransformerDecoderBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att1 = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.att2 = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([Dense(ff_dim, "relu"), Dense(embed_dim)])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)

    def call(self, inputs, encoder_outputs, training=True):
        # Attention causale (use_causal_mask=True empêche de voir le futur)
        attn1 = self.att1(inputs, inputs, use_causal_mask=True)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(inputs + attn1)
        
        # Cross-Attention
        attn2 = self.att2(out1, encoder_outputs)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)
        
        ffn_out = self.ffn(out2)
        ffn_out = self.dropout3(ffn_out, training=training)
        return self.layernorm3(out2 + ffn_out)

# 4. ASSEMBLAGE
encoder_inputs = Input(shape=(maxlen,), dtype="int32", name="encoder_inputs")
decoder_inputs = Input(shape=(maxlen,), dtype="int32", name="decoder_inputs")

# Embeddings
enc_embedding_layer = PositionalEmbedding(vocab_size, embed_dim, maxlen)
dec_embedding_layer = PositionalEmbedding(vocab_size, embed_dim, maxlen)

x_enc = enc_embedding_layer(encoder_inputs)
x_dec = dec_embedding_layer(decoder_inputs)

# Stack Encoder Layers
for _ in range(num_layers):
    x_enc = TransformerEncoderBlock(embed_dim, num_heads, ff_dim)(x_enc)

# Stack Decoder Layers
for _ in range(num_layers):
    x_dec = TransformerDecoderBlock(embed_dim, num_heads, ff_dim)(x_dec, x_enc)

# Sortie finale
outputs = Dense(vocab_size, activation="softmax", dtype="float32")(x_dec) # float32 pour la stabilité softmax

model = Model([encoder_inputs, decoder_inputs], outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.summary()

# 5. ENTRAÎNEMENT LONGUE DURÉE
print("🔥 Démarrage de l'entraînement intensif...")
checkpoint = tf.keras.callbacks.ModelCheckpoint("squad_model_best.keras", save_best_only=True, monitor="val_loss")
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)

model.fit(
    [x_enc, x_dec], y_dec,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=([x_enc_val, x_dec_val], y_dec_val),
    callbacks=[checkpoint, early_stop]
)

print("✅ Entraînement terminé. Modèle sauvegardé.")