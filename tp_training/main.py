import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, MultiHeadAttention, LayerNormalization, Dropout, Add
from tensorflow.keras.models import Model
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

# --- CONFIGURATION ---
maxlen_input = 256  # Longueur max (Question + Contexte)
maxlen_answer = 64  # Longueur max de la réponse générée
vocab_size = 30522  # Taille vocabulaire BERT standard
embed_dim = 128     # Dimension des vecteurs
num_heads = 4
ff_dim = 512
batch_size = 16

print("Chargement du Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 2. DATASET (SQuAD)
print("Chargement de SQuAD...")
ds = load_dataset("rajpurkar/squad", split='train[:2000]') # 2000 pour test
val_ds = load_dataset("rajpurkar/squad", split='validation[:200]')


def preprocess_data(dataset):
    enc_inputs = []
    dec_inputs = []
    dec_outputs = []

    for item in dataset:
        context = item['context']
        question = item['question']
        answer = item['answers']['text'][0]

        # Format entrée Encodeur : "question: ... context: ..."
        input_text = f"question: {question} context: {context}"
        
        # Tokenization Encodeur
        tokenized_in = tokenizer(input_text, max_length=maxlen_input, padding='max_length', truncation=True)
        enc_inputs.append(tokenized_in['input_ids'])

        # Tokenization Décodeur
        tokenized_ans = tokenizer(answer, max_length=maxlen_answer-1, padding='max_length', truncation=True)['input_ids']
        
        # Pour le décodeur : 
        # Input = [CLS] + Réponse
        # Target = Réponse + [SEP]
        # TODO gérer ça plus finement
        dec_in = [101] + tokenized_ans[:-1] # Enlève le dernier padding
        dec_out = tokenized_ans # Déjà contient le SEP ou padding
        
        dec_inputs.append(dec_in)
        dec_outputs.append(dec_out)

    return np.array(enc_inputs), np.array(dec_inputs), np.array(dec_outputs)

print("⚙️ Prétraitement des données...")
x_enc, x_dec, y_dec = preprocess_data(ds)
x_enc_val, x_dec_val, y_dec_val = preprocess_data(val_ds)
print(f"Format: {x_enc.shape}, {x_dec.shape}")



# --- COUCHE ENCODEUR (Classique) ---
class TransformerEncoder(tf.keras.layers.Layer):
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

# --- COUCHE DÉCODEUR ---
class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        # 1. Masked Self-Attention
        self.att1 = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        # 2. Cross-Attention
        self.att2 = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        
        self.ffn = tf.keras.Sequential([Dense(ff_dim, "relu"), Dense(embed_dim)])
        
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)

    def call(self, inputs, encoder_outputs, training=True):
        attn1 = self.att1(inputs, inputs, use_causal_mask=True) 
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(inputs + attn1)
        
        # Cross-Attention : Query = Décodeur, Key/Value = Encodeur
        attn2 = self.att2(out1, encoder_outputs) 
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)
        
        ffn_out = self.ffn(out2)
        ffn_out = self.dropout3(ffn_out, training=training)
        return self.layernorm3(out2 + ffn_out)

# --- ASSEMBLAGE DU MODÈLE ---
# Entrées
encoder_inputs = Input(shape=(maxlen_input,), name="Encoder_Input")
decoder_inputs = Input(shape=(maxlen_answer,), name="Decoder_Input")

# Embedding + Positional Encoding
embedding_layer = Embedding(vocab_size, embed_dim)
x_enc = embedding_layer(encoder_inputs)
x_dec = embedding_layer(decoder_inputs)

# Passage dans l'Encodeur
encoder_block = TransformerEncoder(embed_dim, num_heads, ff_dim)
encoder_outputs = encoder_block(x_enc)

# Passage dans le Décodeur
decoder_block = TransformerDecoder(embed_dim, num_heads, ff_dim)
decoder_outputs = decoder_block(x_dec, encoder_outputs)

# Sortie finale
final_layer = Dense(vocab_size, activation="softmax")
outputs = final_layer(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], outputs)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()


print("🚀 Entraînement...")
model.fit(
    [x_enc, x_dec], y_dec, 
    batch_size=batch_size, 
    epochs=5, 
    validation_data=([x_enc_val, x_dec_val], y_dec_val)
)