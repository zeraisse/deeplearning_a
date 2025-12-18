import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class TinyBlock(layers.Layer):
    def __init__(self, d):
        super().__init__()
        self.ln = layers.LayerNormalization()
        self.fc1 = layers.Dense(4 * d, activation="gelu")
        self.fc2 = layers.Dense(d)

    def call(self, u):
        h = self.ln(u)
        h = self.fc1(h)
        h = self.fc2(h)
        return u + h




class TRM(keras.Model):
    def __init__(
        self,
        vocab_size,
        d=128,
        max_len=16,
        n_rec=6,
        T=3,
        Nsup=8,
    ):
        super().__init__()

        self.d = d
        self.n_rec = n_rec
        self.T = T
        self.Nsup = Nsup

        # embeddings
        self.emb = layers.Embedding(vocab_size, d)
        self.pos = self.add_weight(
            shape=(1, max_len, d),
            initializer="random_normal",
            trainable=True,
        )

        # états initiaux appris
        self.y0 = self.add_weight(
            shape=(1, max_len, d),
            initializer="zeros",
            trainable=True,
        )
        self.z0 = self.add_weight(
            shape=(1, max_len, d),
            initializer="zeros",
            trainable=True,
        )

        # tiny network partagé partout
        self.block1 = TinyBlock(d)
        self.block2 = TinyBlock(d)

        # têtes de sortie
        self.to_vocab = layers.Dense(vocab_size)
        self.halt_head = layers.Dense(1)

  
    def tiny_net(self, u):
        u = self.block1(u)
        u = self.block2(u)
        return u

 
    def update_z(self, x, y, z):
        u = x + y + z
        return self.tiny_net(u)


    def update_y(self, y, z):
        u = y + z
        return self.tiny_net(u)


    def call(self, x_tokens, y_true=None, training=False):
        B = tf.shape(x_tokens)[0]
        L = tf.shape(x_tokens)[1]

        x = self.emb(x_tokens) + self.pos[:, :L, :]

        y = tf.tile(self.y0[:, :L, :], [B, 1, 1])
        z = tf.tile(self.z0[:, :L, :], [B, 1, 1])

        losses = []

        for step in range(self.Nsup):

            for t in range(self.T):

                if t < self.T - 1:
                    # passes sans gradient
                    for _ in range(self.n_rec):
                        z = tf.stop_gradient(self.update_z(x, y, z))
                    y = tf.stop_gradient(self.update_y(y, z))

                else:
                    # passe avec gradient
                    for _ in range(self.n_rec):
                        z = self.update_z(x, y, z)
                    y = self.update_y(y, z)

            logits = self.to_vocab(y)
            halt_p = tf.sigmoid(
                tf.reduce_mean(self.halt_head(y), axis=1)
            )

            if y_true is not None:
                ce = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(
                        y_true, logits, from_logits=True
                    )
                )

                pred = tf.argmax(logits, axis=-1)
                correct = tf.reduce_all(
                    tf.equal(pred, y_true), axis=1
                )
                correct = tf.cast(correct, tf.float32)
                correct = tf.expand_dims(correct, axis=1)

                bce = tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(correct, halt_p)
                )

                losses.append(ce + 0.5 * bce)

            y = tf.stop_gradient(y)
            z = tf.stop_gradient(z)

        if y_true is None:
            return logits, halt_p

        return tf.add_n(losses) / len(losses)