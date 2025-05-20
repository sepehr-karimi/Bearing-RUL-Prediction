# src/rul_estimation/models/transformer.py
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    LayerNormalization,
    Dropout,
    Input,
    Layer,
)
from tensorflow.keras.models import Model


class PositionalEncoding(Layer):
    def call(self, x):
        # dummy position encoding (learnable offset)
        return x + self.add_weight("pos_enc", shape=x.shape[1:], initializer="zeros")


def build_model(cfg):
    d_model = cfg.params.get("d_model", 32)
    n_heads = cfg.params.get("n_heads", 2)
    ff_dim = cfg.params.get("ff_dim", 64)
    seq_len = cfg.params.get("seq_len", 50)

    inputs = Input(shape=(seq_len, 1))
    x = PositionalEncoding()(inputs)
    x = tf.keras.layers.MultiHeadAttention(
        num_heads=n_heads, key_dim=d_model
    )(x, x, x)
    x = Dropout(0.1)(x)
    x = LayerNormalization()(x)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dense(1)(x[:, -1, :])  # use last token
    model = Model(inputs, x)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model
