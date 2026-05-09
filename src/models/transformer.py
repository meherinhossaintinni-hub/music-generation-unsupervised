import tensorflow as tf
from tensorflow.keras import layers, Model

SEQ_LEN = 64
FEATURES = 128
LATENT_DIM = 256


def build_transformer(seq_len=SEQ_LEN, features=FEATURES):
    inputs = layers.Input(shape=(seq_len, features))

    # Embedding projection
    x = layers.Dense(256)(inputs)

    # Transformer block 1
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)

    # Feedforward
    ff = layers.Dense(256, activation="relu")(x)
    ff = layers.Dense(256)(ff)

    x = layers.Add()([x, ff])
    x = layers.LayerNormalization()(x)

    # Output projection
    outputs = layers.Dense(features)(x)

    model = Model(inputs, outputs, name="transformer_music")

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model