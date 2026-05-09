import numpy as np
import tensorflow as tf

from src.config import SEQ_LEN, FEATURES

# =========================
# Load trained Transformer
# =========================
model = tf.keras.models.load_model(
    "outputs/transformer_model.h5",
    compile=False
)

# =========================
# Sampling function
# =========================
def sample_next(pred):
    """
    Greedy + slight randomness for diversity
    """
    pred = pred[0]  # (FEATURES,)
    
    # add small noise for creativity
    pred = pred + np.random.normal(0, 0.01, size=pred.shape)

    return np.clip(pred, 0, 1)


# =========================
# Generate one sequence
# =========================
def generate_sequence(seed=None, length=200):

    if seed is None:
        seq = np.zeros((1, SEQ_LEN, FEATURES))
    else:
        seq = seed.reshape(1, SEQ_LEN, FEATURES)

    output_sequence = []

    for _ in range(length):

        # predict next step
        pred = model.predict(seq, verbose=0)

        next_step = sample_next(pred[:, -1, :])

        output_sequence.append(next_step)

        # shift window (autoregressive)
        seq = np.roll(seq, shift=-1, axis=1)
        seq[0, -1, :] = next_step

    return np.array(output_sequence)


# =========================
# Generate 10 samples
# =========================
samples = []

for i in range(10):
    print(f"Generating sample {i+1}...")

    music = generate_sequence(length=300)

    samples.append(music)

    print(f"Shape: {music.shape}")


samples = np.array(samples)

np.save("outputs/transformer_generated.npy", samples)

print("\n✔ Done: 10 Transformer sequences saved!")
model.save("outputs/transformer_model.h5")