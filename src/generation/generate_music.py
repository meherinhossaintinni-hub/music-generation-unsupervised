import numpy as np
import tensorflow as tf
from src.models.vae import vae
from src.config import LATENT_DIM

def generate_sample():
    z = np.random.normal(size=(1, LATENT_DIM)).astype(np.float32)

    output = vae.decoder(z, training=False)   # IMPORTANT: direct call (no .predict)

    return output.numpy()[0]


samples = []

for i in range(8):
    music = generate_sample()
    samples.append(music)
    print(f"Generated sample {i+1}: shape =", music.shape)

samples = np.array(samples)

import os
os.makedirs("outputs", exist_ok=True)

np.save("outputs/generated_vae_samples.npy", samples)

print("✔ Done: 8 VAE samples saved")