import numpy as np
from src.models.vae import vae

def interpolate(z1, z2, steps=10):
    results = []

    for alpha in np.linspace(0, 1, steps):
        z = (1 - alpha) * z1 + alpha * z2

        output = vae.decoder.predict(z)

        results.append(output[0])

    return results


z1 = np.random.normal(size=(1, 64))
z2 = np.random.normal(size=(1, 64))

interpolated = interpolate(z1, z2)

print("Interpolation generated:", len(interpolated))