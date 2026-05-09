import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("outputs", exist_ok=True)

# Load training history
history = np.load("outputs/vae_history.npy", allow_pickle=True).item()

loss = history.get("loss", [])
recon = history.get("reconstruction_loss", loss)  # fallback
kl = history.get("kl_loss", None)

epochs = range(1, len(loss) + 1)

# =========================
# Plot Total Loss
# =========================
plt.figure()
plt.plot(epochs, loss)
plt.title("VAE Total Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("outputs/vae_total_loss.png")
plt.close()

# =========================
# Plot KL (if exists)
# =========================
if kl is not None:
    plt.figure()
    plt.plot(epochs, kl)
    plt.title("KL Divergence Loss")
    plt.xlabel("Epochs")
    plt.ylabel("KL Loss")
    plt.savefig("outputs/vae_kl_loss.png")
    plt.close()

print("✔ Loss curves saved in outputs/")