import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure output directory exists
os.makedirs("outputs/plots", exist_ok=True)


def plot_pitch_histogram(task1, vae, transformer):
    task1_pitch = task1.flatten()
    vae_pitch = vae.flatten()
    transformer_pitch = transformer.flatten()

    plt.figure()

    plt.hist(task1_pitch, bins=50, alpha=0.5, label="Task 1 (AE)")
    plt.hist(vae_pitch, bins=50, alpha=0.5, label="Task 2 (VAE)")
    plt.hist(transformer_pitch, bins=50, alpha=0.5, label="Task 3 (Transformer)")

    plt.title("Pitch Histogram Comparison")
    plt.legend()
    plt.savefig("outputs/plots/pitch_histogram.png")
    plt.show()


def plot_rhythm_distribution(task1, vae, transformer):
    task1_rhythm = [len(np.unique(s)) / len(s.flatten()) for s in task1]
    vae_rhythm = [len(np.unique(s)) / len(s.flatten()) for s in vae]
    transformer_rhythm = [len(np.unique(s)) / len(s.flatten()) for s in transformer]

    plt.figure()

    plt.plot(task1_rhythm, label="Task 1 (AE)")
    plt.plot(vae_rhythm, label="Task 2 (VAE)")
    plt.plot(transformer_rhythm, label="Task 3 (Transformer)")

    plt.title("Rhythm Diversity Comparison")
    plt.legend()
    plt.savefig("outputs/plots/rhythm_diversity.png")
    plt.show()


def plot_loss(history):
    plt.figure()
    plt.plot(history.history["loss"])
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.savefig("outputs/plots/loss_curve.png")
    plt.show()


def plot_piano_roll(sample):
    plt.figure()

    plt.imshow(sample.T, aspect="auto", origin="lower")

    plt.title("Generated Piano Roll")
    plt.xlabel("Time")
    plt.ylabel("Pitch")

    plt.savefig("outputs/plots/piano_roll.png")
    plt.show()


if __name__ == "__main__":

    task1 = np.load("outputs/task1_samples.npy")
    vae = np.load("outputs/generated_vae_samples.npy")
    transformer = np.load("outputs/transformer_generated.npy")

    plot_pitch_histogram(task1, vae, transformer)
    plot_rhythm_distribution(task1, vae, transformer)

    plot_piano_roll(vae[0])  # VAE sample example

    print("✔ Visualization completed for Task 1 + Task 2 + Task 3")