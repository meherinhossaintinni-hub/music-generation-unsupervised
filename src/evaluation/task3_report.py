import numpy as np
import matplotlib.pyplot as plt

def pitch_histogram(X):
    return X.flatten()

def rhythm_diversity(X):
    return np.mean([len(np.unique(s)) / len(s.flatten()) for s in X])

def repetition_ratio(X):
    flat = X.flatten()
    return len(flat) / (len(np.unique(flat)) + 1e-8)

if __name__ == "__main__":
    X = np.load("outputs/transformer_generated.npy")

    pitch = pitch_histogram(X)
    rhythm = rhythm_diversity(X)
    repetition = repetition_ratio(X)

    print("\n===== TASK 3 METRICS =====")
    print("Rhythm Diversity:", rhythm)
    print("Repetition Ratio:", repetition)

    # Save report
    with open("outputs/task3_metrics.txt", "w") as f:
        f.write(f"Rhythm Diversity: {rhythm}\n")
        f.write(f"Repetition Ratio: {repetition}\n")

    # Plot
    plt.hist(pitch, bins=50)
    plt.title("Task 3 Pitch Distribution")
    plt.savefig("outputs/plots/task3_pitch.png")
    plt.show()
    