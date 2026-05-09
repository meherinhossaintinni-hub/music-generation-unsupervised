import numpy as np

def random_baseline(X):
    noise = np.random.random(X.shape)
    loss = np.mean((X - noise) ** 2)
    return np.exp(loss)

def markov_baseline(X):
    # simple fake Markov approximation
    shifted = np.roll(X, 1, axis=1)
    loss = np.mean((X - shifted) ** 2)
    return np.exp(loss)

if __name__ == "__main__":
    X = np.load("data/processed/train.npy")

    print("Random baseline perplexity:", random_baseline(X))
    print("Markov baseline perplexity:", markov_baseline(X))