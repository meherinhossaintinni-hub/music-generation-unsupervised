import numpy as np
import tensorflow as tf
from src.models.transformer import build_transformer

# SAFE LOADING (fallbacks included)
paths = [
    "data/processed/X.npy",
    "data/processed/task3.npy",
    "outputs/task1_samples.npy"
]

X = None
for p in paths:
    try:
        X = np.load(p)
        print(f"Loaded dataset from: {p}")
        break
    except:
        continue

if X is None:
    raise FileNotFoundError("No valid dataset found for perplexity evaluation")

model = build_transformer()

model.compile(optimizer="adam", loss="mse")

loss = model.evaluate(X, X, verbose=0)

perplexity = np.exp(loss)

print("Loss:", loss)
print("Perplexity:", perplexity)

with open("outputs/perplexity_report.txt", "w") as f:
    f.write(f"Loss: {loss}\nPerplexity: {perplexity}\n")

print("Saved report")