import numpy as np
from src.models.transformer import build_transformer

# Load dataset
X = np.load("data/processed/X.npy")

print("Dataset shape:", X.shape)

model = build_transformer()

history = model.fit(
    X, X,
    epochs=10,
    batch_size=32
)

model.save("outputs/transformer_model.keras")

np.save("outputs/train_history.npy", history.history)

print("Training complete")