import numpy as np

from src.preprocessing.midi_parser import load_midi_data
from src.models.vae import vae
from src.config import DATA_PATH, EPOCHS, BATCH_SIZE

# Load dataset
X = load_midi_data(DATA_PATH, max_files=100)

print("Dataset shape:", X.shape)

# Stop if dataset failed
if len(X) == 0:
    raise ValueError("Dataset is empty. Check DATA_PATH.")

# Train model
history = vae.fit(
    X,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

import os
import numpy as np

os.makedirs("outputs", exist_ok=True)

np.save("outputs/vae_history.npy", history.history)

print("✔ Training history saved successfully")