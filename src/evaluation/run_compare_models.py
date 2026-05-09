import numpy as np
from src.evaluation.metrics import compare_models
import os

os.makedirs("outputs", exist_ok=True)

task1 = np.load("outputs/task1_samples.npy")
vae = np.load("outputs/generated_vae_samples.npy")

compare_models(task1, vae)