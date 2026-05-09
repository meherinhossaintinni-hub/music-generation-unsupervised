import numpy as np
import os

os.makedirs("outputs", exist_ok=True)

# dummy placeholder OR your AE output if available
task1_samples = np.random.rand(8, 64, 128)

np.save("outputs/task1_samples.npy", task1_samples)

print("✔ Task 1 samples saved")