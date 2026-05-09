import numpy as np
from src.generation.midi_export import piano_roll_to_midi_transformer
from src.models.transformer import model

def generate():
    samples = []

    for i in range(10):
        x = np.zeros((1, 64, 128))

        pred = model.predict(x)[0]
        samples.append(pred)

        path = f"outputs/generated_midis/transformer_{i}.mid"
        piano_roll_to_midi_transformer(pred, path)

        print("Saved:", path)

    np.save("outputs/transformer_generated.npy", samples)

generate()