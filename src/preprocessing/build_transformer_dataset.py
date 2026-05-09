import os
import numpy as np
import pretty_midi

from src.config import SEQ_LEN, FEATURES

DATA_PATH = "data/clean_midi"
OUTPUT_PATH = "data/processed/train.npy"


# =========================
# MIDI → Piano Roll
# =========================
def midi_to_piano_roll(file_path):
    try:
        midi = pretty_midi.PrettyMIDI(file_path)
        roll = midi.get_piano_roll(fs=10)  # (128, time)

        # transpose → (time, 128)
        roll = roll.T

        # normalize length to SEQ_LEN
        if len(roll) < SEQ_LEN:
            pad = np.zeros((SEQ_LEN - len(roll), FEATURES))
            roll = np.vstack([roll[:SEQ_LEN], pad])
        else:
            roll = roll[:SEQ_LEN]

        return roll.astype(np.float32)

    except Exception as e:
        print(f"Skipping {file_path} -> {e}")
        return None


# =========================
# Dataset Builder
# =========================
def build_dataset():

    sequences = []

    if not os.path.exists(DATA_PATH):
        print("Directory not found:", DATA_PATH)
        return

    midi_files_found = 0
    valid_files = 0

    # recursive search (VERY IMPORTANT)
    for root, _, files in os.walk(DATA_PATH):
        for file in files:

            # robust extension handling
            if file.lower().endswith((".mid", ".midi")):

                midi_files_found += 1
                file_path = os.path.join(root, file)

                print("Processing:", file_path)

                seq = midi_to_piano_roll(file_path)

                if seq is not None:
                    sequences.append(seq)
                    valid_files += 1

                # safety limit (memory control)
                if valid_files >= 100:
                    print("Reached limit of 100 valid files.")
                    break

        if valid_files >= 100:
            break

    sequences = np.array(sequences)

    print("\n======================")
    print("MIDI files found:", midi_files_found)
    print("Valid sequences:", len(sequences))
    print("Final dataset shape:", sequences.shape)
    print("======================\n")

    # ensure folder exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    np.save(OUTPUT_PATH, sequences)
    print("Saved to:", OUTPUT_PATH)


# =========================
# Run
# =========================
if __name__ == "__main__":
    build_dataset()