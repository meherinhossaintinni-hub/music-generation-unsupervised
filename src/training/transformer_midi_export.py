import numpy as np
import pretty_midi
import os

INPUT_PATH = "outputs/transformer_generated.npy"
OUTPUT_DIR = "outputs/generated_midis/task3/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def piano_roll_to_midi(piano_roll, file_path, fs=16):
    """
    piano_roll shape: (time_steps, 128)
    """

    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    time_steps, pitches = piano_roll.shape

    # safety clamp (prevents out-of-bounds issues)
    time_steps = min(time_steps, piano_roll.shape[0])

    for pitch in range(128):
        start = None

        for t in range(time_steps):
            value = piano_roll[t, pitch]

            if value > 0.5 and start is None:
                start = t

            elif value <= 0.5 and start is not None:
                end = t
                note = pretty_midi.Note(
                    velocity=100,
                    pitch=pitch,
                    start=start / fs,
                    end=end / fs
                )
                instrument.notes.append(note)
                start = None

        # close note if still active
        if start is not None:
            note = pretty_midi.Note(
                velocity=100,
                pitch=pitch,
                start=start / fs,
                end=time_steps / fs
            )
            instrument.notes.append(note)

    midi.instruments.append(instrument)
    midi.write(file_path)


def main():
    samples = np.load(INPUT_PATH)

    print("Loaded samples:", samples.shape)

    for i in range(len(samples)):
        file_path = os.path.join(OUTPUT_DIR, f"transformer_{i+1}.mid")
        piano_roll_to_midi(samples[i], file_path)
        print(f"Saved: {file_path}")


if __name__ == "__main__":
    main()