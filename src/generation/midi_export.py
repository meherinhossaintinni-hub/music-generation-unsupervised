import numpy as np
import pretty_midi
import os

def piano_roll_to_midi_transformer(piano_roll, output_path, fs=8):
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    piano_roll = piano_roll.T  # (time, pitch)

    for pitch in range(piano_roll.shape[1]):
        active = False
        start = 0

        for t in range(len(piano_roll)):
            if piano_roll[t][pitch] > 0.5 and not active:
                active = True
                start = t

            elif piano_roll[t][pitch] <= 0.5 and active:
                active = False
                note = pretty_midi.Note(
                    velocity=100,
                    pitch=pitch,
                    start=start / fs,
                    end=t / fs
                )
                instrument.notes.append(note)

    pm.instruments.append(instrument)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pm.write(output_path)