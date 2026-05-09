import pretty_midi
import numpy as np
import os
from tqdm import tqdm
from src.config import FS, SEQ_LEN

def load_midi_data(folder_path, max_files=1000): # Added limit here
    all_sequences = []
    files_processed = 0
    
    if not os.path.isdir(folder_path):
        print(f"Directory not found: {folder_path}")
        return np.array([])

    for root, dirs, files in os.walk(folder_path):
        midi_files = [f for f in files if f.lower().endswith(('.mid', '.midi'))]
        
        for file in midi_files:
            if files_processed >= max_files:
                print(f"Reached limit of {max_files} files to save memory.")
                return np.array(all_sequences)

            try:
                file_path = os.path.join(root, file)
                pm = pretty_midi.PrettyMIDI(file_path)
                piano_roll = pm.get_piano_roll(fs=FS)
                
                # Using float16 instead of float32 saves 50% more memory
                piano_roll = (piano_roll > 0).astype(np.float16).T
                
                for i in range(0, len(piano_roll) - SEQ_LEN, SEQ_LEN):
                    chunk = piano_roll[i : i + SEQ_LEN]
                    if chunk.shape == (SEQ_LEN, 128):
                        all_sequences.append(chunk)
                
                files_processed += 1
                # Progress update every 10 files
                if files_processed % 10 == 0:
                    print(f"Processed {files_processed}/{max_files} files...")
                    
            except Exception:
                continue
            
    return np.array(all_sequences)