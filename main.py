import os
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
from src.preprocessing.midi_parser import load_midi_data
from src.models.autoencoder import build_autoencoder
from src.config import EPOCHS, BATCH_SIZE

def main():
    # 1. Setup Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, 'data', 'raw_midi')
    output_dir = os.path.join(BASE_DIR, 'outputs')
    
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # 2. Load Data (Limited to 100 files for memory efficiency)
    print("--- Phase 1: Data Preparation ---")
    X_train = load_midi_data(data_path, max_files=100)
    
    if X_train is None or len(X_train) == 0:
        print("Error: No data found. Check your MIDI folder.")
        return

    # 3. Build Model
    print("--- Phase 2: Model Architecture ---")
    model = build_autoencoder()
    model.summary()

    # 4. Training (Unsupervised: Input = Target)
    print("--- Phase 3: Unsupervised Training ---")
    history = model.fit(
        X_train, 
        X_train, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        validation_split=0.1
    )

    # 5. Save Model
    model.save(os.path.join(output_dir, 'task1_autoencoder.h5'))
    print("Model saved to outputs/task1_autoencoder.h5")

    # 6. Save Metrics Plot
    print("--- Phase 4: Evaluation ---")
    plt.figure(figsize=(15, 5))

    # Accuracy Plot
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()

    # Recall Plot (Crucial for music notes)
    plt.subplot(1, 3, 2)
    plt.plot(history.history['recall'], label='Train Recall')
    plt.plot(history.history['val_recall'], label='Val Recall')
    plt.title('Recall')
    plt.legend()

    # Loss Plot
    plt.subplot(1, 3, 3)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Binary Crossentropy Loss')
    plt.legend()

    plot_path = os.path.join(output_dir, 'task1_metrics.png')
    plt.savefig(plot_path)
    print(f"Metrics visualization saved to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    main()