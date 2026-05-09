from tensorflow.keras import layers, Model, Input
from tensorflow.keras.metrics import Recall, Precision
from src.config import SEQ_LEN, LATENT_DIM

def build_autoencoder():
    # Encoder
    inputs = Input(shape=(SEQ_LEN, 128))
    x = layers.LSTM(256, return_sequences=False)(inputs)
    latent = layers.Dense(LATENT_DIM, activation='relu')(x)

    # Decoder
    x = layers.RepeatVector(SEQ_LEN)(latent)
    x = layers.LSTM(256, return_sequences=True)(x)
    outputs = layers.TimeDistributed(layers.Dense(128, activation='sigmoid'))(x)

    autoencoder = Model(inputs, outputs)
    
    # Adding Accuracy and Recall here
    autoencoder.compile(
        optimizer='adam', 
        loss='binary_crossentropy',
        metrics=['accuracy', Recall(name='recall'), Precision(name='precision')]
    )
    return autoencoder