import tensorflow as tf
from tensorflow.keras import layers, Model
from src.config import SEQ_LEN, LATENT_DIM

FEATURES = 128

# =========================
# Encoder
# =========================
encoder_inputs = layers.Input(shape=(SEQ_LEN, FEATURES))

x = layers.LSTM(256, return_sequences=True)(encoder_inputs)
x = layers.LSTM(128)(x)

z_mean = layers.Dense(LATENT_DIM)(x)
z_log_var = layers.Dense(LATENT_DIM)(x)


# =========================
# Sampling Layer
# =========================
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs

        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]

        epsilon = tf.random.normal(shape=(batch, dim))

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


z = Sampling()([z_mean, z_log_var])


# =========================
# Decoder
# =========================
decoder_inputs = layers.RepeatVector(SEQ_LEN)(z)

x = layers.LSTM(128, return_sequences=True)(decoder_inputs)
x = layers.LSTM(256, return_sequences=True)(x)

decoder_outputs = layers.TimeDistributed(
    layers.Dense(FEATURES, activation="sigmoid")
)(x)


# =========================
# VAE Model (FIXED)
# =========================
class VAE(Model):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder model
        self.encoder = Model(
            encoder_inputs,
            [z_mean, z_log_var, z]
        )

        # Decoder model
        self.decoder = Model(
            z,
            decoder_outputs
        )

        # Metrics (IMPORTANT for epoch logging)
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.recon_loss_tracker,
            self.kl_loss_tracker
        ]

    # =========================
    # TRAIN STEP (FIXED CORE)
    # =========================
    def train_step(self, data):
        x = data

        with tf.GradientTape() as tape:

            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)

            # Reconstruction loss (stable)
            recon_loss = tf.reduce_mean(tf.square(x - reconstruction))

            # KL divergence
            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var
                - tf.square(z_mean)
                - tf.exp(z_log_var)
            )

            total_loss = recon_loss + 0.001 * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
        }


# =========================
# Build model
# =========================
vae = VAE()

vae.compile(
    optimizer=tf.keras.optimizers.Adam()
)