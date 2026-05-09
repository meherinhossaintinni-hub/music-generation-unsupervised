import numpy as np
import tensorflow as tf

def compute_perplexity(model, X):
    logits = model(X, training=False)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    loss = loss_fn(X, logits).numpy()

    perplexity = np.exp(loss)
    return loss, perplexity
loss, ppl = compute_perplexity(model, X)

print("Final loss:", loss)
print("Final perplexity:", ppl)