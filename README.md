# Unsupervised Neural Network for Multi-Genre Music Generation

Course: CSE425 / EEE474 Neural Networks  
Project Type: Deep Learning (Unsupervised Music Generation)  

---

#  Project Overview

This project develops **unsupervised deep learning models for MIDI music generation**.  
It learns musical structure without labels and generates new compositions across multiple genres.

The system implements three models:

- 🎹 Task 1: LSTM Autoencoder  
- 🎼 Task 2: Variational Autoencoder (VAE)  
- 🎧 Task 3: Transformer Music Generator  

---

# Core Idea

The model learns a probability distribution over music sequences and generates new samples:

---

#  Technologies Used

- Python 3.x  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- PrettyMIDI  
- tqdm  


---

#  Task 1 — LSTM Autoencoder

### Objective:
Learn compressed latent representation of MIDI sequences and reconstruct them.



# 🎼 Task 2 — Variational Autoencoder (VAE)

### Objective:
Generate diverse multi-genre music using probabilistic latent space.

### Latent Distribution:
\[
q_\phi(z|X) = \mathcal{N}(\mu(X), \sigma(X))
\]

### Reparameterization Trick:
\[
z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
\]

### Loss Function:
\[
L_{VAE} = L_{recon} + \beta D_{KL}(q(z|X)\|p(z))
\]


#  Task 3 — Transformer Music Generator

### Objective:
Generate long coherent musical sequences.

### Autoregressive Model:
\[
p(X) = \prod_{t=1}^{T} p(x_t | x_{<t})
\]

### Loss Function:
\[
L_{TR} = - \sum_{t=1}^{T} \log p_\theta(x_t | x_{<t})
\]

### Perplexity:
\[
Perplexity = \exp\left(\frac{1}{T} L_{TR}\right)
\]

