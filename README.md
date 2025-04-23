# Comparison of Latent Space Distributions in a Variational Auto-Encoder using Cat Dataset
A standard autoeoncoder compresses the observation $x \in X$ into a smaller dimensional latent space $Z$ using an encoder $\phi(x)$ and then decompresses it using decoder $\theta(z)$. It's useful for data compression but we can't use it generate synthetic data by sampling random $z$ and running that through the decoder, as the the model pust no real constraints on the distribution so similar observations can be placed far apart in the latent space and vice versa. Thus, the autoencoder is extended to being probabilistic, not deterministically mapping $X \to Z$ and back $Z \to X$ but mapping to a random distribution $Z \sim p(Z|X)$ and $X \sim p(X|Z)$. It is assumed the data $X$ comes from some true distribution $X \sim p(x)$ and we pick an arbitrary distribution that is easy to work with that the latent space exists in, where can easily sample from. Now the challenge is to build the encoder-decoder model such that $theta(X)$ provides the parameters for the distribution that $Z$ is sampled from and $phi(Z)$ provides the parameters for the latent representation to be sampled from and $theta(Z)$ that provides the parameters for the reconstruction to be sampled from. 

# Files
- `cats.py` has utilities for loading the cat dataset and saving batches of images
- `vae.py` has implementation of a Normal, Exponential, and Laplacian latent spaces

# Results
