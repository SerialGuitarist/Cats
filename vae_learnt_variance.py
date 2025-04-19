import torch
from torch import nn
from torch.nn import functional as F

from cats import *

#### device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

class VAENormal(nn.Module):
    def __init__(self, hidden=8, epochs=100, batch_size=16, checkpoint_path="models/cats/vae_normal/", save_every=10):
        super().__init__()
        self.num_latent = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path + "vae_normal" + str(hidden) + ".pth"
        self.start_epoch = 0  # track where to resume
        self.save_every = save_every

        ## input is [b, 3, 64, 64]
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16,
                      kernel_size=3, padding=1, stride=2), # [b, 16, 32, 32]
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=3, padding=1, stride=2), # [b, 32, 16, 16]
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, padding=1, stride=2), # [b, 64, 8, 8]
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*8*8, self.num_latent)
        )

        self.encoded_mu = nn.Linear(self.num_latent, self.num_latent)
        ## log_var instead of var then exponentiating while sampling
        ## ensures that it remains positive and keeps kl divergence simpler
        self.encoded_log_var = nn.Linear(self.num_latent, self.num_latent)

        self.decoder = nn.Sequential(
            nn.Linear(self.num_latent, 64*8*8), # [b, 64*8*8]
            nn.ReLU(),
            nn.Unflatten(1, (64, 8, 8)), # [b, 64, 8, 8]
            nn.ConvTranspose2d(in_channels=64, out_channels=32,
                               kernel_size=3, padding=1, stride=2,
                               output_padding=1), # [b, 32, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16,
                               kernel_size=3, padding=1, stride=2,
                               output_padding=1), # [b, 16, 32, 32]
            nn.ReLU(),
            # nn.ConvTranspose2d(in_channels=16, out_channels=3,
            nn.ConvTranspose2d(in_channels=16, out_channels=16,
                               kernel_size=3, padding=1, stride=2,
                               output_padding=1), # [b, 3, 64, 64]
            # nn.Sigmoid()
            nn.ReLU()
        )
        ## using gaussian for p(x|z) with learnt variance
        self.decoded_mu = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=1),
            nn.Sigmoid()
        )
        self.decoded_log_var = nn.Conv2d(16, 3, kernel_size=1)

        # Initialize optimizer and (will be used in fit)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # Try loading checkpoint
        self._load_checkpoint()

    def _load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            print(f"Loading checkpoint from {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path)
            self.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.start_epoch = checkpoint["epoch"] + 1
            print(f"Resuming from epoch {self.start_epoch}")
        else:
            print("No checkpoint found. Starting from scratch.")

    def _save_checkpoint(self, epoch, loss):
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
        }, self.checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch}")
        
        
    ## we're maximizing p(x) = integrate p(x|z)p(z) over z
    ## but that's intractable so instead, the ELBO provides
    ## a lower bound for it instead:
    ## ELBO = E_q [ log( p(x,z) / q(w) ) ]
    ##      = E_q [ log p(x|z) + log p(z) ] - E_q [ log q(z|x) ]
    ##      = expected (reconstruction prob + prior prob) + entropy
    ##      ≈ mean of some samples(reconstruction prob + prior prob) + entropy
    def elbo(self, x, n_samples=32):
        batch_size, d = x.shape[0], self.num_latent
        channels, height, width = x.shape[1:] # [3, 64, 64]
        x_dim = channels * height * width # total number of pixels per image

        #### 1. psi(x) gives mu and sigma for the latent space
        ## all 3 below are [b, d]
        encoded = self.encoder(x)
        z_mu = self.encoded_mu(encoded) # [B, d]
        z_log_var = self.encoded_log_var(encoded) # [B, d]

        ## this already provides us with the entropy bit of ELBO
        ## by maximizing entropy, we're spreading out the latent variables
        ## over as much of the support of the latent distribution
        ## "regularization"
        ## since z ~ indepedent gaussians the entropy has a closed form solution
        ## lifted from wikipedia, H(z) = d/2 log(2 pi e) + 1/2 log det(sigma)
        ## log det sigma = log (product of variances) = sum of log variances
        ## H(z) = 1/2 (d (log(2 pi e)) + sum of log_vars)
        # log_2pi_e = torch.log(torch.tensor(2 * torch.pi)) + 1 # [1]
        # entropy_elements = encoded_log_var + log_2pi_e # [B, d]
        # entropy_per_dim = 0.5 * entropy_elements # [B, d]
        # entropy_per_sample = entropy_per_dim.sum(dim=1) # [B]
        # entropy = entropy_per_sample.mean() # [1]

        ## look into replacing above with this potentially
        z_std = torch.exp(0.5 * z_log_var).clamp(min=1e-6) # [B, d]
        posterior = torch.distributions.Normal(loc=z_mu, scale=z_std)
        entropy = posterior.entropy().sum(dim=1).mean()


        #### 2. sample z ~ q(z|x)
        ####    psi(x) = mu and sigma
        ## repameterization shenanigans
        eps = torch.randn(batch_size, n_samples, d, device=x.device) # [B, n_samples, d]
        z_mu_expanded = z_mu.unsqueeze(1) # [B, 1, d]
        z_std_expanded = z_std.unsqueeze(1) # [B, 1, d]
        zs = z_mu_expanded + eps * z_std_expanded # [B, n_samples, d]

        #### 3. theta(z) provides us with the mu and sigma for reconstruction
        zs_flat = zs.view(-1, d) # [B*n_samples, d]

        decoded = self.decoder(zs_flat) # [B*samples, 16, H, W]
        x_mu = self.decoded_mu(decoded) # [B*n_samples, 3, H, W]
        x_log_var = self.decoded_log_var(decoded) # [B*n_samples, 3, H, W]
        x_std = torch.exp(0.5 * x_log_var).clamp(min=1e-6)

        ## reshape to [B, n_samples, 3, H, W]
        x_mu = x_mu.view(batch_size, n_samples, channels, height, width)  # [B, n_samples, 3, H, W]
        x_std = x_std.view(batch_size, n_samples, channels, height, width)

        ## x.shape == [B, C, H, W]
        ## gotta add dimensions for the n_samples
        x_expanded = x.unsqueeze(1).expand_as(x_mu) # [B, n_samples, 3, H, W]
        ## now gotta expand it for each of our samples
        # x_expanded = x_expanded.expand(-1, n_samples, -1, -1, -1) # [B, n_samples, C, H, W]

        #### 4. under the p(x|z) defined by this, we want p(x,z) to be high
        ## p(x,z) = p(x|z)p(z)
        ## log(p(x,z)) = log(p(x|z)) + log(p(z))
        ## since we're making p(x|z) gaussian normal:
        ## log(p(x|z)) = -1/2 ( ||x-mu||^2 + d log(2pi) )
        p_x_given_z = torch.distributions.Normal(loc=x_mu, scale=x_std)
        log_likelihood = p_x_given_z.log_prob(x_expanded) # [B, n_samples, 1, H, W]
        log_likelihood = log_likelihood.view(batch_size, n_samples, -1).sum(dim=2) # [B, n_samples]

        ## we want p(z) to be standard normal
        log_prior = -0.5 * zs.pow(2).sum(dim=2) # [B, n_samples]

        log_joint = log_likelihood + log_prior # [B, n_samples]

        expected_log_joint = log_joint.mean(dim=1).mean() # [1]

        return expected_log_joint + entropy

    def sample(self, n_samples=10):
        z = torch.randn(n_samples, self.num_latent).to(self.device)
        decoded = self.decoder(z)
        x_mu = self.decoded_mu(decoded)
        x_log_var = self.decoded_log_var(decoded)
        x_std = torch.exp(0.5 * x_log_var)
        eps = torch.randn_like(x_std)
        return torch.clamp(x_mu + eps * x_std, 0, 1)


    ## some code repetition with elbo but considered
    ## acceptable because they work slightly differently
    ## as this one doesn't have to sample a bunch of zs
    def forward(self, x):
        ## 1. psi(x) = mu and log var for q(z|x)
        encoded = self.encoder(x)
        z_mu = self.encoded_mu(encoded)
        z_log_var = self.encoded_log_var(encoded)

        ## 2. sample from q(z|x)
        z_std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(z_std)
        z = z_mu + eps * z_std

        ## 3. theta(z) = mu and log var for p(x|z)
        decoded = self.decoder(z)
        # x_mu = self.decoded_mu(decoded)
        x_mu = self.decoded_mu(decoded)
        x_log_var = self.decoded_log_var(decoded)

        ## 3. sample from p(x|z)
        x_std = torch.exp(0.5 * x_log_var)
        eps = torch.randn_like(x_std)
        x_recon = torch.clamp(x_mu + eps * x_std, 0, 1)

        return z, x_recon

    def fit(self, loader):
        fixed_images, _ = next(iter(loader))
        fixed_images = fixed_images[:10].to(self.device)
        for epoch in range(self.start_epoch, self.epochs):
            self.train()
            total_loss = 0.0

            for batch_idx, (data, labels) in enumerate(loader):
                data = data.to(self.device) # [B, 3, 64, 64]

                ## compute negative elbo as loss
                loss = -self.elbo(data)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * data.size(0)

            epoch_loss = total_loss / len(loader.dataset)
            print("Epoch {}/{}: loss={:.4f}".format(epoch + 1, self.epochs, epoch_loss))
            if (epoch+1) % self.save_every == 0:
                self._save_checkpoint(epoch, epoch_loss)
                self.eval()
                with torch.no_grad():
                    _, decoded = self(fixed_images)
                save_images(
                    decoded.detach().cpu(),
                    save_path=f"models/cats/vae_normal_images/hidden_{self.num_latent}/regen/epoch_{epoch+1}.png",
                    title=f"Hidden {self.num_latent}, Epoch {epoch+1}"
                )
                samples = self.sample(n_samples=10).detach().cpu()
                save_images(
                    samples,
                    save_path=f"models/cats/vae_normal_images/hidden_{self.num_latent}/sample/epoch_{epoch+1}.png",
                    title=f"Sample {self.num_latent}, Epoch {epoch+1}"
                )


dataset, loader = load_cats_dataset("data/cats", batch_size=64)
model = VAENormal(hidden=64, epochs=300, save_every=1)
model.fit(loader)


