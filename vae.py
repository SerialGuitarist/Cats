import torch
from torch import nn
from torch.nn import functional as F
import math

from cats import *

#### device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

class VAENormal(nn.Module):
    def __init__(self, hidden=8, epochs=100, batch_size=16, path="models/cats", name="vae_normal", save_every=10, backup_every=20):
        super().__init__()
        self.num_latent = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.path = path
        self.name = name
        self.checkpoint_path = f"{path}/{name}/{name}_{hidden}.pth"
        self.start_epoch = 0  # track where to resume
        self.save_every = save_every
        self.backup_every = backup_every

        ## input is [b, 3, 64, 64]
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=4, padding=1, stride=2), # [b, 64, 32, 32]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=4, padding=1, stride=2), # [b, 128, 16, 16]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=4, padding=1, stride=2), # [b, 256, 8, 8]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(256*8*8, self.num_latent)
        )

        self.encoded_mu = nn.Linear(self.num_latent, self.num_latent)
        ## log_var instead of var then exponentiating while sampling
        ## ensures that it remains positive and keeps kl divergence simpler
        self.encoded_log_var = nn.Linear(self.num_latent, self.num_latent)

        self.decoder = nn.Sequential(
            nn.Linear(self.num_latent, 256*8*8), # [b, 256*8*8]
            nn.ReLU(),
            nn.Unflatten(1, (256, 8, 8)), # [b, 64, 8, 8]

            nn.ConvTranspose2d(in_channels=256, out_channels=128,
                               kernel_size=4, padding=1, stride=2,
                               output_padding=0), # [b, 128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=128, out_channels=64,
                               kernel_size=4, padding=1, stride=2,
                               output_padding=0), # [b, 64, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=64, out_channels=3,
            # nn.ConvTranspose2d(in_channels=64, out_channels=16,
                               kernel_size=4, padding=1, stride=2,
                               output_padding=0), # [b, 3, 64, 64]
            nn.Sigmoid()
            # nn.ReLU()
        )

        ## using gaussian for p(x|z) with fixed variance
        # self.decoded_mu = nn.Sequential(
            # nn.Conv2d(16, 3, kernel_size=1),
            # nn.Sigmoid()
        # )
        # self.decoded_log_var = nn.Conv2d(16, 3, kernel_size=1)
        if self.name == "vae_normal":
            self.finalize()
    
    ## having this in a seperate method is helpful for subclasses
    def finalize(self):
        # Initialize optimizer and (will be used in fit)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # Try loading checkpoint
        self._load_checkpoint()

    def _load_checkpoint(self, path=None):
        if path == None:
            path = self.checkpoint_path
        if os.path.exists(self.checkpoint_path):
            print(f"Loading checkpoint from {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path)
            self.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.start_epoch = checkpoint["epoch"] + 1
            print(f"Resuming from epoch {self.start_epoch}")
        else:
            print("No checkpoint found. Starting from scratch.")

    def _save_checkpoint(self, epoch, loss, path=None):
        if path == None:
            path = self.checkpoint_path
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
        }, path)
        print(f"Epoch {epoch+1} checkpoint saved at {path}")
        
        
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

        #### 1. psi(x) = mu and sigma for the latent space
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

        ## however torch already provides a very clean and easy
        ## way to do that
        z_std = torch.exp(0.5 * z_log_var).clamp(min=1e-6) # [B, d]
        posterior = torch.distributions.Normal(loc=z_mu, scale=z_std)
        entropy = posterior.entropy().sum(dim=1).mean()


        #### 2. sample z ~ q(z|x)
        ## repameterization shenanigans
        eps = torch.randn(batch_size, n_samples, d, device=x.device) # [B, n_samples, d]
        z_mu_expanded = z_mu.unsqueeze(1) # [B, 1, d]
        z_std_expanded = z_std.unsqueeze(1) # [B, 1, d]
        zs = z_mu_expanded + eps * z_std_expanded # [B, n_samples, d]

        #### 3. theta(z) provides us with the mu and sigma for reconstruction
        zs_flat = zs.view(-1, d) # [B*n_samples, d]

        x_mu = self.decoder(zs_flat) # [B*samples, 3, H, W]
        # decoded = self.decoder(zs_flat) # [B*samples, 16, H, W]
        # x_mu = self.decoded_mu(decoded) # [B*n_samples, 3, H, W]
        # x_log_var = self.decoded_log_var(decoded) # [B*n_samples, 3, H, W]
        # x_std = torch.exp(0.5 * x_log_var).clamp(min=1e-6)

        ## reshape to [B, n_samples, 3, H, W]
        x_mu = x_mu.view(batch_size, n_samples, channels, height, width)  # [B, n_samples, 3, H, W]
        # x_std = x_std.view(batch_size, n_samples, channels, height, width)

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
        # p_x_given_z = torch.distributions.Normal(loc=x_mu, scale=x_std)
        p_x_given_z = torch.distributions.Normal(loc=x_mu, scale=1)
        log_likelihood = p_x_given_z.log_prob(x_expanded) # [B, n_samples, 1, H, W]
        log_likelihood = log_likelihood.view(batch_size, n_samples, -1).sum(dim=2) # [B, n_samples]

        ## we want p(z) to be standard normal
        log_prior = -0.5 * zs.pow(2).sum(dim=2) # [B, n_samples]

        log_joint = log_likelihood + log_prior # [B, n_samples]

        expected_log_joint = log_joint.mean(dim=1).mean() # [1]

        return expected_log_joint + entropy

    def sample(self, n_samples=10, scale=1):
        z = torch.randn(n_samples, self.num_latent).to(self.device) * scale
        # decoded = self.decoder(z)
        # x_mu = self.decoded_mu(decoded)
        # x_log_var = self.decoded_log_var(decoded)
        # x_std = torch.exp(0.5 * x_log_var)
        # eps = torch.randn_like(x_std)
        # return torch.clamp(x_mu + eps * x_std, 0, 1)
        return self.decoder(z)


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
        zs = z_mu + eps * z_std

        ## 3. theta(z) = mu and log var for p(x|z)
        # decoded = self.decoder(z)
        # # x_mu = self.decoded_mu(decoded)
        # x_mu = self.decoded_mu(decoded)
        # x_log_var = self.decoded_log_var(decoded)
# 
        # ## 3. sample from p(x|z)
        # x_std = torch.exp(0.5 * x_log_var)
        # eps = torch.randn_like(x_std)
        # x_recon = torch.clamp(x_mu + eps * x_std, 0, 1)
        x_mu = self.decoder(zs)

        return zs, x_mu

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
                path = f"{self.path}/{self.name}_images/hidden_{self.num_latent}"
                save_images(
                    decoded.detach().cpu(),
                    save_path=f"{path}/regen/epoch_{epoch+1}.png",
                    title=f"Hidden {self.num_latent}, Epoch {epoch+1}"
                )
                samples = self.sample(n_samples=10).detach().cpu()
                save_images(
                    samples,
                    save_path=f"{path}/sample/epoch_{epoch+1}.png",
                    title=f"Sample {self.num_latent}, Epoch {epoch+1}"
                )

            if (epoch+1) % self.backup_every == 0:
                self._save_checkpoint(epoch, epoch_loss, path=self.checkpoint_path[:-4] + f"_epoch_{epoch+1}" + ".pth")

class VAEExponential(VAENormal):
    def __init__(self, hidden=8, epochs=100, batch_size=16, path="models/cats/", name="vae_exponential", save_every=10, backup_every=20):
        super().__init__(
            hidden=hidden,
            epochs=epochs,
            batch_size=batch_size,
            path=path,
            name=name,
            save_every=save_every,
            backup_every=backup_every
        )

        ## input is [b, 3, 64, 64]
        ## encoder output is [b, num_latent]

        self.z_rate = nn.Sequential(
            nn.Linear(self.num_latent, self.num_latent),
            nn.Softplus()
        )
        ## latent representation is now [b, num_latent]

        ## using gaussian for p(x|z) with fixed variance

        self.finalize()

    def elbo(self, x, n_samples=32):
        batch_size, d = x.shape[0], self.num_latent
        channels, height, width = x.shape[1:] # [3, 64, 64]
        x_dim = channels * height * width # total number of pixels per image

        #### 1. psi(x) gives rate for the latent space
        encoded = self.encoder(x) # [B, d]
        z_rate = self.z_rate(encoded).clamp(min=1e-4) # [B, d]

        ## since z ~ indepedent exponentials the entropy has a closed form solution
        ## each exponential entropy is 1-log rate
        ## joint entropy of the d iid exponentials is d times that
        entropy = (1 - torch.log(z_rate)).sum(dim=1).mean()

        #### 2. sample z ~ q(z|x)
        ## repameterization shenanigans
        ## the exponential reparameterization is z=-1/rate log(eps), eps~uniform(0,1)
        eps = torch.rand(batch_size, n_samples, d, device=x.device).clamp(min=1e-6) # [B, n_samples, d]
        z_rate_expanded = z_rate.unsqueeze(1) # [B, 1, d]
        zs = -torch.log(eps) / z_rate_expanded  # [B, n_samples, d]

        #### 3. theta(z) provides us with the mu and sigma for reconstruction
        zs_flat = zs.view(-1, d) # [B*n_samples, d]

        x_mu = self.decoder(zs_flat) # [B*samples, 3, H, W]
        x_mu = x_mu.view(batch_size, n_samples, channels, height, width)  # [B, n_samples, 3, H, W]

        ## x.shape == [B, C, H, W]
        ## gotta add dimensions for the n_samples
        x_expanded = x.unsqueeze(1).expand_as(x_mu) # [B, n_samples, 3, H, W]
        ## now gotta expand it for each of our samples

        #### 4. under the p(x|z) defined by this, we want p(x,z) to be high
        ## p(x,z) = p(x|z)p(z)
        ## log(p(x,z)) = log(p(x|z)) + log(p(z))
        p_x_given_z = torch.distributions.Normal(loc=x_mu, scale=1)
        log_likelihood = p_x_given_z.log_prob(x_expanded) # [B, n_samples, 1, H, W]
        log_likelihood = log_likelihood.view(batch_size, n_samples, -1).sum(dim=2) # [B, n_samples]

        ## we want p(z) to be exponential
        ## exponential pdf is rate exp(-rate z)
        ## log that is log(rate) - rate z
        ## since we want rate = 1, log prior becomes -z for each z
        log_prior = -zs.sum(dim=2) # [B, n_samples]

        log_joint = log_likelihood + log_prior # [B, n_samples]
        expected_log_joint = log_joint.mean(dim=1).mean() # [1]

        return expected_log_joint + entropy

    def sample(self, n_samples=10, prior_rate=1):
        eps = torch.rand(n_samples, self.num_latent, device=self.device).clamp(min=1e-6)
        zs = -torch.log(eps) / prior_rate
        return self.decoder(zs)


    def forward(self, x):
        ## 1. psi(x) = rate for q(z|x)
        encoded = self.encoder(x)
        z_rate = self.z_rate(encoded).clamp(min=1e-4)

        ## 2. sample from q(z|x)
        eps = torch.rand_like(z_rate, device=self.device).clamp(min=1e-6) # [B, d]
        zs = -torch.log(eps) / z_rate # [B, d]

        x_mu = self.decoder(zs) # [B, C, H, W]

        return zs, x_mu

class VAELaplace(VAENormal):
    def __init__(self, hidden=8, epochs=100, batch_size=16, path="models/cats", name="vae_laplace", save_every=10, backup_every=20, temperature=2/3):
        super().__init__(
            hidden=hidden,
            epochs=epochs,
            batch_size=batch_size,
            path=path,
            name=name,
            save_every=save_every,
            backup_every=backup_every
        )

        self.z_mu = nn.Linear(self.num_latent, self.num_latent)
        self.z_b = nn.Linear(self.num_latent, self.num_latent)
        self.finalize()

    def elbo(self, x, n_samples=32):
        batch_size, d = x.shape[0], self.num_latent
        channels, height, width = x.shape[1:] # [3, 64, 64]
        x_dim = channels * height * width # total number of pixels per image

        #### 1. psi(x) gives parameters for the latent space
        encoded = self.encoder(x) # [B, d]
        z_mu = self.z_mu(encoded) # [B, d]
        z_b = self.z_b(encoded).clamp(min=1e-4) # [B, d]

        ## since z ~ indepedent exponentials the entropy has a closed form solution
        ## each laplacian entropy is log(2be) = log(2e) + 1
        ## joint entropy of the d iid exponentials is d times that
        entropy = torch.log(2 * z_b) + 1 # [B, d]
        entropy = entropy.sum(dim=1).mean()

        #### 2. sample z ~ q(z|x)
        ## repameterization shenanigans
        eps = torch.rand(batch_size, n_samples, d, device=x.device) - 0.5
        eps = eps.clamp(min=-0.499, max=0.499)
        z_mu_expanded = z_mu.unsqueeze(1) # [B, 1, d]
        z_b_expanded = z_b.unsqueeze(1) # [B, 1, d]
        zs = z_mu_expanded - z_b_expanded * eps.sign() * torch.log(1-2 * torch.abs(eps)) # [B, n_samples, d]

        #### 3. theta(z) provides us with the mu and sigma for reconstruction
        zs_flat = zs.view(-1, d) # [B*n_samples, d]

        x_mu = self.decoder(zs_flat) # [B*samples, 3, H, W]
        x_mu = x_mu.view(batch_size, n_samples, channels, height, width)  # [B, n_samples, 3, H, W]

        ## x.shape == [B, C, H, W]
        ## gotta add dimensions for the n_samples
        x_expanded = x.unsqueeze(1).expand_as(x_mu) # [B, n_samples, 3, H, W]
        ## now gotta expand it for each of our samples

        #### 4. under the p(x|z) defined by this, we want p(x,z) to be high
        ## p(x,z) = p(x|z)p(z)
        ## log(p(x,z)) = log(p(x|z)) + log(p(z))
        p_x_given_z = torch.distributions.Normal(loc=x_mu, scale=1)
        log_likelihood = p_x_given_z.log_prob(x_expanded) # [B, n_samples, 1, H, W]
        log_likelihood = log_likelihood.view(batch_size, n_samples, -1).sum(dim=2) # [B, n_samples]

        ## we want p(z) to be laplacian
        ## laplacian pdf is 1/2b exp(- (|z-mu|) / b )
        ## log that is -log(2b) - ((|mu-z|) / b )
        ## since we standard laplace: mu=0, b=1
        ## so log p(z) = -log(2) - |z|
        log_prior = - math.log(2) - zs.abs() # [B, n_samples, d]
        log_prior = log_prior.sum(dim=2) # [B, n_samples]

        log_joint = log_likelihood + log_prior # [B, n_samples]
        expected_log_joint = log_joint.mean(dim=1).mean() # [1]

        return expected_log_joint + entropy

    def sample(self, n_samples=10):
        eps = torch.rand(n_samples, self.num_latent, device=self.device) - 0.5
        eps = eps.clamp(min=-0.499, max=0.499)
        zs = -eps.sign() * torch.log(1-2 * torch.abs(eps)) # [n_samples, d]
        return self.decoder(zs)


    def forward(self, x):
        batch_size, d = x.shape[0], self.num_latent
        ## 1. psi(x) = param for q(z|x)
        encoded = self.encoder(x) # [B, d]
        z_mu = self.z_mu(encoded) # [B, d]
        z_b = self.z_b(encoded).clamp(min=1e-4) # [B, d]

        ## 2. sample from q(z|x)
        eps = torch.rand(batch_size, 1, d, device=x.device) - 0.5
        eps = eps.clamp(min=-0.499, max=0.499)
        z_mu_expanded = z_mu.unsqueeze(1) # [B, 1, d]
        z_b_expanded = z_b.unsqueeze(1) # [B, 1, d]
        zs = z_mu_expanded - z_b_expanded * eps.sign() * torch.log(1-2 * torch.abs(eps)) # [B, 1, d]
        zs = zs.squeeze(1) # [B, d]

        x_mu = self.decoder(zs) # [B, C, H, W]

        return zs, x_mu


## training
dataset, loader = load_cats_dataset("data/cats", batch_size=64)
model = VAENormal(hidden=512, epochs=100)
# model = VAELaplace(hidden=64, epochs=100, save_every=5, backup_every=10)
# # model = VAEExponential(hidden=64, epochs=100, save_every=5, backup_every=10)
model.fit(loader)

# samples = model.sample(n_samples=10).detach().cpu()
# save_images(
    # samples,
    # save_path=f"{model.path}/{model.name}_images/hidden_{model.num_latent}/sample/epoch_{model.epochs}.png",
    # title=f"Sample {model.num_latent}, Epoch {model.epochs}"
# )


# models = [
        # VAENormal(hidden=64, epochs=100, save_every=5, backup_every=10),
        # VAEExponential(hidden=64, epochs=100, save_every=5, backup_every=10),
        # VAELaplace(hidden=64, epochs=100, save_every=5, backup_every=10)
# ]



## reconstruction test
# dataset, loader = load_cats_dataset("data/cats", batch_size=64)
# images, labels = next(iter(loader))  ## images: [B, 3, 64, 64]
# testing = images[:10]
# save_images(testing, save_path=f"models/cats/reconstruction/original.png", title=f"Original Images")
# testing = testing.to(models[0].device)
# 
# for model in models:
    # zs, xs = model.forward(testing)
    # xs = xs.detach().cpu()
    # save_images(xs, save_path=f"models/cats/reconstruction/{model.name}.png", title=f"{model.name}")

## mean test
# means = [
    # torch.zeros(models[0].num_latent, device=models[0].device),
    # torch.ones(models[2].num_latent, device=models[2].device),
    # torch.zeros(models[1].num_latent, device=models[1].device),
# ]
# 
# decoded = []
# labels = [model.name for model in models]
# 
# for i in range(3):
    # x = models[i].decoder(means[i].unsqueeze(0))
    # decoded.append(x.detach().cpu())
# 
# xs = torch.cat(decoded, dim=0)
# save_images(xs, save_path=f"data/model_means.png", labels=labels, title=f"Model means")

# ## interpolation test
# def interpolate(z1, z2, n=8, include=True):
    # alphas = torch.linspace(1/(n+1), n/(n+1), n, device=z1.device)
    # interpolations = torch.stack([(1 - alpha) * z1 + alpha * z2 for alpha in alphas])
    # if include:
        # output = torch.cat([z1.unsqueeze(0), interpolations, z2.unsqueeze(0)], dim=0)
    # else:
        # output = interpolations
    # return output
# 
# cats = load_images(["data/cat_a.png", "data/cat_b.png"])
# save_images(cats, save_path=f"data/interpolation/original.png", title=f"Original Images")
# cats = cats.to(models[0].device)
# 
# for model in models:
    # testing_zs, _ = model.forward(cats)
    # testing_zs = testing_zs.detach()
# 
    # model.eval()
    # with torch.no_grad():
        # zs = interpolate(*testing_zs)
        # xs = model.decoder(zs).cpu()
        # save_images(xs, save_path=f"data/interpolation/{model.name}.png", title=f"{model.name}")
