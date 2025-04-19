import torch
from torch import nn
from torch.nn import functional as F

from mnist import *

#### device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

#### data shenanigans
data = load_mnist()
## Split into training and testing sets
X_train, y_train, X_test, y_test = data

## Normalize the pixel values
X_train = torch.tensor(X_train.astype(np.float32) / 255.0).to(device)
X_test = torch.tensor(X_test.astype(np.float32) / 255.0).to(device)

## Convert labels to integers
y_train = torch.tensor(y_train.astype(np.int64)).to(device)
y_test = torch.tensor(y_test.astype(np.int64)).to(device)


class AutoEncoder(nn.Module):
    def __init__(self, hidden=8, epochs=100, batch_size=32, checkpoint_path="models/mnist/autoencoder/", save_every=10):
        super().__init__()
        self.num_latent = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path + "autencoder" + str(hidden) + ".pth"
        self.start_epoch = 0  # Track where to resume
        self.save_every = save_every

        self.encoder = nn.Sequential(
                nn.Linear(784, 256),
                nn.ReLU(),
                nn.Linear(256, self.num_latent),
                nn.ReLU()
                )

        self.decoder = nn.Sequential(
                nn.Linear(self.num_latent, 256),
                nn.ReLU(),
                nn.Linear(256, 784),
                nn.Sigmoid(),
                )

        # Initialize optimizer and loss (will be used in fit)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.criteria = nn.MSELoss()

        # Try loading checkpoint
        self._load_checkpoint()

    def forward(self, xb):
        encoded = self.encoder(xb)
        decoded = self.decoder(encoded)
        return encoded, decoded


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

    def fit(self, X):
        train_loader = torch.utils.data.DataLoader(
            X, batch_size=self.batch_size, shuffle=True
        )

        for epoch in range(self.start_epoch, self.epochs):
            total_loss = 0.0
            for batch_idx, data in enumerate(train_loader):
                data = data.view(data.size(0), -1)
                # data = data.to(device)
                encoded, decoded = self(data)
                loss = self.criteria(decoded, data)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * data.size(0)
            epoch_loss = total_loss / len(train_loader.dataset)
            print("Epoch {}/{}: loss={:.4f}".format(epoch + 1, self.epochs, epoch_loss))
            if (epoch+1) % self.save_every == 0:
                self._save_checkpoint(epoch, epoch_loss)
                _, decoded = self(X_train[:10])
                save_images(decoded.detach().numpy(), y_train[:10].numpy(), save_path=f"models/mnist/autoencoder_images/hidden{self.num_latent}_regen{epoch}.png", title=f"Hidden {self.num_latent}, Epoch {epoch+1}")


class AutoEncoderCNN(nn.Module):
    def __init__(self, hidden=8, epochs=100, batch_size=32, checkpoint_path="models/mnist/autoencoderCNN/", save_every=10):
        super().__init__()
        self.num_latent = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path + "autencoderCNN" + str(hidden) + ".pth"
        self.start_epoch = 0  # Track where to resume
        self.save_every = save_every


        ## input is [b, 1, 28, 28]
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=3, padding=1, stride=2), # [b, 16, 14, 14]
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=3, padding=1, stride=2), # [b, 32, 7, 7]
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*7*7, self.num_latent)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.num_latent, 32*7*7), # [b, 32*7*7]
            nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)), # [b, 32, 7, 7]
            nn.ConvTranspose2d(in_channels=32, out_channels=16,
                               kernel_size=3, padding=1, stride=2,
                               output_padding=1), # [b, 16, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=1,
                               kernel_size=3, padding=1, stride=2,
                               output_padding=1), # [b, 1, 28, 28]
            nn.Sigmoid()
        )

        # Initialize optimizer and loss (will be used in fit)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.criteria = nn.MSELoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # Try loading checkpoint
        self._load_checkpoint()



    def forward(self, xb):
        encoded = self.encoder(xb)
        decoded = self.decoder(encoded)
        return encoded, decoded


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

    def fit(self, X):
        train_loader = torch.utils.data.DataLoader(
            X, batch_size=self.batch_size, shuffle=True
        )

        for epoch in range(self.start_epoch, self.epochs):
            self.train()
            total_loss = 0.0
            for batch_idx, data in enumerate(train_loader):
                data = data.to(self.device)
                encoded, decoded = self(data)
                loss = self.criteria(decoded, data)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * data.size(0)
            epoch_loss = total_loss / len(train_loader.dataset)
            print("Epoch {}/{}: loss={:.4f}".format(epoch + 1, self.epochs, epoch_loss))
            if (epoch+1) % self.save_every == 0:
                self._save_checkpoint(epoch, epoch_loss)
                self.eval()
                with torch.no_grad():
                    _, decoded = self(X_train[:10])
                save_images(decoded.detach().cpu().numpy(), y_train[:10].cpu().numpy(), save_path=f"models/mnist/autoencoderCNN_images/hidden{self.num_latent}_regen{epoch}.png", title=f"Hidden {self.num_latent}, Epoch {epoch+1}")

# save_images(X_train[:10].cpu().numpy(), y_train[:10].cpu().numpy(), save_path=f"models/mnist/autoencoderCNN_images/og.png", title="Original Image")
# for i in range(10):
    # print(f"Working on {2**i}")
    # model = AutoEncoderCNN(hidden=2**i, epochs=1, save_every=1)
    # model.fit(X_train)
            

class VAENormal(nn.Module):
    def __init__(self, hidden=8, epochs=100, batch_size=16, checkpoint_path="models/mnist/vae_normal/", save_every=10):
        super().__init__()
        self.num_latent = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path + "vae_normal" + str(hidden) + ".pth"
        self.start_epoch = 0  # track where to resume
        self.save_every = save_every

        ## input is [b, 1, 28, 28]
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=3, padding=1, stride=2), # [b, 16, 14, 14]
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=3, padding=1, stride=2), # [b, 32, 7, 7]
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*7*7, self.num_latent)
        )

        self.encoded_mu = nn.Linear(self.num_latent, self.num_latent)
        ## log_var instead of var then exponentiating while sampling
        ## ensures that it remains positive and keeps kl divergence simpler
        self.encoded_log_var = nn.Linear(self.num_latent, self.num_latent)

        self.decoder = nn.Sequential(
            nn.Linear(self.num_latent, 32*7*7), # [b, 32*7*7]
            nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)), # [b, 32, 7, 7]
            nn.ConvTranspose2d(in_channels=32, out_channels=16,
                               kernel_size=3, padding=1, stride=2,
                               output_padding=1), # [b, 16, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=1,
                               kernel_size=3, padding=1, stride=2,
                               output_padding=1), # [b, 1, 28, 28]
            nn.Sigmoid()
        )
        ## using gaussian for p(x|z) with fixed variance
        ## so the output of encder(z) IS the reconstruction
        # self.decoded_mu = nn.Linear(self.num_latent, self.num_latent)
        # self.decoded_log_var = nn.Linear(self.num_latent, self.num_latent)

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
    def elbo(self, x, n_samples=64):
        batch_size, d = x.shape[0], self.num_latent
        channels, height, width = x.shape[1:] # [1, 28, 28]
        x_dim = channels * height * width # total number of pixels per image

        #### 1. psi(x) gives mu and sigma for the latent space
        ## all 3 below are [b, d]
        encoded = self.encoder(x)
        encoded_mu = self.encoded_mu(encoded)
        encoded_log_var = self.encoded_log_var(encoded)

        ## this already provides us with the entropy bit of ELBO
        ## by maximizing entropy, we're spreading out the latent variables
        ## over as much of the support of the latent distribution
        ## "regularization"
        ## since z ~ indepedent gaussians the entropy has a closed form solution
        ## lifted from wikipedia, H(z) = d/2 log(2 pi e) + 1/2 log det(sigma)
        ## log det sigma = log (product of variances) = sum of log variances
        ## H(z) = 1/2 (d (log(2 pi e)) + sum of log_vars)
        log_2pi_e = torch.log(torch.tensor(2 * torch.pi)) + 1 # [1]
        entropy_elements = encoded_log_var + log_2pi_e # [B, d]
        entropy_per_dim = 0.5 * entropy_elements # [B, d]
        entropy_per_sample = entropy_per_dim.sum(dim=1) # [B]
        entropy = entropy_per_sample.mean() # [1]

        ## look into replacing above with this potentially
        # posterior = torch.distributions.Normal(loc=encoded_mu, scale=torch.exp(0.5 * encoded_log_var))
        # entropy = posterior.entropy().sum(dim=1).mean()


        #### 2. sample z ~ q(z|x)
        ####    psi(x) = mu and sigma
        ## repameterization shenanigans
        eps = torch.randn(batch_size, n_samples, d, device=x.device) # [B, n_samples, d]
        mu = encoded_mu.unsqueeze(1) # [B, 1, d]
        std = torch.exp(0.5 * encoded_log_var).unsqueeze(1) # [B, 1, d]
        zs = mu + eps * std # [B, n_samples, d]

        #### 3. theta(z) provides us with the mu and sigma for reconstruction
        zs_flat = zs.view(-1, d) # [B*n_samples, d]
        decoded_mu = self.decoder(zs_flat) # [B*n_samples, 1, 28, 28]
        decoded_mu = decoded_mu.view(batch_size, n_samples, channels, height, width)  # [B, n_samples, 1, 28, 28]

        ## x.shape == [B, C, H, W]
        ## gotta add dimensions for the n_samples
        x_expanded = x.unsqueeze(1)# [B, 1, C, H, W]
        ## now gotta expand it for each of our samples
        x_expanded = x_expanded.expand(-1, n_samples, -1, -1, -1) # [B, n_samples, C, H, W]

        #### 4. under the p(x|z) defined by this, we want p(x,z) to be high
        ## p(x,z) = p(x|z)p(z)
        ## log(p(x,z)) = log(p(x|z)) + log(p(z))
        ## since we're making p(x|z) gaussian normal:
        ## log(p(x|z)) = -1/2 ( ||x-mu||^2 + d log(2pi) )
        normal = torch.distributions.Normal(decoded_mu, 1.0)
        log_likelihood = normal.log_prob(x_expanded) # [B, n_samples, 1, 28, 28]
        log_likelihood = log_likelihood.view(batch_size, n_samples, -1).sum(dim=2) # [B, n_samples]

        ## we want p(z) to be standard normal
        log_prior = -0.5 * zs.pow(2).sum(dim=2) # [B, n_samples]

        log_joint = log_likelihood + log_prior # [B, n_samples]

        expected_log_joint = log_joint.mean(dim=1).mean() # [1]

        return expected_log_joint + entropy

    def sample(self, n_samples=10):
        z = torch.randn(n_samples, self.num_latent).to(self.device)
        samples = self.decoder(z)
        return samples

    ## some code repetition with elbo but considered
    ## acceptable because they work slightly differently
    ## as this one doesn't have to sample a bunch of zs
    def forward(self, x):
        ## 1. psi(x) = mu and log var
        encoded = self.encoder(x)
        mu = self.encoded_mu(encoded)
        log_var = self.encoded_log_var(encoded)

        ## 2. sample from q(z|x)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std

        ## 3. "sample" from p(x|z)
        x_recon = self.decoder(z)
        return mu, log_var, x_recon

    def fit(self, X):
        train_loader = torch.utils.data.DataLoader(
            X, batch_size=self.batch_size, shuffle=True
        )

        for epoch in range(self.start_epoch, self.epochs):
            self.train()
            total_loss = 0.0

            for batch_idx, data in enumerate(train_loader):
                data = data.to(self.device)

                ## compute negative elbo as loss
                loss = -self.elbo(data)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * data.size(0)

            epoch_loss = total_loss / len(train_loader.dataset)
            print("Epoch {}/{}: loss={:.4f}".format(epoch + 1, self.epochs, epoch_loss))
            if (epoch+1) % self.save_every == 0:
                self._save_checkpoint(epoch, epoch_loss)
                self.eval()
                with torch.no_grad():
                    _, _, decoded = self(X_train[:10])
                save_images(
                    decoded.detach().cpu().numpy(),
                    y_train[:10].cpu().numpy(),
                    save_path=f"models/mnist/vae_normal_images/hidden_{self.num_latent}_regen_{epoch}.png",
                    title=f"Hidden {self.num_latent}, Epoch {epoch+1}"
                )
                save_images(
                    model.sample(n_samples=10).detach().cpu().numpy(),
                    save_path=f"models/mnist/vae_normal_images/hidden_{self.num_latent}_sample_{epoch}.png",
                    title=f"Sample {self.num_latent}, Epoch {epoch+1}"
                )


# save_images(X_train[:10].cpu().numpy(), y_train[:10].cpu().numpy(), save_path=f"models/mnist/autoencoderCNN_images/og.png", title="Original Image")
model = VAENormal(hidden=16, epochs=50, save_every=1)
model.fit(X_train)


