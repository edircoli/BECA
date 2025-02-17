import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm

# ---------- PRIOR CLASSES DEFINITIONS ---------- #

class NormalPrior(nn.Module):
    def __init__(self, d_z):
        """
        Define a Gaussian Normal prior distribution with zero mean and unit variance (Standard distribution).

        Parameters:
            d_z: [int]
                Dimension of the latent space
        """
        super().__init__()
        self.d_z = d_z
        self.mu = nn.Parameter(torch.zeros(self.d_z), requires_grad = False)
        self.var = nn.Parameter(torch.ones(self.d_z), requires_grad = False)

    def forward(self):
        """
        Return prior distribution. This allows the computation of KL-divergence by calling self.prior() in the VAE class.

        Returns:
            prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)
    
class MoGPrior(nn.Module):
    def __init__(self, d_z, multiplier = 1.0):
        """
        Define a Mixture of Gaussian Normal prior distribution.

        Parameters:
            d_z: [int]
                Dimension of the latent space
            multiplier: [float]
                Parameter that controls sparsity of each Gaussian component
        """
        super().__init__()
        self.d_z = d_z
    
    def forward(self):
        """
        Return prior distribution. Allows computation of KL-divergence by calling self.prior() in the VAE class.

        Return:
            prior: [torch.distributions.Distribution]
        """
        return None

# ---------- ENCODER CLASSES DEFINITIONS ---------- #

class NormalEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian Normal encoder to obtain the parameters of the Normal distribution.

        Parameters:
            encoder_net: [torch.nn.Module]
                The encoder network, takes a tensor of dimension (batch, features) and
                outputs a tensor of dimension (batch, 2*d_z), where d_z is the dimension of the 
                latent space.
        """
        super().__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Computes the Gaussian Normal distribution over the latent space.

        Parameters:
            x: [torch.Tensor]
        """
        mu, var = torch.chunk(self.encoder_net(x), 2, dim=-1) #chunk is used for separating the encoder output (batch, 2*d_z) into two separate vectors (batch, d_z)
        return td.Independent(td.Normal(loc = mu, scale = torch.exp(var * 0.5)), 1)
    
# ---------- DECODER CLASSES DEFINITIONS ---------- #

class NegativeBinomialDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Negative Binomial decoder to obtain the parameters of the Negative Binomial distribution.

        Parameters:
            decoder_net: [torch.nn.Module]
                The decoder network, takes a tensor of dimension (batch, d_z) and outputs
                a tensor of dimension (batch, 2*features), where d_z is the dimension of the
                latent space. 
        """
        super().__init__()
        self.decoder_net = decoder_net
    
    def forward(self, z):
        """
        Computes the Negative Binomial distribution over the data space. What we are getting is the mean
        and the dispersion parameters, so it is needed a parameterization in order to get the NB 
        distribution parameters: total_count (dispersion) and probs (dispersion/(dispersion + mean))

        Parameters:
            z: [torch.Tensor]
        """
        mu, theta = torch.chunk(self.decoder_net(z), 2, dim=-1)
        # Ensure mean and dispersion are positive numbers
        mu = F.softplus(mu)
        theta = F.softplus(theta)
        # Parameterization into NB parameters
        p = theta / (theta + mu)
        r = theta

        return td.Independent(td.NegativeBinomial(total_count = r, probs = p), 1)

class ZINBDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Zero-inflated Negative Binomial decoder to obtain the parameters of the ZINB distribution.

        Parameters:
            decoder_net: [torch.nn.Module]
                The decoder network, takes a tensor of dimension (batch, d_z) and outputs
                a tensor of dimension (batch, 3*features), where d_z is the dimension of the
                latent space. 
        """
        super().__init__()
        self.decoder_net = decoder_net

    def forward(self, z):
        """
        Computes the Zero-inflated Negative Binomial distribution over the data space. What we are getting is the mean
        and the dispersion parameters, so it is needed a parameterization in order to get the NB 
        distribution parameters: total_count (dispersion) and probs (dispersion/(dispersion + mean)). We are also
        getting the pi_logits parameters, which accounts for the zero inflation probability.

        Parameters:
            z: [torch.Tensor]
        """
        mu, theta, pi_logits = torch.chunk(self.decoder_net(z), 3, dim = -1)
        # Ensure mean and dispersion are positive numbers and pi is in range [0,1]
        mu = F.softplus(mu)
        theta = F.softplus(theta)
        pi = torch.sigmoid(pi_logits)
        # Parameterization into NB parameters
        p = theta / (theta + mu)
        r = theta
        # Create Negative Binomial component
        nb = td.Independent(td.NegativeBinomial(total_count = r, probs = p), 1)

        return ZINB(nb, pi)


# ---------- DISTRIBUTIONS CLASSES DEFINITIONS ---------- #
class ZINB(td.Distribution):
    """
    Zero-inflated Negative Binomial (ZINB) distribution definition. The ZINB distribution is defined by the following:
        For x = 0:
            ZINB.log_prob(0) = log(pi + (1 - pi) * NegativeBinomial(0 | r, p))
        
        For x > 0:
            ZINB.log_prob(x) = log(1 - pi) + NegativeBinomial(x |r, p).log_prob(x)
    """
    arg_constraints = {}
    support = td.constraints.nonnegative_integer
    has_rsample = False

    def __init__(self, nb, pi, validate_args = None):
        """
        Parameters:
            nb: [torch.distributions.Independent(torch.distributions.NegativeBinomial)]
                Negative Binomial distribution component
            pi: [torch.Tensor]
                Zero-inflation probability from decoder output
        """
        self.nb = nb
        self.pi = pi
        batch_shape = nb.batch_shape
        super().__init__(batch_shape = batch_shape, validate_args = validate_args)
    
    def log_prob(self, x):
        """
        Defines the log_prob() function inherent from torch.distributions.Distribution.

        Parameters:
            x: [torch.Tensor]
        """
        nb_log_prob = self.nb.log_prob(x) #log probability of NB where x > 0
        nb_prob_zero = self.nb.prob(0) #probability of NB where x = 0

        log_prob_zero = torch.log(self.pi + (1 - self.pi) * nb_prob_zero + 1e-8)
        log_prob_nonzero = torch.log(1 - self.pi + 1e-8) * nb_log_prob

        return torch.where(x == 0, log_prob_zero, log_prob_nonzero)
    
    def sample(self, sample_shape=torch.Size()):
        """
        Defines the sample() function, which is used to sample data points using the distribution parameters.
            For x = 0:
                Binary mask for zero-inflated values
            For x > 0:
                Sample from Negative Binomial distribution
        """
        shape = self._extended_shape(sample_shape)
        zero_inflated = torch.bernoulli(self.pi.expand(shape))

        nb_sample = self.nb.sample(sample_shape)

        return torch.where(zero_inflated.bool(), torch.zeros_like(nb_sample), nb_sample)
    
# ---------- VAE CLASSES DEFINITIONS ---------- #

class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model
    """
    def __init__(self, prior, encoder, decoder):
        """
        Parameters:
            prior: [torch.nn.Module]
                Prior distribution over the latent space.
            encoder: [torch.nn.Module]
                Encoder distribution over the latent space.
            decoder: [torch.nn.Module]
                Decoder distribution over the data space.
        """
        super().__init__()
        self.prior = prior
        self.encoder = encoder
        self.decoder = decoder

    def elbo(self, x):
        """
        Compute the ELBO for the given data

        Parameters:
            x: [torch.Tensor]
        """
        q = self.encoder(x)
        z = q.rsample()
        elbo = torch.mean(self.decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()), dim=0)

        return elbo
    
    def sample(self, n_samples = 1):
        """
        Sample from the model.

        Parameters:
            n_samples: [int]
        """
        z = self.prior().sample(torch.Size([n_samples]))

        return self.decoder(z).sample()
    
    def loss(self, x):
        """
        Compute negative ELBO for the given data (loss)
        """
        return -self.elbo(x)
    
    def forward(self, x):
        """
        Computes a simple forward pass for the given data
        """
        q = self.encoder(x)
        z = q.rsample()
        return self.decoder(z)

# ---------- TRAINING LOOP ---------- #

def train(model, optimizer, data_loader, epochs, device):
    """
    Training loop for VAE model.

    Parameters:
        model: [VAE]
            The VAE model to train.
        optimizer: [torch.optim.Optimizer]
            The optimizer to use for training.
        data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
        epochs: [int]
            The number of epochs to train for.
        device: [torch.device]
            The device to use for training.
    """
    model.train()

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            x = x[0].to(device)
            optimizer.zero_grad()
            loss = model.loss(x)
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss = f"{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()