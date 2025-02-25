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
        return td.Independent(td.Normal(loc=self.mu, scale=self.var), 1)
    
class MoGPrior(nn.Module):
    def __init__(self, d_z, n_comp, multiplier = 1.0):
        """
        Define a Mixture of Gaussian Normal prior distribution.

        Parameters:
            d_z: [int]
                Dimension of the latent space
            n_comp: [int]
                Number of components for the MoG distribution
            multiplier: [float]
                Parameter that controls sparsity of each Gaussian component
        """
        super().__init__()
        self.d_z = d_z
        self.n_comp = n_comp

        self.mu = nn.Parameter(torch.randn(n_comp, self.d_z)*multiplier)
        self.var = nn.Parameter(torch.randn(n_comp, self.d_z))
        self.pi = nn.Parameter(torch.zeros(n_comp))
    
    def forward(self):
        """
        Return prior distribution, allowing for the computation of the KL-divergence by calling self.prior().

        Returns:
            prior: [torch.distributions.Distribution]
        """
        # Get parameters for each MoG component
        means = self.mu
        stds = torch.sqrt(torch.nn.functional.softplus(self.var) + 1e-8)
        logits = self.pi

        # Call MoG distribution
        prior = MixtureOfGaussians(logits, means, stds)

        return prior

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
        
        std = torch.clamp(torch.exp(var * 0.5), min = 1e-8, max = 1e8)

        return td.Independent(td.Normal(loc = mu, scale = std), 1)

class MoGEncoder(nn.Module):
    def __init__(self, encoder_net, n_comp):
        """
        Define a Mixture of Gaussians encoder to obtain the parameters of the MoG distribution.

        Parameters:
            encoder_net: [torch.nn.Module]
                The encoder network, takes a tensor of dimension (batch, features) and
                outputs a tensor of dimension (batch, n_comp*(2*d_z + 1)), where d_z is the dimension
                of the latent space, and n_comp the number of components of the MoG distribution.
            n_comp: [int]
                Number of components for the MoG distribution.
        """
        super().__init__()
        self.n_comp = n_comp
        self.encoder_net = encoder_net
    
    def forward(self, x):
        """
        Computes the MoG distribution over the latent space.

        Parameters:
            x: [torch.Tensor]
        """
        comps = torch.chunk(self.encoder_net(x), self.n_comp, dim=-1) #chunk used for separating the encoder output (batch, n_comp*(2*d_z + 1)) into n_comp separate vectors (batch, n_comp)
        
        # Parameters list (for extracting in loop)
        mu_list = []
        var_list = []
        pi_list = []
        
        for comp in comps:
            params = comp[:, :-1] # parameters mu and var are on the 2*d_z first values of the component
            pi_comp = comp[:, -1] # mixing probabilities is the last value of the component

            mu, var = torch.chunk(params, 2, dim=-1) #separating mu from var using chunk

            mu_list.append(mu)
            var_list.append(var)
            pi_list.append(pi_comp)
        
        # Convert parameters list into tensor
        means = torch.stack(mu_list, dim=1)
        stds = torch.sqrt(torch.nn.functional.softplus(torch.stack(var_list, dim=1)) + 1e-8)
        pis = torch.stack(pi_list, dim=1)

        # Clamp to avoid error values (too low or too high)
        stds = torch.clamp(stds, min = 1e-5, max = 1e5)
        # Create individual Gaussian distribution per component
        mog_dist = MixtureOfGaussians(pis, means, stds)

        return mog_dist


    
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
        theta = F.softplus(theta) + 1e-4

        pi = torch.sigmoid(pi_logits)

        # Parameterization into NB parameters
        p = theta / (theta + mu)

        r = theta 

        # Clamp values to avoid huge / small probabilities
        p = torch.clamp(p, min=1e-5, max = 1 - 1e-5)

        # Create Negative Binomial component
        nb = td.NegativeBinomial(total_count = r, probs = p)
        # nb = td.Independent(nb, 1)

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
        nb_prob_zero = torch.exp(self.nb.log_prob(torch.zeros_like(x))) #probability of NB where x = 0
        log_prob_zero = torch.log(self.pi + (1 - self.pi) * nb_prob_zero + 1e-8)
        log_prob_nonzero = torch.log(1 - self.pi + 1e-8) + nb_log_prob

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
    
class MixtureOfGaussians(td.Distribution):
    """
    A Mixture of Gaussians distribution with reparameterized sampling. Computation of gradients is possible.
    """
    arg_constraints = {}
    support = td.constraints.real
    has_rsample = True # Implemented the rsample() through the Gumbel-softmax reparameterization trick.

    def __init__(self, mixture_logits, means, stds, temperature = 0.1, validate_args=None):
        self.mixture_logits = mixture_logits
        self.means = means
        self.stds = stds
        self.temperature = temperature

        batch_shape = self.mixture_logits.shape[:-1]
        event_shape = self.means.shape[-1:]
        super().__init__(batch_shape = batch_shape, event_shape = event_shape, validate_args = validate_args)

    def rsample(self, sample_shape = torch.Size()):
        """
        Reparameterized sampling using the Gubel-softmax trick.
        """

        # Step 1 - Sample for every component

        logits = self.mixture_logits.expand(sample_shape + self.mixture_logits.shape)
        means = self.means.expand(sample_shape + self.means.shape)
        stds = self.stds.expand(sample_shape + self.stds.shape)

        eps = torch.rand_like(means)
        comp_samples = means + eps * stds

        # Step 2 - Generate Gumbel noise for each component
        u = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(u + 1e-5) + 1e-5)

        # Step 3 - Compute y_i (gumbel-softmax trick)
        weights = F.softmax((logits + gumbel_noise) / self.temperature, dim=-1)
        weights = weights.unsqueeze(-1)

        # Step 4 - Sum every component for final sampling
        sample = torch.sum(weights * comp_samples, dim=-2)
        return sample
    
    def sample(self, sample_shape=torch.Size()):
        """
        Sample from the MoG distribution.
        """
        return self.rsample(sample_shape)
    
    def log_prob(self, value):
        """
        Compute the log probability of a given value. The log prob of a MoG is defined as:

            log_prob(x) = log [sum_k (pi_k * N(x; mu_k, sigma_k^2)]

        Where pi_k are the mixture probabilities.
        """
        value = value.unsqueeze(-2)

        normal = td.Normal(self.means, self.stds)
        log_prob_comp = normal.log_prob(value)
        log_prob_comps = log_prob_comp.sum(dim = -1)

        log_weights = F.log_softmax(self.mixture_logits, dim=-1)
        log_weights = log_weights.expand(log_prob_comps.shape)

        log_prob = torch.logsumexp(log_weights + log_prob_comps, dim = -1)

        return log_prob
    
    @property
    def mean(self):
        """
        Mixture mean: weighted sum of component means
        """
        weights = F.softmax(self.mixture_logits, dim = -1)

        return torch.sum(weights.unsqueeze(-1) * self.means, dim = -2)
    
    def variance(self):
        """
        Mixture variance: weighted sum of (variance + squared mean) minus squared mixture mean
        """
        weights = F.softmax(self.mixture_logits, dim=-1)
        mixture_mean = self.mean

        comp_var = self.stds ** 2
        second_moment = torch.sum(weights.unsqueeze(-1) * (comp_var + self.means ** 2), dim = -2)

        return second_moment - mixture_mean ** 2
    
    def entropy(self):

        raise NotImplementedError("Entropy is not implemented in Mixture of Gaussians distribution.")

# In order to register the kd.kl_divergence() function for the MixtureOfGaussians class
@td.kl.register_kl(MixtureOfGaussians, MixtureOfGaussians)
def kl_mog_mog(p, q):
    # Monte Carlo sampling from p
    num_samples = 5000
    samples = p.rsample((num_samples,))
    log_p = p.log_prob(samples)
    log_q = q.log_prob(samples)

    kl = (log_p - log_q).mean(dim=0)
    return kl

# ---------- VAE CLASSES DEFINITIONS ---------- #

class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model
    """
    def __init__(self, prior, encoder, decoder, beta = 1.0):
        """
        Parameters:
            prior: [torch.nn.Module]
                Prior distribution over the latent space.
            encoder: [torch.nn.Module]
                Encoder distribution over the latent space.
            decoder: [torch.nn.Module]
                Decoder distribution over the data space.
            beta: [float]
                Parameter controling force of kl_divergence
        """
        super().__init__()
        self.prior = prior
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta

    def elbo(self, x):
        """
        Compute the ELBO for the given data

        Parameters:
            x: [torch.Tensor]
        """
        q = self.encoder(x)
        z = q.rsample()

        recon_log_prob = self.decoder(z).log_prob(x)
        recon_loss = recon_log_prob.sum(dim=1)

        kl_loss = self.beta * td.kl_divergence(q, self.prior())

        elbo = torch.mean(recon_loss - kl_loss, dim=0)

        return elbo, torch.mean(recon_loss), torch.mean(kl_loss)
    
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
        loss_elbo, loss_recon, loss_kl = self.elbo(x)
        return -loss_elbo, -loss_recon, -loss_kl
    
    def forward(self, x):
        """
        Computes a simple forward pass for the given data
        """
        q = self.encoder(x)
        z = q.rsample()
        return self.decoder(z)

# ---------- DISCRIMINATOR AND CLASSIFIERS ---------- #

class BatchDiscriminator(nn.Module):
    """
    Define the Batch Discriminator for adversarial training
    """
    def __init__(self, net):
        super().__init__()
        self.net = net
    
    def forward(self, x):
        batch_class = self.net(x)
        return batch_class
    
    def loss(self, pred, true):
        
        loss = nn.CrossEntropyLoss()

        return loss(pred, true)
    



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
            total_loss, recon_loss, kl_loss = model.loss(x)
            total_loss.backward()
            optimizer.step()

            progress_bar.set_postfix(
                loss = f"{total_loss.item():12.4f}", 
                recon_loss = f"{recon_loss.item():12.4f}", 
                kl_loss = f"{kl_loss.item():12.4f}", 
                epoch = f"{epoch+1}/{epochs}")
            progress_bar.update()

def adversarial_loss(pred_batch, real_batch):
    
    loss = nn.CrossEntropyLoss()

    return -loss(pred_batch, real_batch)

def train_abaco(vae, discriminator, data_loader, ohe_batch, epochs, device, w_disc = 1.0, w_adv = 1.0):

    vae.train()
    discriminator.train()

    adv_optim = torch.optim.Adam(vae.encoder.parameters(), lr = 1e-4)
    vae_optim = torch.optim.Adam(vae.parameters(), lr = 1e-4)
    disc_optim = torch.optim.Adam(discriminator.parameters(), lr = 1e-4)

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            x = x[0].to(device)
            ohe_batch = ohe_batch.to(device).float()

            #First step: Forward pass through encoder and to discriminator
            with torch.no_grad():
                q = vae.encoder(x)
                z = q.rsample()
            
            # Detach z
            z = z.detach()

            # Discriminator Loss
            disc_optim.zero_grad()
            b_pred = discriminator(z)
            disc_loss = w_disc * discriminator.loss(b_pred, ohe_batch)
            disc_loss.backward()
            disc_optim.step()

            # Second step: Compute adversarial loss to encoder

            # Adversarial loss
            adv_optim.zero_grad()
            q = vae.encoder(x)
            z = q.rsample()
            b_pred = discriminator(z)
            adv_loss = w_adv * adversarial_loss(b_pred, ohe_batch)
            adv_loss.backward()
            adv_optim.step()

            # Third step: VAE ELBO computation
            vae_optim.zero_grad()
            vae_loss, _, _ = vae.loss(x)
            vae_loss.backward()
            vae_optim.step()

            # Update progress bar
            progress_bar.set_postfix(
                vae_loss = f"{vae_loss.item():12.4f}",
                disc_loss = f"{disc_loss.item():12.4f}",
                adv_loss = f"{adv_loss.item():12.4f}",
                epoch=f"{epoch+1}/{epochs}"
                )
            progress_bar.update()


            


