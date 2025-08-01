import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import math
from src.ABaCo.BatchEffectDataLoader import class_to_int, one_hot_encoding
import random
import seaborn as sns

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
        self.mu = nn.Parameter(torch.zeros(self.d_z), requires_grad=False)
        self.var = nn.Parameter(torch.ones(self.d_z), requires_grad=False)

    def forward(self):
        """
        Return prior distribution. This allows the computation of KL-divergence by calling self.prior() in the VAE class.

        Returns:
            prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mu, scale=self.var), 1)


class MoGPrior(nn.Module):
    def __init__(self, d_z, n_comp, multiplier=1.0):
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

        self.mu = nn.Parameter(torch.randn(n_comp, self.d_z) * multiplier)
        self.var = nn.Parameter(torch.randn(n_comp, self.d_z))

    def forward(self, k_ohe):
        """
        Return prior distribution, allowing for the computation of the KL-divergence by calling self.prior().

        Returns:
            prior: [torch.distributions.Distribution]
        """
        # Get parameters for each MoG component
        mu_k = k_ohe @ self.mu
        std_k = k_ohe @ torch.sqrt(torch.nn.functional.softplus(self.var) + 1e-8)

        return td.Independent(td.Normal(loc=mu_k, scale=std_k), 1)

    def cluster_loss(self):
        """
        Compute the clustering loss for the MoG prior. This loss encourages the components to be well separated
        by maximizing the pairwise KL divergence between the Gaussian components.
        """
        # Compute softplus to ensure positive variances
        stds = torch.sqrt(torch.nn.functional.softplus(self.var) + 1e-8)
        mus = self.mu
        n = self.n_comp

        # Compute pairwise KL divergences between all components
        kl_matrix = torch.zeros((n, n), device=mus.device)
        for i in range(n):
            for j in range(n):
                if i != j:
                    mu1, mu2 = mus[i], mus[j]
                    std1, std2 = stds[i], stds[j]
                    var1, var2 = std1**2, std2**2

                    # KL(N1 || N2) for diagonal Gaussians
                    kl = 0.5 * (
                        torch.sum(var1 / var2)
                        + torch.sum((mu2 - mu1) ** 2 / var2)
                        - self.d_z
                        + torch.sum(torch.log(var2))
                        - torch.sum(torch.log(var1))
                    )
                    kl_matrix[i, j] = kl

        # Take the minimum KL divergence between any two components
        min_kl = kl_matrix[kl_matrix > 0].min()
        # Loss is inverse of min KL (maximize separation)
        return 1.0 / (min_kl + 1e-8)


class VMMPrior(nn.Module):
    def __init__(
        self, d_z, n_features, n_comp, n_batch, encoder, multiplier=1.0, dataloader=None
    ):
        """
        Define a VampPrior Mixture Model prior distribution.

        Parameters:
            d_z: [int]
                Dimension of the latent space
            n_u: [int]
                Number of pseudo-inputs for the VMM distribution
            multiplier: [float]
                Parameter that controls sparsity of each Gaussian component
        """
        super().__init__()
        self.d_z = d_z
        self.n_features = n_features
        self.n_comp = n_comp
        self.n_batch = n_batch
        self.encoder = encoder
        self.dataloader = dataloader

        if self.dataloader is not None:
            self.u = self.sample_from_dataloader()
        else:
            self.u = nn.Parameter(
                torch.cat(
                    [
                        torch.rand(n_comp, n_features) * multiplier,
                        torch.zeros(n_comp, n_batch),
                    ],
                    dim=1,
                )
            )

        self.var = nn.Parameter(torch.randn(n_comp, self.d_z))

    def sample_from_dataloader(self):
        all_data = []
        bio_label = []
        # Collect until we have at least K samples
        for batch in self.dataloader:
            x = batch[0]
            z = batch[2]  # biological variability
            all_data.append(x)
            bio_label.append(z)
            if len(all_data) * x.shape[0] >= self.n_comp:
                break

        all_data = torch.cat(all_data, dim=0)  # (N, D)
        bio_label = torch.cat(bio_label, dim=0)  # (N, L)

        # Find the unique labels
        bio_dict = torch.unique(bio_label, dim=0)  # (G, L), G = #groups

        # For each unique label, compute the mean of its members
        selected_u = []
        for label in bio_dict:
            # mask of samples matching this label
            mask = (bio_label == label).all(dim=1)  # (N,)
            group_data = all_data[mask]  # (n_i, D)
            group_mean = group_data.mean(dim=0)  # (D,)
            selected_u.append(group_mean)

        selected_u = torch.stack(selected_u, dim=0)  # (G, D)

        # Zero pad for batch dim
        zeros_pad = torch.zeros(self.n_comp, self.n_batch, device=selected_u.device)
        selected_u = torch.cat([selected_u, zeros_pad], dim=1)

        # Return as a learnable parameter
        return nn.Parameter(selected_u.clone().detach().requires_grad_(True))

    def forward(self, k_ohe):
        """
        Return prior distribution, allowing for the computation of the KL-divergence by calling self.prior().

        Parameters:
            k_ohe: [torch.tensor]
                One-hot encoded tensor with the corresponding component the point belongs to.
            b_ohe: [torch.tensor]
                One-hot encoded tensor with the corresponding batch label, necessary to append at the Encoder input.
            encoder: [nn.Module]
                Encoder used for getting the centroid of the cluster by encoding the pseudo-input.

        Returns:
            prior: [torch.distributions.Distribution]
        """
        # Encode the pseudo-input
        u_k = k_ohe @ self.u
        _, mu_k, _ = self.encoder.encode(u_k)
        mu_k = mu_k[torch.arange(mu_k.size(0)), k_ohe.argmax(dim=1), :]  # (batch, d_z)

        # Get parameters for each MoG component
        std_k = k_ohe @ torch.sqrt(torch.nn.functional.softplus(self.var) + 1e-8)

        return td.Independent(td.Normal(loc=mu_k, scale=std_k), 1)

    def cluster_loss(self):
        """
        Compute the clustering loss for the MoG prior. This loss encourages the components to be well separated
        by maximizing the pairwise KL divergence between the Gaussian components.
        """

        # Compute softplus to ensure positive variances
        stds = torch.sqrt(torch.nn.functional.softplus(self.var) + 1e-8)
        _, mus, _ = self.encoder.encode(self.u)
        n = self.n_comp

        # Compute pairwise KL divergences between all components
        kl_matrix = torch.zeros((n, n), device=mus.device)
        for i in range(n):
            for j in range(n):
                if i != j:
                    mu1, mu2 = mus[i], mus[j]
                    std1, std2 = stds[i], stds[j]
                    var1, var2 = std1**2, std2**2

                    # KL(N1 || N2) for diagonal Gaussians
                    kl = 0.5 * (
                        torch.sum(var1 / var2)
                        + torch.sum((mu2 - mu1) ** 2 / var2)
                        - self.d_z
                        + torch.sum(torch.log(var2))
                        - torch.sum(torch.log(var1))
                    )
                    kl_matrix[i, j] = kl

        # Take the minimum KL divergence between any two components
        min_kl = kl_matrix[kl_matrix > 0].min()
        # Loss is inverse of min KL (maximize separation)
        return 1.0 / (min_kl + 1e-8)


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

    def encode(self, x):

        mu, var = torch.chunk(
            self.encoder_net(x), 2, dim=-1
        )  # chunk is used for separating the encoder output (batch, 2*d_z) into two separate vectors (batch, d_z)

        return mu, var

    def forward(self, x):
        """
        Computes the Gaussian Normal distribution over the latent space.

        Parameters:
            x: [torch.Tensor]
        """
        mu, var = self.encode(x)
        std = torch.sqrt(torch.nn.functional.softplus(var) + 1e-8)

        return td.Independent(td.Normal(loc=mu, scale=std), 1)


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

    def encode(self, x):

        comps = torch.chunk(
            self.encoder_net(x), self.n_comp, dim=-1
        )  # chunk used for separating the encoder output (batch, n_comp*(2*d_z + 1)) into n_comp separate vectors (batch, n_comp)

        # Parameters list (for extracting in loop)
        mu_list = []
        var_list = []
        pi_list = []

        for comp in comps:
            params = comp[
                :, :-1
            ]  # parameters mu and var are on the 2*d_z first values of the component
            pi_comp = comp[
                :, -1
            ]  # mixing probabilities is the last value of the component

            mu, var = torch.chunk(
                params, 2, dim=-1
            )  # separating mu from var using chunk

            mu_list.append(mu)
            var_list.append(var)
            pi_list.append(pi_comp)

        # Convert parameters list into tensor
        means = torch.stack(mu_list, dim=1)
        stds = torch.sqrt(
            torch.nn.functional.softplus(torch.stack(var_list, dim=1)) + 1e-8
        )
        pis = torch.stack(pi_list, dim=1)

        # Clamp to avoid error values (too low or too high)
        stds = torch.clamp(stds, min=1e-5, max=1e5)

        return pis, means, stds

    def forward(self, x):
        """
        Computes the Categorical distribution over the latent space, sampling the component index
        and returning the parameters of the selected component.

        Parameters:
            x: [torch.Tensor]
        """
        pis, means, stds = self.encode(x)

        # From the categorical distribution, sample the component index
        probs = F.softmax(pis, dim=-1)
        cat = td.Categorical(probs=probs)
        k = cat.sample()

        # Get the parameters of the selected component
        k_exp = k.view(-1, 1, 1).expand(-1, 1, means.size(-1))
        mu_k = means.gather(dim=1, index=k_exp).squeeze(1)
        std_k = stds.gather(dim=1, index=k_exp).squeeze(1)

        return td.Independent(td.Normal(loc=mu_k, scale=std_k), 1)


# ---------- DECODER CLASSES DEFINITIONS ---------- #


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
        mu, theta, pi_logits = torch.chunk(self.decoder_net(z), 3, dim=-1)
        # Ensure mean and dispersion are positive numbers and pi is in range [0,1]

        mu = F.softplus(mu)
        theta = F.softplus(theta) + 1e-4

        pi = torch.sigmoid(pi_logits)

        # Parameterization into NB parameters
        p = theta / (theta + mu)

        r = theta

        # Clamp values to avoid huge / small probabilities
        p = torch.clamp(p, min=1e-5, max=1 - 1e-5)

        # Create Negative Binomial component
        nb = td.NegativeBinomial(total_count=r, probs=p)
        # nb = td.Independent(nb, 1)

        return td.Independent(ZINB(nb, pi), 1)


class DMDecoder(nn.Module):
    def __init__(self, decoder_net, total_count, eps=1e-8):
        """
        Define a Dirichlet-Multinomial decoder to obtain the parameters of the distribution.

        Parameters:
            decoder_net: torch.nn.Module
                The decoder network, takes a tensor of dimension (batch, d_z) and outputs
                a tensor of dimension (batch, features), where d_z is the dimension of the
                latent space.
            total_count: int
                Total number of reads (or organisms) per sample. In practice, it is just x_i.sum(),
                where x_i is the sample i from the dataset.
            eps: float
                Small offset to avoid log(0) in probability computation.
        """
        super().__init__()
        self.decoder_net = decoder_net
        self.total_count = total_count
        self.eps = eps

    def forward(self, z):
        """
        Computes the Dirichlet-Multinomial distribution over the data space. We are obtaining the concentration parameter
        of the distribution, hence the decoder output would have shape (batch, features).

        Parameters:
            z: torch.Tensor
        """
        conc_logits = self.decoder_net(z)
        concentration = F.softplus(conc_logits) + self.eps

        dm = DirichletMultinomial(
            total_count=self.total_count, concentration=concentration
        )

        return td.Independent(dm, 1)


class ZIDMDecoder(nn.Module):
    def __init__(self, decoder_net, total_count, eps=1e-8):
        """
        Define a Zero-inflated Dirichlet-Multinomial decoder to obtain the parameters of the distribution.

        Parameters:
            decoder_net: [torch.nn.Module]
                The decoder network, takes a tensor of dimension (batch, d_z) and outputs
                a tensor of dimension (batch, features), where d_z is the dimension of the
                latent space.
            total_count: int
                Total number of reads (or organisms) per sample. In practice, it is just x_i.sum(),
                where x_i is the sample i from the dataset.
            eps: float
                Small offset to avoid log(0) in probability computation.

        """
        super().__init__()
        self.decoder_net = decoder_net
        self.total_count = total_count
        self.eps = eps

    def forward(self, z):
        conc_logits, pi_logits = torch.chunk(self.decoder_net(z), 2, dim=-1)

        concentration = F.softplus(conc_logits) + self.eps
        dm = DirichletMultinomial(
            total_count=self.total_count, concentration=concentration
        )

        pi = torch.sigmoid(pi_logits)
        zidm = ZIDM(dm=dm, pi=pi, eps=self.eps)

        return td.Independent(zidm, 1)


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

    def __init__(self, nb, pi, validate_args=None):
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
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def log_prob(self, x):
        """
        Defines the log_prob() function inherent from torch.distributions.Distribution.

        Parameters:
            x: [torch.Tensor]
        """
        nb_log_prob = self.nb.log_prob(x)  # log probability of NB where x > 0
        nb_prob_zero = torch.exp(
            self.nb.log_prob(torch.zeros_like(x))
        )  # probability of NB where x = 0
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


@td.kl.register_kl(ZINB, ZINB)
def kl_zinb_zinb(p, q):
    # Monte Carlo sampling from p
    num_samples = 5000
    samples = p.sample((num_samples,))
    log_p = p.log_prob(samples)
    log_q = q.log_prob(samples)

    kl = (log_p - log_q).mean(dim=0)
    return kl


class DirichletMultinomial(td.Distribution):
    """
    Dirichlet-Multinomial distribution, defined by:
        p ~ Dirichlet(alpha)
        x | p ~ Multinomial(total_count, p)
    """

    arg_constraints = {
        "total_count": td.constraints.nonnegative_integer,
        "concentration": td.constraints.positive,
    }
    support = td.constraints.nonnegative_integer
    has_rsample = False

    def __init__(
        self, total_count: int, concentration: torch.Tensor, validate_args=None
    ):
        """
        Parameters:
            total_count: scalar int for the Multinomial total counts N
            concentration: tensor of shape (..., num_categories) for Dirichlet alphas
        """
        self.total_count = total_count
        self.concentration = concentration
        batch_shape = concentration.shape[:-1]
        event_shape = concentration.shape[-1:]
        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def log_prob(self, x: torch.Tensor):
        """
        Defines the log_prob() function inherent from torch.distributions.Distribution.

        Parameters:
            x: torch.Tensor
        """
        # if self._validate_args:
        #     total = x.sum(dim=-1)
        #     if not torch.all(total == self.total_count):
        #         raise ValueError("DirichletMultinomial counts must sum to total_count")

        alpha = self.concentration
        N = self.total_count

        term1 = torch.lgamma(
            torch.tensor(N + 1, dtype=torch.float, device=x.device)
        ) - torch.lgamma(x + 1).sum(dim=-1)

        sum_alpha = alpha.sum(dim=-1)
        term2 = torch.lgamma(sum_alpha) - torch.lgamma(N + sum_alpha)

        term3 = torch.lgamma(x + alpha).sum(dim=-1) - torch.lgamma(alpha).sum(dim=-1)

        return term1 + term2 + term3

    def sample(self, sample_shape=torch.Size()):
        """
        Defines the sample() function, which is used to sample data points using the distribution parameters.
        """
        shape = self._extended_shape(sample_shape)
        p = td.Dirichlet(self.concentration).sample(sample_shape)

        batch_dims = p.shape[:-1]
        C = p.size(-1)

        # flatten batch dims
        p_flat = p.reshape(-1, C)

        # total_count per flat sample
        tc = self.total_count

        if isinstance(tc, torch.Tensor):
            tc_flat = tc.reshape(-1)

        else:
            tc_flat = None

        counts = []
        for i, probs in enumerate(p_flat):
            n = int(tc_flat[i]) if tc_flat is not None else self.total_count
            idx = torch.multinomial(probs, n, replacement=True)
            counts.append(torch.bincount(idx, minlength=C))

        x = torch.stack(counts, dim=0)

        return x.reshape(*batch_dims, C)


class ZIDM(td.Distribution):
    """
    Zero-inflated Dirichlet-Multinomial (ZIDM) distribution.
    Mixture of structural zeros per-category with a Dirichlet-Multinomial core.
    """

    arg_constraints = {}
    support = td.constraints.nonnegative_integer
    has_rsample = False

    def __init__(
        self,
        dm: DirichletMultinomial,
        pi: torch.Tensor,
        eps: float = 1e-8,
        validate_args=None,
    ):
        """
        Parameters:
            dm: DirichletMultinomial instance of the distribution
            pi: tensor of shape (..., num_categories), zero-inflation probability per category
            eps: small value to ensure non-zero concentration for masked-out categories
        """
        self.dm = dm
        self.pi = pi
        self.eps = eps
        batch_shape = dm.batch_shape
        event_shape = dm.event_shape
        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def log_prob(self, x: torch.Tensor):
        """
        Defines the log_prob() function inherent from torch.distributions.Distribution.

        Parameters:
            x: torch.Tensor
        """
        mask_nonzero = (x > 0).float()
        term_pi = torch.log((1 - self.pi) + self.eps) * mask_nonzero
        log_dm = self.dm.log_prob(x)

        return term_pi.sum(dim=-1) + log_dm

    def sample(self, sample_shape=torch.Size()):
        """
        Defines the sample() function, which is used to sample data points using the distribution parameters.
        """
        shape = self._extended_shape(sample_shape)
        zero_mask = torch.bernoulli(self.pi.expand(shape))
        alpha_adj = self.dm.concentration * (1 - zero_mask) + self.eps
        dm_adj = DirichletMultinomial(
            total_count=self.dm.total_count, concentration=alpha_adj
        )
        sample = dm_adj.sample()
        return sample


# ---------- VARIATIONAL AUTOENCODER CLASSES DEFINITIONS ---------- #


class VAE(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        prior,
        input_size,
        device,
    ):
        """
        Variational Autoencoder class definition.

        Parameters:
            encoder: [torch.nn.Module]
                Encoder network, takes a tensor of dimension (batch, features) and outputs a tensor of dimension (batch, 2*d_z)
            decoder: [torch.nn.Module]
                Decoder network, takes a tensor of dimension (batch, d_z) and outputs a tensor of dimension (batch, features)
            prior: [torch.distributions.Distribution]
                Prior distribution over the latent space
            input_size: [int]
                Dimension of the input data
            device: [str]
                Device to use for computations
            prior_type: [str]
                Type of prior distribution used in the VAE
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        self.input_size = input_size
        self.device = device

    def encode(self, x):
        """
        Forward pass through the VAE.

        Parameters:
            x: [torch.Tensor]
                Input data tensor of shape (batch, features)
        """
        z = self.encoder(x)

        return z

    def decode(self, z):
        """
        Decode the latent space representation to the data space.

        Parameters:
            z: [torch.Tensor]
                Latent space tensor of shape (batch, d_z)
        """
        recon_x = self.decoder(z)

        return recon_x

    def kl_divergence(self, z, k_ohe=None):
        """
        Compute the KL-divergence between the prior and the posterior distribution.

        Parameters:
            z: [torch.Tensor]
                Latent space tensor of shape (batch, d_z)
        """
        # 1. Encode the input data to get the posterior distribution parameters
        q_zx = self.encoder(z)

        # 2. Get the prior distribution
        if isinstance(self.prior, MoGPrior):
            p_z = self.prior(k_ohe)  # select Gaussian component of the prior
        elif isinstance(self.prior, VMMPrior):
            p_z = self.prior(k_ohe)
        else:
            p_z = self.prior()

        # 3. Compute the KL-divergence
        kl_div = td.kl_divergence(q_zx, p_z)

        return kl_div

    def get_posterior(self, x):
        """
        Given a set of points, compute the posterior distribution.

        Parameters:
        x: [torch.Tensor]
            Samples to pass to the encoder
        """
        q = self.encoder(x)
        return q.rsample()

    def pca_posterior(self, x):
        """
        Given a set of points, compute the PCA of the posterior distribution.

        Parameters:
        x: [torch.Tensor]
            Samples to pass to the encoder
        """
        z = self.get_posterior(x)
        pca = PCA(n_components=2)
        return pca.fit_transform(z.detach().cpu())


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


class metaABaCo(nn.Module):
    def __init__(
        self,
        data,
        n_bios,
        bio_label,
        n_batches,
        batch_label,
        n_features,
        device,
        # VAE parameters
        prior="MoG",
        pdist="ZINB",
        d_z=16,
        epochs=[1000, 2000, 2000],
        encoder_net=[512, 256, 128],
        decoder_net=[128, 256, 512],
        vae_act_fun=nn.ReLU(),
        # Discriminator parameters
        disc_net=[128, 64],
        disc_act_fun=nn.ReLU(),
    ):
        super().__init__()

        # Define known model parameters
        self.device = device
        self.data = data
        self.n_bios = n_bios
        self.bio_label = bio_label
        self.n_batches = n_batches
        self.batch_label = batch_label
        self.n_features = n_features
        self.d_z = d_z
        self.phase_1_epochs = epochs[0]
        self.phase_2_epochs = epochs[1]
        self.phase_3_epochs = epochs[2]

        # Defince dataloader
        self.dataloader = DataLoader(
            TensorDataset(
                torch.tensor(
                    self.data.select_dtypes(include="number").values,
                    dtype=torch.float32,
                ),  # samples
                one_hot_encoding(self.data[self.batch_label])[
                    0
                ],  # one hot encoded batch information
                one_hot_encoding(self.data[self.bio_label])[
                    0
                ],  # one hot encoded biological information
            ),
            batch_size=len(self.data),
        )

        for x, y, z in self.dataloader:  # just one iteration
            self.total_count = x.sum(dim=1).to(self.device)

        # Define Encoder
        encoder_net = [n_features + n_batches] + encoder_net  # first layer: conditional
        encoder_net.append(
            n_bios * (2 * d_z + 1)
        )  # last layer: gaussian parameters (mu*d_z + sigma*d_z + pi = 2*d_z + 1) times the number of gaussians (n_bios)
        modules = []
        for i in range(len(encoder_net) - 1):
            modules.append(nn.Linear(encoder_net[i], encoder_net[i + 1]))
            modules.append(vae_act_fun)
        modules.pop()  # Drop last activation function

        if prior == "MoG":
            self.encoder = MoGEncoder(nn.Sequential(*modules), n_bios)
            self.prior = MoGPrior(d_z, n_bios)

        elif prior == "VMM":
            self.encoder = MoGEncoder(nn.Sequential(*modules), n_bios)
            self.prior = VMMPrior(
                d_z,
                n_features,
                n_bios,
                n_batches,
                self.encoder,
                dataloader=self.dataloader,
            )

        else:
            raise NotImplementedError(
                "Only 'MoG' and 'VMM' prior are currently implemented in metaAbaco."
            )

        # Define Decoder
        decoder_net = [d_z + n_batches] + decoder_net  # first layer: conditional

        if pdist == "ZINB":
            decoder_net.append(
                3 * n_features
            )  # last layer: ZINB distribution parameters (n_features * (dispersion + dropout + mean))

        elif pdist == "DM":
            decoder_net.append(
                n_features
            )  # last layer: Dirichlet-Multinomial distribution parameters (n_feature * concentration)

        elif pdist == "ZIDM":
            decoder_net.append(
                2 * n_features
            )  # last layer: ZIDM distribution parameters (n_features * (concentration + dropout))

        else:
            raise NotImplementedError(
                "Only 'ZINB', 'DM' and 'ZIDM' decoders are currently implemented in metaAbaco."
            )

        modules = []
        for i in range(len(decoder_net) - 1):
            modules.append(nn.Linear(decoder_net[i], decoder_net[i + 1]))
            modules.append(vae_act_fun)
        modules.pop()  # Drop last activation function

        if pdist == "ZINB":
            self.decoder = ZINBDecoder(nn.Sequential(*modules))

        elif pdist == "DM":
            self.decoder = DMDecoder(nn.Sequential(*modules), self.total_count)

        elif pdist == "ZIDM":
            self.decoder = ZIDMDecoder(nn.Sequential(*modules), self.total_count)

        else:
            raise NotImplementedError(
                "Only 'ZINB', 'DM', and 'ZIDM' decoders are currently implemented in metaAbaco."
            )

        # Define the VAE
        self.vae = VAE(self.encoder, self.decoder, self.prior, n_features, device).to(
            device
        )

        # Define Batch Discriminator
        disc_net = [d_z + n_bios] + disc_net  # first layer: conditional
        disc_net.append(n_batches)  # last layer
        modules = []
        for i in range(len(disc_net) - 1):
            modules.append(nn.Linear(disc_net[i], disc_net[i + 1]))
            modules.append(disc_act_fun)
        modules.pop()  # remove last activation function

        self.disc = BatchDiscriminator(nn.Sequential(*modules)).to(device)

    def train_vae(
        self,
        train_loader,
        optimizer,
        w_elbo_nll=1.0,
        w_elbo_kl=1.0,
        w_bio_penalty=1.0,
        w_cluster_penalty=1.0,
    ):
        """
        Train the conditional VAE model. If clustering prior is used, penalization term is applied to increase sparsity of the clusters.

        Parameters:
            vae: [VAE]
                Variational Autoencoder model
            train_loader: [torch.utils.data.DataLoader]
                DataLoader for the training data
            optimizer: [torch.optim.Optimizer]
                Optimizer for training
            epochs: [int]
                Number of training epochs
            device: [str]
                Device to use for computations
        """
        self.vae.train()

        total_steps = len(train_loader) * self.phase_1_epochs
        progress_bar = tqdm(
            range(total_steps),
            desc="Training: VAE for learning meaningful embeddings",
        )

        for epoch in range(self.phase_1_epochs):
            total_loss = 0.0
            data_iter = iter(train_loader)
            for loader_data in data_iter:

                x = loader_data[0].to(self.device)
                ohe_batch = loader_data[1].to(self.device).float()  # Batch label
                ohe_bio = loader_data[2].to(self.device).float()  # Bio type label

                optimizer.zero_grad()

                # Encode and decode the input data along with the one-hot encoded batch label
                q_zx = self.vae.encoder(
                    torch.cat([x, ohe_batch], dim=1)
                )  # td.Distribution
                z = q_zx.rsample()  # latent points
                p_xz = self.vae.decoder(
                    torch.cat([z, ohe_batch], dim=1)
                )  # td.Distribution

                # Compute the reconstruction loss (Negative log-likelihood)
                recon_loss = (w_elbo_nll) * -p_xz.log_prob(x).mean()

                # Compute the KL-divergence loss
                kl_loss = (w_elbo_kl) * self.vae.kl_divergence(
                    torch.cat([x, ohe_batch], dim=1), k_ohe=ohe_bio
                ).mean()  # KL divergence function first encodes the input data

                # Compute extra penalization term for clustering priors
                bio_penalty = 0.0  # ensures points from the same biological group to be mapped on the same cluster
                cluster_penalty = 0.0  # ensures gaussian components to not overlap

                if isinstance(self.vae.prior, (MoGPrior, VMMPrior)):
                    # Compute penalty for biological mapping
                    pred_bio, _, _ = self.vae.encoder.encode(
                        torch.cat([x, ohe_batch], dim=1)
                    )

                    bio_penalty += (w_bio_penalty) * F.cross_entropy(
                        pred_bio, ohe_bio.argmax(dim=1)
                    )

                    # Compute penalty for group clusters
                    cluster_penalty += (
                        w_cluster_penalty
                    ) * self.vae.prior.cluster_loss()

                # Total loss is reconstruction loss + KL divergence loss
                loss = recon_loss + kl_loss + bio_penalty + cluster_penalty

                # Backpropagation and optimization step
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Update progress bar
                progress_bar.set_postfix(
                    vae_loss=f"{total_loss:.4f}",
                    bio_penalty=f"{bio_penalty:.4f}",
                    clustering_loss=f"{cluster_penalty:.4f}",
                    elbo=f"{(recon_loss+kl_loss):.4f}",
                    epoch=f"{epoch}/{self.phase_1_epochs+1}",
                )
                progress_bar.update()

        progress_bar.close()

    def batch_correct(
        self,
        train_loader,
        vae_optimizer,
        disc_optimizer,
        adv_optimizer,
        w_disc=1.0,
        w_adv=1.0,
        w_elbo_nll=1.0,
        w_elbo_kl=1.0,
        w_bio_penalty=1.0,
        w_cluster_penalty=1.0,
    ):
        """
        Train the conditional VAE model for batch correction. This is trained after VAE prior parameters are defined,
        """
        self.vae.train()

        total_steps = len(train_loader) * self.phase_2_epochs
        progress_bar = tqdm(
            range(total_steps),
            desc="Training: Embeddings batch effect correction using adversrial training",
        )

        for epoch in range(self.phase_2_epochs):
            total_loss = 0.0
            data_iter = iter(train_loader)
            for loader_data in data_iter:

                x = loader_data[0].to(self.device)
                ohe_batch = loader_data[1].to(self.device).float()  # Batch label
                ohe_bio = loader_data[2].to(self.device).float()  # Bio type label

                # 1. Forward pass latent points to discriminator
                disc_optimizer.zero_grad()
                with torch.no_grad():
                    q_zx = self.vae.encoder(
                        torch.cat([x, ohe_batch], dim=1)
                    )  # td.Distribution
                    z = q_zx.rsample()  # latent points

                pred_batch = self.disc(torch.cat([z, ohe_bio], dim=1))
                disc_loss = (w_disc) * self.disc.loss(
                    pred_batch, ohe_batch.argmax(dim=1)
                )

                # 2. Backpropagation and optimization step for discriminator
                disc_loss.backward()
                disc_optimizer.step()

                # 3. Adversarial backpropagation and optimization step for encoder
                adv_optimizer.zero_grad()
                q_zx = self.vae.encoder(
                    torch.cat([x, ohe_batch], dim=1)
                )  # td.Distribution
                z = q_zx.rsample()  # latent points
                pred_batch = self.disc(torch.cat([z, ohe_bio], dim=1))
                disc_loss = self.disc.loss(pred_batch, ohe_batch.argmax(dim=1))
                adv_loss = (w_adv) * -disc_loss
                adv_loss.backward()
                adv_optimizer.step()

                # 4. Forward pass through VAE
                vae_optimizer.zero_grad()
                q_zx = self.vae.encoder(
                    torch.cat([x, ohe_batch], dim=1)
                )  # td.Distribution
                z = q_zx.rsample()  # latent points
                p_xz = self.vae.decoder(
                    torch.cat([z, ohe_batch], dim=1)
                )  # td.Distribution

                # Compute the reconstruction loss (Negative log-likelihood)
                recon_loss = (w_elbo_nll) * -p_xz.log_prob(x).mean()

                # Compute the KL-divergence loss
                kl_loss = (w_elbo_kl) * self.vae.kl_divergence(
                    torch.cat([x, ohe_batch], dim=1), k_ohe=ohe_bio
                ).mean()  # KL divergence function first encodes the input data

                # Compute extra penalization term for clustering priors
                bio_penalty = 0.0  # ensures points from the same biological group to be mapped on the same cluster
                cluster_penalty = 0.0  # ensures gaussian components to not overlap

                if isinstance(self.vae.prior, (MoGPrior, VMMPrior)):
                    # Compute penalty for biological mapping
                    pred_bio, _, _ = self.vae.encoder.encode(
                        torch.cat([x, ohe_batch], dim=1)
                    )

                    bio_penalty += (w_bio_penalty) * F.cross_entropy(
                        pred_bio, ohe_bio.argmax(dim=1)
                    )

                # Total loss is reconstruction loss + KL divergence loss
                vae_loss = recon_loss + kl_loss + bio_penalty + cluster_penalty

                # Backpropagation and optimization step
                vae_loss.backward()
                vae_optimizer.step()

                total_loss += vae_loss.item()

                # Update progress bar
                progress_bar.set_postfix(
                    vae_loss=f"{total_loss:.4f}",
                    bio_penalty=f"{bio_penalty:.4f}",
                    clustering_loss=f"{cluster_penalty:.4f}",
                    elbo=f"{(recon_loss+kl_loss):.4f}",
                    disc_loss=f"{disc_loss:.4f}",
                    adv_loss=f"{adv_loss:.4f}",
                    epoch=f"{epoch}/{self.phase_2_epochs+1}",
                )
                progress_bar.update()

        progress_bar.close()

    def batch_mask(
        self,
        train_loader,
        decoder_optimizer,
        smooth_annealing=True,
        cycle_reg=None,
        w_elbo_nll=1.0,
        w_cycle=1e-3,
    ):
        """
        Pre-trained VAE will now have frozen encoder and batch labels masked at the encoder.
        """

        self.vae.train()

        total_steps = len(train_loader) * self.phase_3_epochs
        progress_bar = tqdm(
            range(total_steps), desc="Training: VAE decoder with masked batch labels"
        )

        for epoch in range(self.phase_3_epochs):
            # Introduce slow transition to full batch masking
            if smooth_annealing:
                alpha = max(0.0, 1.0 - (2 * epoch / self.phase_3_epochs))
            else:
                alpha = 0.0

            data_iter = iter(train_loader)
            for loader_data in data_iter:

                x = loader_data[0].to(self.device)
                ohe_batch = loader_data[1].to(self.device).float()  # Batch label
                ohe_bio = loader_data[2].to(self.device).float()  # Bio type label

                # VAE ELBO computation with masked batch label
                decoder_optimizer.zero_grad()

                # Forward pass to encoder
                q_zx = self.vae.encoder(torch.cat([x, ohe_batch], dim=1))

                # Sample from encoded point
                z = q_zx.rsample()

                # Forward pass to the decoder
                p_xz = self.vae.decoder(
                    torch.cat([z, alpha * ohe_batch], dim=1)
                )  # masked batch label

                # Compute the reconstruction loss (Negative log-likelihood)
                recon_loss = (w_elbo_nll) * -p_xz.log_prob(x).mean()

                # Cycle loss for regularization (reconstructed point should be mapped to same cluster)
                if cycle_reg is not None:
                    # Reconstruct point
                    recon_x = p_xz.sample()
                    recon_z_params = self.vae.encoder.encode(
                        [recon_x, ohe_batch], dim=1
                    )

                    recon_pi = (
                        recon_z_params[0]
                        if isinstance(self.vae.prior, (MoGPrior, VMMPrior))
                        else None
                    )

                    cycle_loss = (
                        (w_cycle) * F.cross_entropy(recon_pi, ohe_bio.argmax(dim=1))
                        if recon_pi is not None
                        else 0
                    )

                else:
                    cycle_loss = 0

                # Compute loss
                vae_loss = recon_loss + cycle_loss
                vae_loss.backward()
                decoder_optimizer.step()

                # Update progress bar
                progress_bar.set_postfix(
                    vae_loss=f"{vae_loss:12.4f}",
                    cycle_loss=f"{cycle_loss:12.4f}",
                    epoch=f"{epoch+1}/{self.phase_3_epochs}",
                )
                progress_bar.update()

        progress_bar.close()

    def correct(
        self,
        smooth_annealing=True,
        cycle_reg=None,
        seed=None,
        # VAE Model parameters
        w_elbo_nll=1.0,
        w_elbo_kl=1.0,
        w_bio_penalty=1.0,
        w_cluster_penalty=1.0,
        w_cycle=1e-3,
        # Batch discriminator parameters
        w_disc=1.0,
        w_adv=1.0,
        # Optimizers learning rates
        phase_1_vae_lr=1e-3,
        phase_2_vae_lr=1e-3,
        phase_3_vae_lr=1e-6,
        disc_lr=1e-3,
        adv_lr=1e-3,
    ):

        # Define optimizer
        if isinstance(self.vae.prior, MoGPrior):
            prior_params = self.vae.prior.parameters()

        elif isinstance(self.vae.prior, VMMPrior):
            prior_params = [self.vae.prior.u, self.vae.prior.var]

        vae_optimizer_1 = torch.optim.Adam(
            [
                {"params": self.vae.encoder.parameters()},
                {"params": self.vae.decoder.parameters()},
                {"params": prior_params},
            ],
            lr=phase_1_vae_lr,
        )

        vae_optimizer_2 = torch.optim.Adam(
            [
                {"params": self.vae.encoder.parameters()},
                {
                    "params": self.vae.decoder.parameters()
                },  # only VAE weights are updated, prior distribution is fixed on this stage
            ],
            lr=phase_2_vae_lr,
        )

        vae_optimizer_3 = torch.optim.Adam(
            [
                {
                    "params": self.vae.decoder.parameters()
                },  # onlye Decoder weights are updated, Encoder is fixed on this stage
            ],
            lr=phase_3_vae_lr,
        )

        disc_optimizer = torch.optim.Adam(
            self.disc.parameters(),
            lr=disc_lr,
        )

        adv_optimizer = torch.optim.Adam(
            [
                {"params": self.vae.encoder.parameters()},
            ],
            lr=adv_lr,
        )

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # PHASE 1: Train VAE for reconstructing data and getting meaningful embeddings
        self.train_vae(
            self.dataloader,
            vae_optimizer_1,
            w_elbo_nll,
            w_elbo_kl,
            w_bio_penalty,
            w_cluster_penalty,
        )
        # PHASE 2: Batch effect correction on the latent space with learned prior distribution
        self.batch_correct(
            self.dataloader,
            vae_optimizer_2,
            disc_optimizer,
            adv_optimizer,
            w_disc,
            w_adv,
            w_elbo_nll,
            w_elbo_kl,
            w_bio_penalty,
            w_cluster_penalty,
        )
        # PHASE 3: Batch masking to the decoder to reconstruct data without batch effect
        self.batch_mask(
            self.dataloader,
            vae_optimizer_3,
            smooth_annealing,
            cycle_reg,
            w_elbo_nll,
            w_cycle,
        )

    def reconstruct(
        self,
        seed=None,
        mask=True,
    ):
        self.vae.eval()

        recon_data = []

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        for loader_data in iter(self.dataloader):
            x = loader_data[0].to(self.device)
            ohe_batch = loader_data[1].to(self.device).float()  # Batch label
            ohe_bio = loader_data[2].to(self.device).float()  # Bio type label

            # Encode and decode the input data along with the one-hot encoded batch label
            q_zx = self.vae.encoder(torch.cat([x, ohe_batch], dim=1))  # td.Distribution
            z = q_zx.rsample()  # latent points
            if mask == True:
                p_xz = self.vae.decoder(
                    torch.cat([z, torch.zeros_like(ohe_batch.to(self.device))], dim=1)
                )  # td.Distribution
            else:
                p_xz = self.vae.decoder(
                    torch.cat([z, ohe_batch], dim=1)
                )  # useful when there is no batch effect to correct

            # Sample from the output distribution
            x_recon = p_xz.sample()  # Reconstructed data

            # Rebuild the input data format for analysis
            recon_data.append(x_recon.cpu().detach().numpy())

        np_recon_data = np.vstack([t for t in recon_data])

        x_recon_data = pd.concat(
            [
                self.data.select_dtypes(exclude="number"),
                pd.DataFrame(
                    np_recon_data,
                    index=self.data.index,
                    columns=self.data.select_dtypes("number").columns,
                ),
            ],
            axis=1,
        )

        return x_recon_data

    def plot_pca_posterior(self, figsize=(14, 6), palette="tab10"):
        """
        Get the plot of the first 2 principal components of the posterior distribution.
        """
        self.vae.eval()
        z_pca = []
        for loader_data in iter(self.dataloader):
            x = loader_data[0].to(self.device)
            ohe_batch = loader_data[1].to(self.device).float()  # Batch label
            ohe_bio = loader_data[2].to(self.device).float()  # Bio type label

            l1 = ohe_batch.detach().cpu().numpy().argmax(axis=1)
            l2 = ohe_bio.detach().cpu().numpy().argmax(axis=1)

            coords = self.vae.pca_posterior(torch.cat([x, ohe_batch], dim=1))
            z_pca.append(coords)

        coords = np.vstack(coords)

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        sns.scatterplot(
            x=coords[:, 0],
            y=coords[:, 1],
            hue=l1,
            palette=palette,
            ax=axes[0],
            legend="full",
        )
        axes[0].set_title("Posterior PCA colored by batch group")
        sns.scatterplot(
            x=coords[:, 0],
            y=coords[:, 1],
            hue=l2,
            palette=palette,
            ax=axes[1],
            legend="full",
        )
        axes[1].set_title("Posterior PCA colored by biological group")
        plt.tight_layout()
        plt.show()
