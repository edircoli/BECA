import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.decomposition import PCA
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import math
from BatchEffectDataLoader import class_to_int

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


class VMMPrior(nn.Module):
    def __init__(self, d_z, n_comp, mu, multiplier=1.0):
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

        self.mu = mu
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
        Computes the MoG distribution over the latent space.

        Parameters:
            x: [torch.Tensor]
        """
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
        # Create individual Gaussian distribution per component
        mog_dist = MixtureOfGaussians(pis, means, stds)

        return mog_dist


class AttentionMoGEncoder(nn.Module):
    def __init__(self, pre_encoder_net, attention, post_encoder_net, n_comp):
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
        self.pre_encoder_net = pre_encoder_net
        self.attention = attention
        self.post_encoder_net = post_encoder_net

    def encode(self, x):

        encoder_output = self.post_encoder_net(
            self.attention(self.pre_encoder_net(x).unsqueeze(1)).squeeze(1)
        )

        comps = torch.chunk(
            encoder_output, self.n_comp, dim=-1
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
        Computes the MoG distribution over the latent space.

        Parameters:
            x: [torch.Tensor]
        """
        # Encode
        pis, means, stds = self.encode(x)
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

        return td.Independent(td.NegativeBinomial(total_count=r, probs=p), 1)


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


class AttentionZINBDecoder(nn.Module):
    def __init__(self, pre_decoder_net, attention, post_decoder_net):
        """
        Define a Zero-inflated Negative Binomial decoder to obtain the parameters of the ZINB distribution.

        Parameters:
            decoder_net: [torch.nn.Module]
                The decoder network, takes a tensor of dimension (batch, d_z) and outputs
                a tensor of dimension (batch, 3*features), where d_z is the dimension of the
                latent space.
        """
        super().__init__()
        self.pre_decoder_net = pre_decoder_net
        self.attention = attention
        self.post_decoder_net = post_decoder_net

    def forward(self, z):
        """
        Computes the Zero-inflated Negative Binomial distribution over the data space. What we are getting is the mean
        and the dispersion parameters, so it is needed a parameterization in order to get the NB
        distribution parameters: total_count (dispersion) and probs (dispersion/(dispersion + mean)). We are also
        getting the pi_logits parameters, which accounts for the zero inflation probability.

        Parameters:
            z: [torch.Tensor]
        """

        decoder_output = self.post_decoder_net(
            self.attention(self.pre_decoder_net(z).unsqueeze(1)).squeeze(1)
        )

        mu, theta, pi_logits = torch.chunk(decoder_output, 3, dim=-1)
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


class MixtureOfGaussians(td.Distribution):
    """
    A Mixture of Gaussians distribution with reparameterized sampling. Computation of gradients is possible.
    """

    arg_constraints = {}
    support = td.constraints.real
    has_rsample = True  # Implemented the rsample() through the Gumbel-softmax reparameterization trick.

    def __init__(
        self, mixture_logits, means, stds, temperature=1e-5, validate_args=None
    ):
        self.mixture_logits = mixture_logits
        self.means = means
        self.stds = stds
        self.temperature = temperature

        batch_shape = self.mixture_logits.shape[:-1]
        event_shape = self.means.shape[-1:]
        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def rsample(self, sample_shape=torch.Size()):
        """
        Reparameterized sampling using the Gubel-softmax trick.
        """

        # Step 1 - Sample for every component

        logits = self.mixture_logits.expand(sample_shape + self.mixture_logits.shape)
        means = self.means.expand(sample_shape + self.means.shape)
        stds = self.stds.expand(sample_shape + self.stds.shape)

        eps = torch.randn_like(means)
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
        log_prob_comps = log_prob_comp.sum(dim=-1)

        log_weights = F.log_softmax(self.mixture_logits, dim=-1)
        log_weights = log_weights.expand(log_prob_comps.shape)

        log_prob = torch.logsumexp(log_weights + log_prob_comps, dim=-1)

        return log_prob

    @property
    def mean(self):
        """
        Mixture mean: weighted sum of component means
        """
        weights = F.softmax(self.mixture_logits, dim=-1)

        return torch.sum(weights.unsqueeze(-1) * self.means, dim=-2)

    def variance(self):
        """
        Mixture variance: weighted sum of (variance + squared mean) minus squared mixture mean
        """
        weights = F.softmax(self.mixture_logits, dim=-1)
        mixture_mean = self.mean

        comp_var = self.stds**2
        second_moment = torch.sum(
            weights.unsqueeze(-1) * (comp_var + self.means**2), dim=-2
        )

        return second_moment - mixture_mean**2

    def entropy(self):

        raise NotImplementedError(
            "Entropy is not implemented in Mixture of Gaussians distribution."
        )


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
    Define a Variational Autoencoder (VAE) model.
    """

    def __init__(self, prior, decoder, encoder, beta=1.0):
        """
        Parameters:
        prior: [torch.nn.Module]
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """

        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder
        self.beta = beta

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample()
        elbo = torch.mean(
            self.decoder(z).log_prob(x) - self.beta * td.kl_divergence(q, self.prior()),
            dim=0,
        )

        return elbo

    def kl_div_loss(self, x):
        q = self.encoder(x)
        z = q.rsample()
        kl_loss = torch.mean(
            self.beta * td.kl_divergence(q, self.prior()),
            dim=0,
        )
        return kl_loss

    def sample(self, n_samples=1):
        """
        Sample from the model.

        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()

    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)

    def get_posterior(self, x):
        """
        Given a set of points, compute the posterior distribution.

        Parameters:
        x: [torch.Tensor]
            Samples to pass to the encoder
        """
        q = self.encoder(x)
        z = q.rsample()
        return z

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

    def pca_prior(self, n_samples):
        """
        Given a number of samples, get the PCA from the sampling of the prior distribution.
        """
        samples = self.prior().sample(torch.Size([n_samples]))
        pca = PCA(n_components=2)
        return pca.fit_transform(samples.detach().cpu())


class VampPriorVAE(nn.Module):
    """
    Define a VampPrior Variational Autoencoder model.
    """

    def __init__(
        self,
        encoder,
        decoder,
        input_dim,
        K_pseudo_inputs,
        d_z,
        beta=1.0,
        data_loader=None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_dim = input_dim
        self.K = K_pseudo_inputs
        self.d_z = d_z
        self.beta = beta

        # Pseudo-inputs
        if data_loader is not None:
            self.u = self.sample_from_dataloader(data_loader)
        else:
            self.u = nn.Parameter(torch.rand(K_pseudo_inputs, input_dim))

        # Mixing weights
        self.w = nn.Parameter(torch.zeros(K_pseudo_inputs))

    def sample_from_dataloader(self, data_loader):
        all_data = []
        for batch in data_loader:
            x = batch[0]
            all_data.append(x)
            if len(all_data) * x.shape[0] >= self.K:
                break

        all_data = torch.cat(all_data, dim=0)[: self.K]
        return nn.Parameter(all_data.clone().detach().requires_grad_(True))

    def sample(self, n_samples):
        # encode pseudo inputs
        mu_vp, var_vp = self.encoder.encode(self.u)
        std_vp = torch.sqrt(torch.nn.functional.softplus(var_vp) + 1e-8)

        # mixing probabilities
        w = self.w.view(-1)

        mog = MixtureOfGaussians(w, mu_vp, std_vp)

        return mog.rsample(sample_shape=torch.Size([n_samples]))

    def pca_prior(self, n_samples):
        """
        Given a number of samples, get the PCA from the sampling of the prior distribution.
        """
        samples = self.sample(n_samples)
        pca = PCA(n_components=2)
        return pca.fit_transform(samples.detach().cpu())

    def get_posterior(self, x):
        """
        Given a set of points, compute the posterior distribution.

        Parameters:
        x: [torch.Tensor]
            Samples to pass to the encoder
        """
        q = self.encoder(x)
        z = q.rsample()
        return z

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

    def log_prob(self, z):
        mu_vp, var_vp = self.encoder.encode(self.u)
        std_vp = torch.sqrt(torch.nn.functional.softplus(var_vp) + 1e-8)

        # mixing probabilities
        w = self.w.view(-1)

        # mog definition (VampPrior has the same form as the MoG class)
        mog = MixtureOfGaussians(w, mu_vp, std_vp)

        return mog.log_prob(z)

    def forward(self, x):
        # Obtain posterior and sample
        q_zx = self.encoder(x)
        z = q_zx.rsample()

        # Forward pass to the decoder
        p_xz = self.decoder(z)

        # Loss function
        log_q_zx = q_zx.log_prob(z)
        log_p_z = self.log_prob(z)

        recon_term = p_xz.log_prob(x).mean()
        kl_term = self.beta * (log_q_zx - log_p_z).mean()

        return -recon_term + kl_term


class VampPriorMixtureVAE(nn.Module):
    """
    Define a VampPrior Variational Autoencoder model.
    """

    def __init__(
        self,
        encoder,
        decoder,
        input_dim,
        n_comps,
        d_z,
        beta=1.0,
        data_loader=None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_dim = input_dim
        self.K = n_comps
        self.d_z = d_z
        self.beta = beta

        # Pseudo-inputs
        if data_loader is not None:
            self.u = self.sample_from_dataloader(data_loader)
        else:
            self.u = nn.Parameter(torch.rand(n_comps, input_dim))

        # MoG prior parameters besides location
        self.prior_var = nn.Parameter(torch.randn(n_comps, self.d_z))
        self.prior_pi = nn.Parameter(torch.zeros(n_comps))

    def sample_from_dataloader(self, data_loader):
        all_data = []
        bio_label = []
        for batch in data_loader:
            x = batch[0]
            z = batch[2]  # biological variability
            all_data.append(x)
            bio_label.append(z)
            if len(all_data) * x.shape[0] >= self.K:
                break

        all_data = torch.cat(all_data, dim=0)
        bio_label = torch.cat(bio_label, dim=0)
        bio_dict = torch.unique(bio_label, dim=0)

        selected_u = []

        for label in bio_dict:
            for i in range(all_data.shape[0]):
                if torch.equal(bio_label[i], label):
                    selected_u.append(all_data[i])
                    break

        selected_u = torch.stack(selected_u)
        return nn.Parameter(selected_u.clone().detach().requires_grad_(True))

    def get_prior(self):
        # encode pseudo inputs
        mog_u = self.encoder(self.u)

        # sample from encoded distribution to compute prior components centroids
        mu_u = mog_u.rsample()

        # compute prior
        w = self.prior_pi.view(-1)
        stds = torch.sqrt(torch.nn.functional.softplus(self.prior_var) + 1e-8)
        prior = MixtureOfGaussians(w, mu_u, stds)

        return prior

    def sample(self, n_samples):

        prior = self.get_prior()

        return prior.rsample(sample_shape=torch.Size([n_samples]))

    def pca_prior(self, n_samples):
        """
        Given a number of samples, get the PCA from the sampling of the prior distribution.
        """
        samples = self.sample(n_samples)
        pca = PCA(n_components=2)
        return pca.fit_transform(samples.detach().cpu())

    def get_posterior(self, x):
        """
        Given a set of points, compute the posterior distribution.

        Parameters:
        x: [torch.Tensor]
            Samples to pass to the encoder
        """
        q = self.encoder(x)
        z = q.rsample()
        return z

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

    def log_prob(self, z):

        prior = self.get_prior()

        return prior.log_prob(z)

    def forward(self, x):
        # Obtain posterior and sample
        q_zx = self.encoder(x)
        z = q_zx.rsample()

        # Forward pass to the decoder
        p_xz = self.decoder(z)

        # Loss function
        log_q_zx = q_zx.log_prob(z)
        log_p_z = self.log_prob(z)

        recon_term = p_xz.log_prob(x).mean()
        kl_term = self.beta * (log_q_zx - log_p_z).mean()

        return -recon_term + kl_term


# ---------- SELF ATTENTION CLASSES ---------- #


class GeneAttention(nn.Module):

    def __init__(self, feature_dim, heads=8):
        super().__init__()
        self.feature_dim = feature_dim

        # heads: Number of attention heads (8 by default)
        self.heads = heads

        # Dimension per head
        self.head_dim = feature_dim // heads

        # Projection Layers:
        # self.q: Projects input to "query" space (what features am I looking for?)
        # self.k: Projects input to "key" space (what features do I contain?)
        # self.v: Projects input to "value" space (what information should be passed?)
        # self.out: Final projection that combines attended information
        # Query, Key, Value projections
        self.q = nn.Linear(feature_dim, feature_dim)
        self.k = nn.Linear(feature_dim, feature_dim)
        self.v = nn.Linear(feature_dim, feature_dim)
        self.out = nn.Linear(feature_dim, feature_dim)

        self.scale = self.head_dim**-0.5

    def forward(self, x):
        batch_size = x.size(0)
        # Step-by-Step Process:

        # 1.Projections:
        # Each feature vector is projected into query, key, and value spaces.
        # In a gene expression context, this transforms each gene's expression value into representations that capture different aspects of its significance.

        # 2.Multi-head Reshaping:
        # Input is split into multiple "heads" (8 in our implementation)
        # Each head can learn different types of relationships between genes
        # For example, one head might focus on immune-related genes, another on cell cycle genes

        # 3.Attention Score Calculation:
        # Computes dot product between all queries and keys
        # For genes, this means calculating how relevant each gene is to every other gene
        # The scale factor (1/√head_dim) prevents gradients from becoming too small

        # Project to queries, keys, values
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Softmax Converts scores to attention weights that sum to 1
        # Each gene now has a probability distribution showing which other genes it should "pay attention to"
        attn = F.softmax(scores, dim=-1)

        # Apply attention
        # Each gene's new representation is a weighted sum of all genes' values
        # Weights come from the attention scores
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.feature_dim)

        # Reshapes and projects the attended information back to the original feature space
        out = self.out(out)

        return out


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


class InputOutputDiscriminator(nn.Module):
    """
    Define a Discriminator that can tell decoder's output from input data apart.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        pred = self.net(x)
        return pred

    def loss(self, pred, true):
        loss = nn.CrossEntropyLoss()

        return loss(pred, true)


class BiologicalConservationClassifier(nn.Module):
    """
    Define a classifier in order to conserve biological relevant information.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        pred = self.net(x)
        return pred

    def loss(self, pred, true):
        loss = nn.CrossEntropyLoss()

        return loss(pred, true)


# ---------- TRAINING LOOP ---------- #


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()

    total_steps = len(data_loader) * epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            x = x[0].to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(
                loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}"
            )
            progress_bar.update()


def adversarial_loss(pred_batch, real_batch, batch_size, loss_type="CrossEntropy"):

    if loss_type == "CrossEntropy":
        loss = nn.CrossEntropyLoss()  # Uniform distribution could work
        return -loss(pred_batch, real_batch)

    elif loss_type == "Uniform":
        loss = nn.KLDivLoss(reduction="batchmean")
        target_batch = torch.full_like(pred_batch, 1.0 / batch_size)
        return loss(torch.log_softmax(pred_batch, dim=1), target_batch)


def train_abaco(
    vae,
    vae_optim,
    discriminator,
    disc_optim,
    adv_optim,
    bio_classifier,
    bio_optim,
    data_loader,
    epochs,
    device,
    w_disc=1.0,
    w_adv=1.0,
    w_bio=1.0,
    w_elbo=1.0,
    disc_loss_type="CrossEntropy",
):

    vae.train()
    discriminator.train()

    total_steps = len(data_loader) * epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for loader_data in data_iter:
            x = loader_data[0].to(device)
            y = loader_data[1].to(device).float()
            z = loader_data[2].to(device).float()

            # First step: Forward pass through encoder and to discriminator
            with torch.no_grad():
                q_disc = vae.encoder(x)
                z_disc = q_disc.rsample()

            # Detach z
            z_disc = z_disc.detach()

            # Discriminator Loss
            disc_optim.zero_grad()
            b_pred_disc = discriminator(z_disc)
            disc_loss = w_disc * discriminator.loss(b_pred_disc, y)
            disc_loss.backward()
            disc_optim.step()

            # Second step: Compute adversarial loss to encoder - second pass with gradient

            # Adversarial loss
            adv_optim.zero_grad()
            q_adv = vae.encoder(x)
            z_adv = q_adv.rsample()

            b_pred_adv = discriminator(z_adv)

            adv_loss = w_adv * adversarial_loss(
                pred_batch=b_pred_adv,
                real_batch=y,
                batch_size=y.shape[1],
                loss_type=disc_loss_type,
            )

            adv_loss.backward()
            adv_optim.step()

            # Detach elements
            z_adv = z_adv.detach()

            # Third step: Forward pass to biological classifier

            bio_optim.zero_grad()
            q_bio = vae.encoder(x)
            z_bio = q_bio.rsample()
            bio_pred_class = bio_classifier(z_bio)
            bio_loss = w_bio * bio_classifier.loss(bio_pred_class, z)
            bio_loss.backward()
            bio_optim.step()

            # Detach elements
            z_bio = z_bio.detach()

            # Forth step: VAE ELBO computation
            vae_optim.zero_grad()
            vae_loss = w_elbo * vae(x)
            vae_loss.backward()
            vae_optim.step()

            # Update progress bar
            progress_bar.set_postfix(
                vae_loss=f"{vae_loss.item():12.4f}",
                disc_loss=f"{disc_loss.item():12.4f}",
                adv_loss=f"{adv_loss.item():12.4f}",
                bio_loss=f"{bio_loss.item():12.4f}",
                epoch=f"{epoch+1}/{epochs}",
            )
            progress_bar.update()


def train_abaco_dual_batch(
    vae,
    vae_optim,
    discriminator,
    disc_optim,
    adv_optim,
    data_loader,
    epochs,
    device,
    w_disc=1.0,
    w_adv=1.0,
    disc_loss_type="CrossEntropy",
):

    vae.train()
    discriminator.train()

    total_steps = len(data_loader) * epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for loader_data in data_iter:
            x = loader_data[0].to(device)
            y = loader_data[1].to(device).float()

            # First step: Forward pass through encoder and to discriminator
            with torch.no_grad():
                q = vae.encoder(x)
                z = q.rsample()

            # Detach z
            z = z.detach()

            # Discriminator Loss
            disc_optim.zero_grad()
            b_pred = discriminator(z)
            disc_loss = w_disc * discriminator.loss(b_pred, y)
            disc_loss.backward()
            disc_optim.step()

            # Second step: Compute adversarial loss to encoder - second pass with gradient

            # Adversarial loss
            adv_optim.zero_grad()
            q = vae.encoder(x)
            z = q.rsample()

            b_pred = discriminator(z)

            adv_loss = w_adv * adversarial_loss(
                pred_batch=b_pred,
                real_batch=y,
                batch_size=y.shape[1],
                loss_type=disc_loss_type,
            )
            adv_loss.backward()
            adv_optim.step()

            # Third step: VAE ELBO computation
            vae_optim.zero_grad()
            vae_loss = vae(x)
            vae_loss.backward()
            vae_optim.step()

            # Update progress bar
            progress_bar.set_postfix(
                vae_loss=f"{vae_loss.item():12.4f}",
                disc_loss=f"{disc_loss.item():12.4f}",
                adv_loss=f"{adv_loss.item():12.4f}",
                epoch=f"{epoch+1}/{epochs}",
            )
            progress_bar.update()


# Other functions


def contour_plot(samples, n_levels=10, x_offset=5, y_offset=5, alpha=0.8):
    """
    Given an array computes the contour plot

    Parameters:
        samples: [np.array]
            An array with (,2) dimensions
    """
    x = samples[:, 0]
    y = samples[:, 1]

    x_min, x_max = x.min() - x_offset, x.max() + x_offset
    y_min, y_max = y.min() - y_offset, y.max() + y_offset
    x_grid = np.linspace(x_min, x_max, 500)
    y_grid = np.linspace(y_min, y_max, 500)
    X, Y = np.meshgrid(x_grid, y_grid)

    kde = gaussian_kde(samples.T)
    Z = kde(np.vstack([X.ravel(), Y.ravel()]))
    Z = Z.reshape(X.shape)

    contour = plt.contourf(X, Y, Z, levels=n_levels, alpha=alpha)

    return contour


def log_normal_diag(x, mu, log_var, reduction=None, dim=None):
    # Compute constant term outside the exponentiation
    const = torch.log(torch.tensor(2 * math.pi, device=x.device))

    # Compute log probability
    log_p = -0.5 * (const + log_var + torch.exp(-log_var) * (x - mu) ** 2)

    if dim is None:
        dim = -1
    log_p = log_p.sum(dim=dim)

    if reduction == "avg":
        return torch.mean(log_p, dim)
    elif reduction == "sum":
        return torch.sum(log_p, dim)
    else:
        return log_p
