# Essentials
import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from umap import UMAP
import random
import seaborn as sns

# User libraries
from BatchEffectDataLoader import (
    DataPreprocess,
    DataTransform,
    ABaCoDataLoader,
    one_hot_encoding,
    class_to_int,
)
from BatchEffectCorrection import correctCombat, correctLimma_rBE
from BatchEffectPlots import plotPCA, plotPCoA
from BatchEffectMetrics import kBET, iLISI, cLISI, ARI, ASW
from MetaABaCo import (
    NormalPrior,
    NormalEncoder,
    MoGPrior,
    MoGEncoder,
    ZINBDecoder,
    VAE,
    train,
    MixtureOfGaussians,
    BatchDiscriminator,
    BiologicalConservationClassifier,
    train_abaco,
    train_abaco_two_step,
    contour_plot,
    GeneAttention,
    AttentionMoGEncoder,
    AttentionZINBDecoder,
    VampPriorVAE,
    VampPriorMixtureVAE,
    train_abaco_contra,
    train_abaco_adversarial,
    train_abaco_adversarial_2,
    train_c_abaco_contra,
    train_c_abaco_contra_modified,
    train_c_abaco_adversarial_2,
    train_c_abaco_adversarial_2_modified,
    VampPriorMixtureConditionalVAE,
    LatentProjector,
    abaco_drl,
    abaco_drl_2,
    train_c_abaco_batch_masking,
)

input_size = 435
d_z = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = "data/MGnify/IBD/IBD_dataset.csv"
batch_label = "project ID"
sample_label = "run ID"
exp_label = "associated phenotype"

data = DataPreprocess(path, factors=[sample_label, batch_label, exp_label])

# train DataLoader: [samples, ohe_batch]
train_dataloader = DataLoader(
    TensorDataset(
        torch.tensor(
            data.select_dtypes(include="number").values, dtype=torch.float32
        ),  # samples
        one_hot_encoding(data[batch_label])[0],  # one hot encoded batch information
        one_hot_encoding(data[exp_label])[0],  # one hot encoded biological information
    ),
    batch_size=1000,
)

train_batch_dataloader = DataLoader(
    TensorDataset(
        torch.tensor(
            data.select_dtypes(include="number").values, dtype=torch.float32
        ),  # samples
        one_hot_encoding(data[exp_label])[0],  # one hot encoded batch information
        one_hot_encoding(data[batch_label])[
            0
        ],  # one hot encoded biological information
    ),
    batch_size=1000,
)

# Several run framework - 2000 epochs to 2000 epochs with KL divergence cycle loss

iterations = 50
performance = []

# Set random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Start training loop
for iter_index in tqdm(range(iterations), desc="ABaCo iteration"):

    # Model parameters
    K = 3
    n_batches = 2

    # Encoder
    mog_encoder_net = nn.Sequential(
        nn.Linear(input_size + n_batches, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 258),
        nn.ReLU(),
        nn.Linear(258, K * (2 * d_z + 1)),
    )

    # Decoder
    zinb_decoder_net = nn.Sequential(
        nn.Linear(d_z + n_batches, 258),
        nn.ReLU(),
        nn.Linear(258, 512),
        nn.ReLU(),
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 3 * input_size),
    )

    # Defining VMM model
    encoder = MoGEncoder(mog_encoder_net, n_comp=K)
    decoder = ZINBDecoder(zinb_decoder_net)
    bio_vae_model = VampPriorMixtureConditionalVAE(
        encoder=encoder,
        decoder=decoder,
        input_dim=input_size,
        n_comps=K,
        batch_dim=n_batches,
        d_z=d_z,
        beta=20.0,
        data_loader=train_dataloader,
    ).to(device)

    # Batch discriminator
    batch_latent_disc_net = nn.Sequential(
        nn.Linear(d_z + K, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, n_batches),
    )

    # Defining batch discriminator
    discriminator_latent = BatchDiscriminator(batch_latent_disc_net).to(device)

    # Optimizers

    vae_optim_pre = torch.optim.Adam(
        [
            {"params": bio_vae_model.encoder.parameters()},
            {"params": bio_vae_model.decoder.parameters()},
            {
                "params": [
                    bio_vae_model.u,
                    bio_vae_model.prior_pi,
                    bio_vae_model.prior_var,
                ]
            },
        ],
        lr=1e-3,
    )

    disc_latent_optim = torch.optim.Adam(discriminator_latent.parameters(), lr=1e-5)

    adv_latent_optim = torch.optim.Adam(bio_vae_model.encoder.parameters(), lr=1e-5)

    epochs_pre = 2000
    train_c_abaco_contra_modified(
        vae=bio_vae_model,
        vae_optim_pre=vae_optim_pre,
        discriminator=discriminator_latent,
        disc_optim=disc_latent_optim,
        adv_optim=adv_latent_optim,
        data_loader=train_dataloader,
        epochs=epochs_pre,
        device=device,
        w_elbo=1.0,
        w_contra=100.0,
        temp=0.1,
        w_adv=1.0,
        w_disc=1.0,
        disc_loss_type="CrossEntropy",
    )

    # Define new optimizer only for decoder
    vae_optim_post = torch.optim.Adam(
        [
            {"params": bio_vae_model.decoder.parameters()},
        ],
        lr=1e-4,
    )

    # Training run
    epochs_post = 2000
    train_c_abaco_batch_masking(
        vae=bio_vae_model,
        vae_optim_post=vae_optim_post,
        data_loader=train_dataloader,
        epochs=epochs_post,
        device=device,
        w_elbo=1.0,
        w_cycle=0.1,
        cycle="KL",
    )

    ohe_batch = one_hot_encoding(data[batch_label])[0]

    # Reconstructing data with trained model
    recon_data = []

    for x in train_batch_dataloader:
        x = x[0].to(device)
        encoded = bio_vae_model.encoder(torch.cat([x, ohe_batch.to(device)], dim=1))
        z = encoded.rsample()
        decoded = bio_vae_model.decoder(
            torch.cat([z, torch.zeros_like(ohe_batch.to(device))], dim=1)
        )
        recon = decoded.sample()
        recon_data.append(recon)

    np_recon_data = np.vstack([t.detach().cpu().numpy() for t in recon_data])

    otu_corrected_pd = pd.concat(
        [
            pd.DataFrame(
                np_recon_data,
                index=data.index,
                columns=data.select_dtypes("number").columns,
            ),
            data[batch_label],
            data[exp_label],
            data[sample_label],
        ],
        axis=1,
    )

    # Normalize for comparison
    norm_corrected_data_abaco = DataTransform(
        otu_corrected_pd,
        factors=[sample_label, batch_label, exp_label],
        transformation="CLR",
        count=True,
    )

    # Compute metrics and save
    kbet = kBET(norm_corrected_data_abaco, batch_label=batch_label)
    ilisi = iLISI(norm_corrected_data_abaco, batch_label=batch_label)
    ari = ARI(norm_corrected_data_abaco, bio_label=exp_label, n_clusters=3)

    clisi = cLISI(norm_corrected_data_abaco, cell_label=exp_label)
    asw = ASW(norm_corrected_data_abaco, interest_label=exp_label)

    performance.append(
        {
            "iter": iter_index,
            "kBET": kbet,
            "iLISI": ilisi,
            "ARI": ari,
            "cLISI": clisi,
            "ASW": asw,
        }
    )

performance_df = pd.DataFrame(performance)
performance_df.to_csv(
    f"performance_metrics/IBD/{epochs_pre}_epochs_pre_{epochs_post}_epochs_post_KL_cycle.csv",
    index=False,
)

# Several run framework - 2000 epochs to 2000 epochs with no cycle loss

iterations = 50
performance = []

# Set random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Start training loop
for iter_index in tqdm(range(iterations), desc="ABaCo iteration"):

    # Model parameters
    K = 3
    n_batches = 2

    # Encoder
    mog_encoder_net = nn.Sequential(
        nn.Linear(input_size + n_batches, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 258),
        nn.ReLU(),
        nn.Linear(258, K * (2 * d_z + 1)),
    )

    # Decoder
    zinb_decoder_net = nn.Sequential(
        nn.Linear(d_z + n_batches, 258),
        nn.ReLU(),
        nn.Linear(258, 512),
        nn.ReLU(),
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 3 * input_size),
    )

    # Defining VMM model
    encoder = MoGEncoder(mog_encoder_net, n_comp=K)
    decoder = ZINBDecoder(zinb_decoder_net)
    bio_vae_model = VampPriorMixtureConditionalVAE(
        encoder=encoder,
        decoder=decoder,
        input_dim=input_size,
        n_comps=K,
        batch_dim=n_batches,
        d_z=d_z,
        beta=20.0,
        data_loader=train_dataloader,
    ).to(device)

    # Batch discriminator
    batch_latent_disc_net = nn.Sequential(
        nn.Linear(d_z + K, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, n_batches),
    )

    # Defining batch discriminator
    discriminator_latent = BatchDiscriminator(batch_latent_disc_net).to(device)

    # Optimizers

    vae_optim_pre = torch.optim.Adam(
        [
            {"params": bio_vae_model.encoder.parameters()},
            {"params": bio_vae_model.decoder.parameters()},
            {
                "params": [
                    bio_vae_model.u,
                    bio_vae_model.prior_pi,
                    bio_vae_model.prior_var,
                ]
            },
        ],
        lr=1e-3,
    )

    disc_latent_optim = torch.optim.Adam(discriminator_latent.parameters(), lr=1e-5)

    adv_latent_optim = torch.optim.Adam(bio_vae_model.encoder.parameters(), lr=1e-5)

    epochs_pre = 2000
    train_c_abaco_contra_modified(
        vae=bio_vae_model,
        vae_optim_pre=vae_optim_pre,
        discriminator=discriminator_latent,
        disc_optim=disc_latent_optim,
        adv_optim=adv_latent_optim,
        data_loader=train_dataloader,
        epochs=epochs_pre,
        device=device,
        w_elbo=1.0,
        w_contra=100.0,
        temp=0.1,
        w_adv=1.0,
        w_disc=1.0,
        disc_loss_type="CrossEntropy",
    )

    # Define new optimizer only for decoder
    vae_optim_post = torch.optim.Adam(
        [
            {"params": bio_vae_model.decoder.parameters()},
        ],
        lr=1e-4,
    )

    # Training run
    epochs_post = 2000
    train_c_abaco_batch_masking(
        vae=bio_vae_model,
        vae_optim_post=vae_optim_post,
        data_loader=train_dataloader,
        epochs=epochs_post,
        device=device,
        w_elbo=1.0,
        w_cycle=0.1,
        cycle="None",
    )

    ohe_batch = one_hot_encoding(data[batch_label])[0]

    # Reconstructing data with trained model
    recon_data = []

    for x in train_batch_dataloader:
        x = x[0].to(device)
        encoded = bio_vae_model.encoder(torch.cat([x, ohe_batch.to(device)], dim=1))
        z = encoded.rsample()
        decoded = bio_vae_model.decoder(
            torch.cat([z, torch.zeros_like(ohe_batch.to(device))], dim=1)
        )
        recon = decoded.sample()
        recon_data.append(recon)

    np_recon_data = np.vstack([t.detach().cpu().numpy() for t in recon_data])

    otu_corrected_pd = pd.concat(
        [
            pd.DataFrame(
                np_recon_data,
                index=data.index,
                columns=data.select_dtypes("number").columns,
            ),
            data[batch_label],
            data[exp_label],
            data[sample_label],
        ],
        axis=1,
    )

    # Normalize for comparison
    norm_corrected_data_abaco = DataTransform(
        otu_corrected_pd,
        factors=[sample_label, batch_label, exp_label],
        transformation="CLR",
        count=True,
    )

    # Compute metrics and save
    kbet = kBET(norm_corrected_data_abaco, batch_label=batch_label)
    ilisi = iLISI(norm_corrected_data_abaco, batch_label=batch_label)
    ari = ARI(norm_corrected_data_abaco, bio_label=exp_label, n_clusters=3)

    clisi = cLISI(norm_corrected_data_abaco, cell_label=exp_label)
    asw = ASW(norm_corrected_data_abaco, interest_label=exp_label)

    performance.append(
        {
            "iter": iter_index,
            "kBET": kbet,
            "iLISI": ilisi,
            "ARI": ari,
            "cLISI": clisi,
            "ASW": asw,
        }
    )

performance_df = pd.DataFrame(performance)
performance_df.to_csv(
    f"performance_metrics/IBD/{epochs_pre}_epochs_pre_{epochs_post}_epochs_post_no_cycle.csv",
    index=False,
)
