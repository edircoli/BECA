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
)

# Loading data
input_size = 567
d_z = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = "data/dataset_ad.csv"
batch_label = "batch"
sample_label = "sample"
exp_label = "trt"

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
    batch_size=128,
)

# Several run framework - 5000 epochs to 2000 epochs

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
    # Define model
    # Model parameters
    K = 2
    n_batches = 5

    # Encoder
    mog_encoder_net = nn.Sequential(
        nn.Linear(input_size, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 258),
        nn.ReLU(),
        nn.Linear(258, K * (2 * d_z + 1)),
    )

    # Decoder
    zinb_decoder_net = nn.Sequential(
        nn.Linear(d_z, 258),
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
    vae_model = VampPriorMixtureVAE(
        encoder=encoder,
        decoder=decoder,
        input_dim=input_size,
        n_comps=K,
        d_z=d_z,
        beta=20.0,
        data_loader=train_dataloader,
    ).to(device)

    # Optimizers

    vae_optim_pre = torch.optim.Adam(
        [
            {"params": vae_model.encoder.parameters()},
            {"params": vae_model.decoder.parameters()},
            {"params": [vae_model.u, vae_model.prior_pi, vae_model.prior_var]},
        ],
        lr=3e-3,
    )

    epochs_pre = 5000
    train_abaco_contra(
        vae=vae_model,
        vae_optim_pre=vae_optim_pre,
        data_loader=train_dataloader,
        epochs=epochs_pre,
        device=device,
        w_elbo=1.0,
        w_contra=100.0,
        temp=0.1,
    )

    # Batch discriminator
    batch_disc_net = nn.Sequential(
        nn.Linear(d_z + K, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, n_batches),
    )

    # Defining batch discriminator
    discriminator = BatchDiscriminator(batch_disc_net).to(device)

    # Optimizers
    adv_optim = torch.optim.Adam(vae_model.encoder.parameters(), lr=1e-4)

    vae_optim_post = torch.optim.Adam(
        [
            {"params": vae_model.encoder.parameters()},
            {"params": vae_model.decoder.parameters()},
            #        {"params": [vae_model.u, vae_model.prior_pi, vae_model.prior_var]},
        ],
        lr=1e-4,
    )

    disc_optim = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

    epochs_post = 2000

    train_abaco_adversarial(
        vae=vae_model,
        vae_optim_post=vae_optim_post,
        discriminator=discriminator,
        disc_optim=disc_optim,
        adv_optim=adv_optim,
        data_loader=train_dataloader,
        epochs=epochs_post,
        device=device,
        w_disc=1.0,
        w_adv=1.0,
        w_elbo=1.0,
        w_contra=100.0,
        temp=0.1,
        disc_loss_type="CrossEntropy",
    )

    # Reconstructing data with trained model
    recon_data = []

    for x in train_dataloader:
        x = x[0].to(device)
        encoded = vae_model.encoder(x)
        z = encoded.rsample()
        decoded = vae_model.decoder(z)
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
        factors=["sample", "batch", "trt"],
        transformation="CLR",
        count=True,
    )

    # Compute metrics and save
    kbet = kBET(norm_corrected_data_abaco, batch_label="batch")
    ilisi = (iLISI(norm_corrected_data_abaco, batch_label="batch") - 1) / (
        5 - 1
    )  # iLISI goes from 1 to 5
    ari = ARI(norm_corrected_data_abaco, bio_label="trt", n_clusters=2)

    clisi = (5 - cLISI(norm_corrected_data_abaco, cell_label="trt")) / (5 - 1)
    asw = 1 - ASW(norm_corrected_data_abaco, interest_label="trt")

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
    f"performance_metrics/AD_count_contra/{epochs_pre}_epochs_pre_{epochs_post}_epochs_post.csv",
    index=False,
)

# Several run framework - 5000 epochs to 1000 epochs

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
    # Define model
    # Model parameters
    K = 2
    n_batches = 5

    # Encoder
    mog_encoder_net = nn.Sequential(
        nn.Linear(input_size, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 258),
        nn.ReLU(),
        nn.Linear(258, K * (2 * d_z + 1)),
    )

    # Decoder
    zinb_decoder_net = nn.Sequential(
        nn.Linear(d_z, 258),
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
    vae_model = VampPriorMixtureVAE(
        encoder=encoder,
        decoder=decoder,
        input_dim=input_size,
        n_comps=K,
        d_z=d_z,
        beta=20.0,
        data_loader=train_dataloader,
    ).to(device)

    # Optimizers

    vae_optim_pre = torch.optim.Adam(
        [
            {"params": vae_model.encoder.parameters()},
            {"params": vae_model.decoder.parameters()},
            {"params": [vae_model.u, vae_model.prior_pi, vae_model.prior_var]},
        ],
        lr=3e-3,
    )

    epochs_pre = 5000
    train_abaco_contra(
        vae=vae_model,
        vae_optim_pre=vae_optim_pre,
        data_loader=train_dataloader,
        epochs=epochs_pre,
        device=device,
        w_elbo=1.0,
        w_contra=100.0,
        temp=0.1,
    )

    # Batch discriminator
    batch_disc_net = nn.Sequential(
        nn.Linear(d_z + K, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, n_batches),
    )

    # Defining batch discriminator
    discriminator = BatchDiscriminator(batch_disc_net).to(device)

    # Optimizers
    adv_optim = torch.optim.Adam(vae_model.encoder.parameters(), lr=1e-4)

    vae_optim_post = torch.optim.Adam(
        [
            {"params": vae_model.encoder.parameters()},
            {"params": vae_model.decoder.parameters()},
            #        {"params": [vae_model.u, vae_model.prior_pi, vae_model.prior_var]},
        ],
        lr=1e-4,
    )

    disc_optim = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

    epochs_post = 1000

    train_abaco_adversarial(
        vae=vae_model,
        vae_optim_post=vae_optim_post,
        discriminator=discriminator,
        disc_optim=disc_optim,
        adv_optim=adv_optim,
        data_loader=train_dataloader,
        epochs=epochs_post,
        device=device,
        w_disc=1.0,
        w_adv=1.0,
        w_elbo=1.0,
        w_contra=100.0,
        temp=0.1,
        disc_loss_type="CrossEntropy",
    )

    # Reconstructing data with trained model
    recon_data = []

    for x in train_dataloader:
        x = x[0].to(device)
        encoded = vae_model.encoder(x)
        z = encoded.rsample()
        decoded = vae_model.decoder(z)
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
        factors=["sample", "batch", "trt"],
        transformation="CLR",
        count=True,
    )

    # Compute metrics and save
    kbet = kBET(norm_corrected_data_abaco, batch_label="batch")
    ilisi = (iLISI(norm_corrected_data_abaco, batch_label="batch") - 1) / (
        5 - 1
    )  # iLISI goes from 1 to 5
    ari = ARI(norm_corrected_data_abaco, bio_label="trt", n_clusters=2)

    clisi = (5 - cLISI(norm_corrected_data_abaco, cell_label="trt")) / (5 - 1)
    asw = 1 - ASW(norm_corrected_data_abaco, interest_label="trt")

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
    f"performance_metrics/AD_count_contra/{epochs_pre}_epochs_pre_{epochs_post}_epochs_post.csv",
    index=False,
)

# Several run framework - 5000 epochs to 5000 epochs

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
    # Define model
    # Model parameters
    K = 2
    n_batches = 5

    # Encoder
    mog_encoder_net = nn.Sequential(
        nn.Linear(input_size, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 258),
        nn.ReLU(),
        nn.Linear(258, K * (2 * d_z + 1)),
    )

    # Decoder
    zinb_decoder_net = nn.Sequential(
        nn.Linear(d_z, 258),
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
    vae_model = VampPriorMixtureVAE(
        encoder=encoder,
        decoder=decoder,
        input_dim=input_size,
        n_comps=K,
        d_z=d_z,
        beta=20.0,
        data_loader=train_dataloader,
    ).to(device)

    # Optimizers

    vae_optim_pre = torch.optim.Adam(
        [
            {"params": vae_model.encoder.parameters()},
            {"params": vae_model.decoder.parameters()},
            {"params": [vae_model.u, vae_model.prior_pi, vae_model.prior_var]},
        ],
        lr=3e-3,
    )

    epochs_pre = 5000
    train_abaco_contra(
        vae=vae_model,
        vae_optim_pre=vae_optim_pre,
        data_loader=train_dataloader,
        epochs=epochs_pre,
        device=device,
        w_elbo=1.0,
        w_contra=100.0,
        temp=0.1,
    )

    # Batch discriminator
    batch_disc_net = nn.Sequential(
        nn.Linear(d_z + K, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, n_batches),
    )

    # Defining batch discriminator
    discriminator = BatchDiscriminator(batch_disc_net).to(device)

    # Optimizers
    adv_optim = torch.optim.Adam(vae_model.encoder.parameters(), lr=1e-4)

    vae_optim_post = torch.optim.Adam(
        [
            {"params": vae_model.encoder.parameters()},
            {"params": vae_model.decoder.parameters()},
            #        {"params": [vae_model.u, vae_model.prior_pi, vae_model.prior_var]},
        ],
        lr=1e-4,
    )

    disc_optim = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

    epochs_post = 5000

    train_abaco_adversarial(
        vae=vae_model,
        vae_optim_post=vae_optim_post,
        discriminator=discriminator,
        disc_optim=disc_optim,
        adv_optim=adv_optim,
        data_loader=train_dataloader,
        epochs=epochs_post,
        device=device,
        w_disc=1.0,
        w_adv=1.0,
        w_elbo=1.0,
        w_contra=100.0,
        temp=0.1,
        disc_loss_type="CrossEntropy",
    )

    # Reconstructing data with trained model
    recon_data = []

    for x in train_dataloader:
        x = x[0].to(device)
        encoded = vae_model.encoder(x)
        z = encoded.rsample()
        decoded = vae_model.decoder(z)
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
        factors=["sample", "batch", "trt"],
        transformation="CLR",
        count=True,
    )

    # Compute metrics and save
    kbet = kBET(norm_corrected_data_abaco, batch_label="batch")
    ilisi = (iLISI(norm_corrected_data_abaco, batch_label="batch") - 1) / (
        5 - 1
    )  # iLISI goes from 1 to 5
    ari = ARI(norm_corrected_data_abaco, bio_label="trt", n_clusters=2)

    clisi = (5 - cLISI(norm_corrected_data_abaco, cell_label="trt")) / (5 - 1)
    asw = 1 - ASW(norm_corrected_data_abaco, interest_label="trt")

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
    f"performance_metrics/AD_count_contra/{epochs_pre}_epochs_pre_{epochs_post}_epochs_post.csv",
    index=False,
)
