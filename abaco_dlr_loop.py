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
from src.ABaCo.BatchEffectDataLoader import (
    DataPreprocess,
    DataTransform,
    ABaCoDataLoader,
    one_hot_encoding,
    class_to_int,
)
from src.ABaCo.BatchEffectCorrection import correctCombat, correctLimma_rBE
from src.ABaCo.BatchEffectPlots import plotPCA, plotPCoA
from src.ABaCo.BatchEffectMetrics import kBET, iLISI, cLISI, ARI, ASW
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
)

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
    batch_size=128,
)

# Several run framework - 2000 epochs to 500 epochs

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

    # Define biological informed model
    # Model parameters

    K = 2
    n_batches = 5

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

    disc_latent_optim = torch.optim.Adam(discriminator_latent.parameters(), lr=5e-6)

    adv_latent_optim = torch.optim.Adam(bio_vae_model.encoder.parameters(), lr=5e-6)

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

    # Define batch informed model

    # Model parameters
    K = 5
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
    batch_vae_model = VampPriorMixtureConditionalVAE(
        encoder=encoder,
        decoder=decoder,
        input_dim=input_size,
        n_comps=K,
        batch_dim=n_batches,
        d_z=d_z,
        beta=20.0,
        data_loader=train_batch_dataloader,
    ).to(device)

    # Batch discriminator
    bio_latent_disc_net = nn.Sequential(
        nn.Linear(d_z + K, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, n_batches),
    )

    # Defining batch discriminator
    discriminator_latent = BatchDiscriminator(bio_latent_disc_net).to(device)

    # Optimizers

    vae_optim_pre = torch.optim.Adam(
        [
            {"params": batch_vae_model.encoder.parameters()},
            {"params": batch_vae_model.decoder.parameters()},
            {
                "params": [
                    batch_vae_model.u,
                    batch_vae_model.prior_pi,
                    batch_vae_model.prior_var,
                ]
            },
        ],
        lr=1e-3,
    )

    disc_latent_optim = torch.optim.Adam(discriminator_latent.parameters(), lr=5e-6)

    adv_latent_optim = torch.optim.Adam(bio_vae_model.encoder.parameters(), lr=5e-6)

    train_c_abaco_contra_modified(
        vae=batch_vae_model,
        vae_optim_pre=vae_optim_pre,
        discriminator=discriminator_latent,
        disc_optim=disc_latent_optim,
        adv_optim=adv_latent_optim,
        data_loader=train_batch_dataloader,
        epochs=epochs_pre,
        device=device,
        w_elbo=1.0,
        w_contra=100.0,
        temp=0.1,
        w_adv=1.0,
        w_disc=1.0,
        disc_loss_type="CrossEntropy",
    )

    # Disentanglement representation learning !

    # Latent space projectors

    projected_dim = 16

    bio_projector_net = nn.Sequential(
        nn.Linear(d_z, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, projected_dim),
    )

    batch_projector_net = nn.Sequential(
        nn.Linear(d_z, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, projected_dim),
    )

    bio_projector = LatentProjector(net=bio_projector_net).to(device)
    batch_projector = LatentProjector(net=batch_projector_net).to(device)

    projectors_optim = torch.optim.Adam(
        [
            {"params": bio_projector.parameters()},
            {"params": batch_projector.parameters()},
        ],
        lr=1e-4,
    )

    # Bio projector batch discriminator

    n_bios = 2
    n_batches = 5

    batch_disc_net = nn.Sequential(
        nn.Linear(projected_dim + n_bios, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, n_batches),
    )

    batch_discriminator = BatchDiscriminator(net=batch_disc_net).to(device)

    batch_disc_optim = torch.optim.Adam(batch_discriminator.parameters(), lr=1e-4)
    batch_adv_optim = torch.optim.Adam(bio_projector.parameters(), lr=1e-4)

    # Biological classifier

    bio_classifier_net = nn.Sequential(
        nn.Linear(projected_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, n_bios),
    )

    bio_classifier = BiologicalConservationClassifier(net=bio_classifier_net).to(device)

    bio_classifier_optim = torch.optim.Adam(bio_classifier.parameters(), lr=1e-4)

    # Decoders

    full_decoder_net = nn.Sequential(
        nn.Linear(2 * projected_dim, 258),
        nn.ReLU(),
        nn.Linear(258, 512),
        nn.ReLU(),
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 3 * input_size),
    )

    bio_decoder_net = nn.Sequential(
        nn.Linear(projected_dim, 258),
        nn.ReLU(),
        nn.Linear(258, 512),
        nn.ReLU(),
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 3 * input_size),
    )

    full_decoder = ZINBDecoder(full_decoder_net).to(device)

    bio_decoder = ZINBDecoder(bio_decoder_net).to(device)

    full_decoder_optim = torch.optim.Adam(full_decoder.parameters(), lr=1e-4)
    bio_decoder_optim = torch.optim.Adam(bio_decoder.parameters(), lr=1e-4)

    epochs_post = 500

    abaco_drl_2(
        # training params
        epochs=epochs_post,
        data_loader=train_dataloader,
        data_batch_loader=train_batch_dataloader,
        device=device,
        # pre-trained VAEs
        frozen_bio_vae=bio_vae_model,
        frozen_batch_vae=batch_vae_model,
        # projectors
        bio_projector=bio_projector,
        batch_projector=batch_projector,
        projectors_optim=projectors_optim,
        # discriminator
        batch_discriminator=batch_discriminator,
        batch_disc_optim=batch_disc_optim,
        batch_adv_optim=batch_adv_optim,
        # bio classifier
        bio_classifier=bio_classifier,
        bio_classifier_optim=bio_classifier_optim,
        # decoders
        full_decoder=full_decoder,
        full_decoder_optim=full_decoder_optim,
        bio_decoder=bio_decoder,
        bio_decoder_optim=bio_decoder_optim,
        # default settings
        w_projector=1.0,
        w_disc_batch=1.0,
        w_adv_batch=1.0,
        w_bio_class=1.0,
        w_full_dec=1.0,
        w_bio_dec=1.0,
        disc_loss_type="CrossEntropy",
    )

    # Reconstructing data with trained model
    ohe_bio = one_hot_encoding(data["trt"])[0]
    ohe_batch = one_hot_encoding(data["batch"])[0]

    # Reconstructing data with trained model
    recon_data = []

    for x in train_dataloader:
        x = x[0].to(device)
        encoded = bio_vae_model.encoder(torch.cat([x, ohe_batch.to(device)], dim=1))
        z = encoded.rsample()
        s = bio_projector(z)
        decoded = bio_decoder(s)
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
    f"performance_metrics/AD_count_drl/{epochs_pre}_epochs_pre_{epochs_post}_epochs_post.csv",
    index=False,
)

# -------------------------------------------------------------------------------------------------------

# Several run framework - 2000 epochs to 1000 epochs

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

    # Define biological informed model
    # Model parameters

    K = 2
    n_batches = 5

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

    disc_latent_optim = torch.optim.Adam(discriminator_latent.parameters(), lr=5e-6)

    adv_latent_optim = torch.optim.Adam(bio_vae_model.encoder.parameters(), lr=5e-6)

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

    # Define batch informed model

    # Model parameters
    K = 5
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
    batch_vae_model = VampPriorMixtureConditionalVAE(
        encoder=encoder,
        decoder=decoder,
        input_dim=input_size,
        n_comps=K,
        batch_dim=n_batches,
        d_z=d_z,
        beta=20.0,
        data_loader=train_batch_dataloader,
    ).to(device)

    # Batch discriminator
    bio_latent_disc_net = nn.Sequential(
        nn.Linear(d_z + K, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, n_batches),
    )

    # Defining batch discriminator
    discriminator_latent = BatchDiscriminator(bio_latent_disc_net).to(device)

    # Optimizers

    vae_optim_pre = torch.optim.Adam(
        [
            {"params": batch_vae_model.encoder.parameters()},
            {"params": batch_vae_model.decoder.parameters()},
            {
                "params": [
                    batch_vae_model.u,
                    batch_vae_model.prior_pi,
                    batch_vae_model.prior_var,
                ]
            },
        ],
        lr=1e-3,
    )

    disc_latent_optim = torch.optim.Adam(discriminator_latent.parameters(), lr=5e-6)

    adv_latent_optim = torch.optim.Adam(bio_vae_model.encoder.parameters(), lr=5e-6)

    train_c_abaco_contra_modified(
        vae=batch_vae_model,
        vae_optim_pre=vae_optim_pre,
        discriminator=discriminator_latent,
        disc_optim=disc_latent_optim,
        adv_optim=adv_latent_optim,
        data_loader=train_batch_dataloader,
        epochs=epochs_pre,
        device=device,
        w_elbo=1.0,
        w_contra=100.0,
        temp=0.1,
        w_adv=1.0,
        w_disc=1.0,
        disc_loss_type="CrossEntropy",
    )

    # Disentanglement representation learning !

    # Latent space projectors

    projected_dim = 16

    bio_projector_net = nn.Sequential(
        nn.Linear(d_z, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, projected_dim),
    )

    batch_projector_net = nn.Sequential(
        nn.Linear(d_z, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, projected_dim),
    )

    bio_projector = LatentProjector(net=bio_projector_net).to(device)
    batch_projector = LatentProjector(net=batch_projector_net).to(device)

    projectors_optim = torch.optim.Adam(
        [
            {"params": bio_projector.parameters()},
            {"params": batch_projector.parameters()},
        ],
        lr=1e-4,
    )

    # Bio projector batch discriminator

    n_bios = 2
    n_batches = 5

    batch_disc_net = nn.Sequential(
        nn.Linear(projected_dim + n_bios, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, n_batches),
    )

    batch_discriminator = BatchDiscriminator(net=batch_disc_net).to(device)

    batch_disc_optim = torch.optim.Adam(batch_discriminator.parameters(), lr=1e-4)
    batch_adv_optim = torch.optim.Adam(bio_projector.parameters(), lr=1e-4)

    # Biological classifier

    bio_classifier_net = nn.Sequential(
        nn.Linear(projected_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, n_bios),
    )

    bio_classifier = BiologicalConservationClassifier(net=bio_classifier_net).to(device)

    bio_classifier_optim = torch.optim.Adam(bio_classifier.parameters(), lr=1e-4)

    # Decoders

    full_decoder_net = nn.Sequential(
        nn.Linear(2 * projected_dim, 258),
        nn.ReLU(),
        nn.Linear(258, 512),
        nn.ReLU(),
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 3 * input_size),
    )

    bio_decoder_net = nn.Sequential(
        nn.Linear(projected_dim, 258),
        nn.ReLU(),
        nn.Linear(258, 512),
        nn.ReLU(),
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 3 * input_size),
    )

    full_decoder = ZINBDecoder(full_decoder_net).to(device)

    bio_decoder = ZINBDecoder(bio_decoder_net).to(device)

    full_decoder_optim = torch.optim.Adam(full_decoder.parameters(), lr=1e-4)
    bio_decoder_optim = torch.optim.Adam(bio_decoder.parameters(), lr=1e-4)

    epochs_post = 1000

    abaco_drl_2(
        # training params
        epochs=epochs_post,
        data_loader=train_dataloader,
        data_batch_loader=train_batch_dataloader,
        device=device,
        # pre-trained VAEs
        frozen_bio_vae=bio_vae_model,
        frozen_batch_vae=batch_vae_model,
        # projectors
        bio_projector=bio_projector,
        batch_projector=batch_projector,
        projectors_optim=projectors_optim,
        # discriminator
        batch_discriminator=batch_discriminator,
        batch_disc_optim=batch_disc_optim,
        batch_adv_optim=batch_adv_optim,
        # bio classifier
        bio_classifier=bio_classifier,
        bio_classifier_optim=bio_classifier_optim,
        # decoders
        full_decoder=full_decoder,
        full_decoder_optim=full_decoder_optim,
        bio_decoder=bio_decoder,
        bio_decoder_optim=bio_decoder_optim,
        # default settings
        w_projector=1.0,
        w_disc_batch=1.0,
        w_adv_batch=1.0,
        w_bio_class=1.0,
        w_full_dec=1.0,
        w_bio_dec=1.0,
        disc_loss_type="CrossEntropy",
    )

    # Reconstructing data with trained model
    ohe_bio = one_hot_encoding(data["trt"])[0]
    ohe_batch = one_hot_encoding(data["batch"])[0]

    # Reconstructing data with trained model
    recon_data = []

    for x in train_dataloader:
        x = x[0].to(device)
        encoded = bio_vae_model.encoder(torch.cat([x, ohe_batch.to(device)], dim=1))
        z = encoded.rsample()
        s = bio_projector(z)
        decoded = bio_decoder(s)
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
    f"performance_metrics/AD_count_drl/{epochs_pre}_epochs_pre_{epochs_post}_epochs_post.csv",
    index=False,
)

# -------------------------------------------------------------------------------------------------------

# Several run framework - 2000 epochs to 2000 epochs

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

    # Define biological informed model
    # Model parameters

    K = 2
    n_batches = 5

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

    disc_latent_optim = torch.optim.Adam(discriminator_latent.parameters(), lr=5e-6)

    adv_latent_optim = torch.optim.Adam(bio_vae_model.encoder.parameters(), lr=5e-6)

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

    # Define batch informed model

    # Model parameters
    K = 5
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
    batch_vae_model = VampPriorMixtureConditionalVAE(
        encoder=encoder,
        decoder=decoder,
        input_dim=input_size,
        n_comps=K,
        batch_dim=n_batches,
        d_z=d_z,
        beta=20.0,
        data_loader=train_batch_dataloader,
    ).to(device)

    # Batch discriminator
    bio_latent_disc_net = nn.Sequential(
        nn.Linear(d_z + K, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, n_batches),
    )

    # Defining batch discriminator
    discriminator_latent = BatchDiscriminator(bio_latent_disc_net).to(device)

    # Optimizers

    vae_optim_pre = torch.optim.Adam(
        [
            {"params": batch_vae_model.encoder.parameters()},
            {"params": batch_vae_model.decoder.parameters()},
            {
                "params": [
                    batch_vae_model.u,
                    batch_vae_model.prior_pi,
                    batch_vae_model.prior_var,
                ]
            },
        ],
        lr=1e-3,
    )

    disc_latent_optim = torch.optim.Adam(discriminator_latent.parameters(), lr=5e-6)

    adv_latent_optim = torch.optim.Adam(bio_vae_model.encoder.parameters(), lr=5e-6)

    train_c_abaco_contra_modified(
        vae=batch_vae_model,
        vae_optim_pre=vae_optim_pre,
        discriminator=discriminator_latent,
        disc_optim=disc_latent_optim,
        adv_optim=adv_latent_optim,
        data_loader=train_batch_dataloader,
        epochs=epochs_pre,
        device=device,
        w_elbo=1.0,
        w_contra=100.0,
        temp=0.1,
        w_adv=1.0,
        w_disc=1.0,
        disc_loss_type="CrossEntropy",
    )

    # Disentanglement representation learning !

    # Latent space projectors

    projected_dim = 16

    bio_projector_net = nn.Sequential(
        nn.Linear(d_z, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, projected_dim),
    )

    batch_projector_net = nn.Sequential(
        nn.Linear(d_z, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, projected_dim),
    )

    bio_projector = LatentProjector(net=bio_projector_net).to(device)
    batch_projector = LatentProjector(net=batch_projector_net).to(device)

    projectors_optim = torch.optim.Adam(
        [
            {"params": bio_projector.parameters()},
            {"params": batch_projector.parameters()},
        ],
        lr=1e-4,
    )

    # Bio projector batch discriminator

    n_bios = 2
    n_batches = 5

    batch_disc_net = nn.Sequential(
        nn.Linear(projected_dim + n_bios, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, n_batches),
    )

    batch_discriminator = BatchDiscriminator(net=batch_disc_net).to(device)

    batch_disc_optim = torch.optim.Adam(batch_discriminator.parameters(), lr=1e-4)
    batch_adv_optim = torch.optim.Adam(bio_projector.parameters(), lr=1e-4)

    # Biological classifier

    bio_classifier_net = nn.Sequential(
        nn.Linear(projected_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, n_bios),
    )

    bio_classifier = BiologicalConservationClassifier(net=bio_classifier_net).to(device)

    bio_classifier_optim = torch.optim.Adam(bio_classifier.parameters(), lr=1e-4)

    # Decoders

    full_decoder_net = nn.Sequential(
        nn.Linear(2 * projected_dim, 258),
        nn.ReLU(),
        nn.Linear(258, 512),
        nn.ReLU(),
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 3 * input_size),
    )

    bio_decoder_net = nn.Sequential(
        nn.Linear(projected_dim, 258),
        nn.ReLU(),
        nn.Linear(258, 512),
        nn.ReLU(),
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 3 * input_size),
    )

    full_decoder = ZINBDecoder(full_decoder_net).to(device)

    bio_decoder = ZINBDecoder(bio_decoder_net).to(device)

    full_decoder_optim = torch.optim.Adam(full_decoder.parameters(), lr=1e-4)
    bio_decoder_optim = torch.optim.Adam(bio_decoder.parameters(), lr=1e-4)

    epochs_post = 2000

    abaco_drl_2(
        # training params
        epochs=epochs_post,
        data_loader=train_dataloader,
        data_batch_loader=train_batch_dataloader,
        device=device,
        # pre-trained VAEs
        frozen_bio_vae=bio_vae_model,
        frozen_batch_vae=batch_vae_model,
        # projectors
        bio_projector=bio_projector,
        batch_projector=batch_projector,
        projectors_optim=projectors_optim,
        # discriminator
        batch_discriminator=batch_discriminator,
        batch_disc_optim=batch_disc_optim,
        batch_adv_optim=batch_adv_optim,
        # bio classifier
        bio_classifier=bio_classifier,
        bio_classifier_optim=bio_classifier_optim,
        # decoders
        full_decoder=full_decoder,
        full_decoder_optim=full_decoder_optim,
        bio_decoder=bio_decoder,
        bio_decoder_optim=bio_decoder_optim,
        # default settings
        w_projector=1.0,
        w_disc_batch=1.0,
        w_adv_batch=1.0,
        w_bio_class=1.0,
        w_full_dec=1.0,
        w_bio_dec=1.0,
        disc_loss_type="CrossEntropy",
    )

    # Reconstructing data with trained model
    ohe_bio = one_hot_encoding(data["trt"])[0]
    ohe_batch = one_hot_encoding(data["batch"])[0]

    # Reconstructing data with trained model
    recon_data = []

    for x in train_dataloader:
        x = x[0].to(device)
        encoded = bio_vae_model.encoder(torch.cat([x, ohe_batch.to(device)], dim=1))
        z = encoded.rsample()
        s = bio_projector(z)
        decoded = bio_decoder(s)
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
    f"performance_metrics/AD_count_drl/{epochs_pre}_epochs_pre_{epochs_post}_epochs_post.csv",
    index=False,
)
