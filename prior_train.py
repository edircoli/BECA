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

# Several run framework - VampPrior Mixture Model

# iterations = 50
# performance = []

# # Set random seed
# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)

# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)

# # Start training loop
# for iter_index in tqdm(range(iterations), desc="ABaCo iteration"):
#     # Define model
#     # Model parameters
#     K = 2
#     n_batches = 5

#     # Encoder
#     mog_encoder_net = nn.Sequential(
#         nn.Linear(input_size, 1024),
#         nn.ReLU(),
#         nn.Linear(1024, 512),
#         nn.ReLU(),
#         nn.Linear(512, 258),
#         nn.ReLU(),
#         nn.Linear(258, K * (2 * d_z + 1)),
#     )

#     # Decoder
#     zinb_decoder_net = nn.Sequential(
#         nn.Linear(d_z, 258),
#         nn.ReLU(),
#         nn.Linear(258, 512),
#         nn.ReLU(),
#         nn.Linear(512, 1024),
#         nn.ReLU(),
#         nn.Linear(1024, 3 * input_size),
#     )

#     # Batch discriminator
#     batch_disc_net = nn.Sequential(
#         nn.Linear(d_z, 256),
#         nn.ReLU(),
#         nn.Linear(256, 128),
#         nn.ReLU(),
#         nn.Linear(128, 64),
#         nn.ReLU(),
#         nn.Linear(64, n_batches),
#     )

#     # Biological classifier
#     bio_class_net = nn.Sequential(
#         nn.Linear(d_z, 256),
#         nn.ReLU(),
#         nn.Linear(256, 128),
#         nn.ReLU(),
#         nn.Linear(128, 64),
#         nn.ReLU(),
#         nn.Linear(64, K),
#     )

#     # Defining VMM model
#     encoder = MoGEncoder(mog_encoder_net, n_comp=K)
#     decoder = ZINBDecoder(zinb_decoder_net)
#     vae_model = VampPriorMixtureVAE(
#         encoder=encoder,
#         decoder=decoder,
#         input_dim=input_size,
#         n_comps=K,
#         d_z=d_z,
#         beta=5.0,
#         data_loader=train_dataloader,
#     ).to(device)

#     # Defining batch discriminator
#     discriminator = BatchDiscriminator(batch_disc_net).to(device)

#     bio_classifier = BiologicalConservationClassifier(bio_class_net).to(device)

#     # Optimizers
#     adv_optim = torch.optim.Adam(
#         vae_model.encoder.parameters(), lr=1e-4, weight_decay=1e-5
#     )

#     vae_optim_pre = torch.optim.Adam(
#         [
#             {"params": vae_model.encoder.parameters()},
#             {"params": vae_model.decoder.parameters()},
#             {"params": [vae_model.u]},
#         ],
#         lr=1e-3,
#     )

#     vae_optim_post = torch.optim.Adam(
#         [
#             {"params": vae_model.encoder.parameters()},
#             {"params": vae_model.decoder.parameters()},
#         ],
#         lr=1e-4,
#     )

#     disc_optim = torch.optim.Adam(
#         discriminator.parameters(), lr=1e-4, weight_decay=1e-5
#     )
#     bio_optim = torch.optim.Adam(
#         [
#             {"params": bio_classifier.parameters()},
#             {"params": vae_model.encoder.parameters()},
#         ],
#         lr=1e-4,
#         weight_decay=1e-5,
#     )

#     # Train model
#     epochs = 1000

#     w_disc = 10.0
#     w_adv = 10.0
#     w_elbo = 1.0
#     w_bio = 10.0

#     train_abaco_two_step(
#         vae=vae_model,
#         vae_optim_pre=vae_optim_pre,
#         vae_optim_post=vae_optim_post,
#         discriminator=discriminator,
#         disc_optim=disc_optim,
#         adv_optim=adv_optim,
#         bio_classifier=bio_classifier,
#         bio_optim=bio_optim,
#         data_loader=train_dataloader,
#         epochs=epochs,
#         device=device,
#         w_disc=w_disc,
#         w_adv=w_adv,
#         w_elbo=w_elbo,
#         w_bio=w_bio,
#         disc_loss_type="CrossEntropy",
#     )

#     # Reconstructing data with trained model
#     recon_data = []

#     for x in train_dataloader:
#         x = x[0].to(device)
#         encoded = vae_model.encoder(x)
#         z = encoded.rsample()
#         decoded = vae_model.decoder(z)
#         recon = decoded.sample()
#         recon_data.append(recon)

#     np_recon_data = np.vstack([t.detach().cpu().numpy() for t in recon_data])

#     otu_corrected_pd = pd.concat(
#         [
#             pd.DataFrame(
#                 np_recon_data,
#                 index=data.index,
#                 columns=data.select_dtypes("number").columns,
#             ),
#             data[batch_label],
#             data[exp_label],
#             data[sample_label],
#         ],
#         axis=1,
#     )

#     # Normalize for comparison
#     norm_corrected_data_abaco = DataTransform(
#         otu_corrected_pd,
#         factors=["sample", "batch", "trt"],
#         transformation="CLR",
#         count=True,
#     )

#     # Compute metrics and save
#     kbet = kBET(norm_corrected_data_abaco, batch_label="batch")
#     ilisi = (iLISI(norm_corrected_data_abaco, batch_label="batch") - 1) / (
#         5 - 1
#     )  # iLISI goes from 1 to 5
#     ari = ARI(norm_corrected_data_abaco, bio_label="trt", n_clusters=2)

#     clisi = (5 - cLISI(norm_corrected_data_abaco, cell_label="trt")) / (5 - 1)
#     asw = 1 - ASW(norm_corrected_data_abaco, interest_label="trt")

#     performance.append(
#         {
#             "iter": iter_index,
#             "kBET": kbet,
#             "iLISI": ilisi,
#             "ARI": ari,
#             "cLISI": clisi,
#             "ASW": asw,
#         }
#     )

# performance_df = pd.DataFrame(performance)
# performance_df.to_csv(
#     f"performance_metrics/AD_count/VMM_{epochs}_epochs_{w_disc}_disc_{w_adv}_adv_{w_bio}_bio_{w_elbo}_elbo.csv",
#     index=False,
# )

# Several run framework - Mixture of Gaussians Model

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
    n_comp = 2
    K = 2
    n_batches = 5

    # Prior
    prior = MoGPrior(d_z, n_comp)

    # Encoder
    mog_encoder_net = nn.Sequential(
        nn.Linear(input_size, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 258),
        nn.ReLU(),
        nn.Linear(258, n_comp * (2 * d_z + 1)),
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

    # Batch discriminator
    batch_disc_net = nn.Sequential(
        nn.Linear(d_z, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, n_batches),
    )

    # Biological classifier
    bio_class_net = nn.Sequential(
        nn.Linear(d_z, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, K),
    )

    # Defining VAE model
    encoder = MoGEncoder(mog_encoder_net, n_comp)
    decoder = ZINBDecoder(zinb_decoder_net)
    vae_model = VAE(prior, encoder=encoder, decoder=decoder, beta=10.0).to(device)

    # Defining batch discriminator and biological classifier
    discriminator = BatchDiscriminator(batch_disc_net).to(device)
    bio_classifier = BiologicalConservationClassifier(bio_class_net).to(device)

    # Optimizers
    adv_optim = torch.optim.Adam(
        vae_model.encoder.parameters(), lr=1e-4, weight_decay=1e-5
    )
    vae_optim_pre = torch.optim.Adam(vae_model.parameters(), lr=1e-3, weight_decay=1e-5)

    vae_optim_post = torch.optim.Adam(
        [
            {"params": vae_model.encoder.parameters()},
            {"params": vae_model.decoder.parameters()},
        ],
        lr=1e-4,
    )

    disc_optim = torch.optim.Adam(
        discriminator.parameters(), lr=1e-4, weight_decay=1e-5
    )
    bio_optim = torch.optim.Adam(
        [
            {"params": bio_classifier.parameters()},
            {"params": vae_model.encoder.parameters()},
        ],
        lr=1e-4,
        weight_decay=1e-5,
    )

    # Train model
    epochs = 1000

    w_disc = 10.0
    w_adv = 10.0
    w_elbo = 1.0
    w_bio = 10.0

    train_abaco_two_step(
        vae=vae_model,
        vae_optim_pre=vae_optim_pre,
        vae_optim_post=vae_optim_post,
        discriminator=discriminator,
        disc_optim=disc_optim,
        adv_optim=adv_optim,
        bio_classifier=bio_classifier,
        bio_optim=bio_optim,
        data_loader=train_dataloader,
        epochs=epochs,
        device=device,
        w_disc=w_disc,
        w_adv=w_adv,
        w_elbo=w_elbo,
        w_bio=w_bio,
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
    f"performance_metrics/AD_count/MoG_{epochs}_epochs_{w_disc}_disc_{w_adv}_adv_{w_bio}_bio_{w_elbo}_elbo.csv",
    index=False,
)

# Several run framework - Standard Gaussian Model

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

    # Standard gaussian prior architecture

    # Model parameters
    n_comp = 2
    n_batches = 5

    # Prior
    prior = NormalPrior(d_z)

    # Encoder
    normal_encoder_net = nn.Sequential(
        nn.Linear(input_size, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 258),
        nn.ReLU(),
        nn.Linear(258, (2 * d_z)),
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

    # Batch discriminator
    batch_disc_net = nn.Sequential(
        nn.Linear(d_z, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, n_batches),
    )

    # Biological classifier
    bio_class_net = nn.Sequential(
        nn.Linear(d_z, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, K),
    )

    # Defining VAE model
    encoder = NormalEncoder(normal_encoder_net)
    decoder = ZINBDecoder(zinb_decoder_net)
    vae_model = VAE(prior, encoder=encoder, decoder=decoder, beta=1.0).to(device)

    # Defining batch discriminator and biological classifier
    discriminator = BatchDiscriminator(batch_disc_net).to(device)
    bio_classifier = BiologicalConservationClassifier(bio_class_net).to(device)

    # Optimizers
    adv_optim = torch.optim.Adam(
        vae_model.encoder.parameters(), lr=1e-4, weight_decay=1e-5
    )

    vae_optim_pre = torch.optim.Adam(vae_model.parameters(), lr=1e-3, weight_decay=1e-5)

    vae_optim_post = torch.optim.Adam(
        vae_model.parameters(), lr=1e-4, weight_decay=1e-5
    )

    disc_optim = torch.optim.Adam(
        discriminator.parameters(), lr=1e-4, weight_decay=1e-5
    )
    bio_optim = torch.optim.Adam(
        [
            {"params": bio_classifier.parameters()},
            {"params": vae_model.encoder.parameters()},
        ],
        lr=1e-4,
        weight_decay=1e-5,
    )

    # Train model
    epochs = 1000

    w_disc = 10.0
    w_adv = 10.0
    w_elbo = 1.0
    w_bio = 10.0

    train_abaco_two_step(
        vae=vae_model,
        vae_optim_pre=vae_optim_pre,
        vae_optim_post=vae_optim_post,
        discriminator=discriminator,
        disc_optim=disc_optim,
        adv_optim=adv_optim,
        bio_classifier=bio_classifier,
        bio_optim=bio_optim,
        data_loader=train_dataloader,
        epochs=epochs,
        device=device,
        w_disc=w_disc,
        w_adv=w_adv,
        w_elbo=w_elbo,
        w_bio=w_bio,
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
    f"performance_metrics/AD_count/Standard_{epochs}_epochs_{w_disc}_disc_{w_adv}_adv_{w_bio}_bio_{w_elbo}_elbo.csv",
    index=False,
)
