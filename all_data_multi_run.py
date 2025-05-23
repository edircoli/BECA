# Essentials
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random
import seaborn as sns

# User libraries
from BatchEffectDataLoader import DataPreprocess, DataTransform, one_hot_encoding
from BatchEffectCorrection import (
    correctCombat,
    correctLimma_rBE,
    correctBMC,
    correctPLSDAbatch_R,
    correctCombatSeq,
    correctConQuR,
)
from BatchEffectPlots import plotPCA, plotPCoA, plot_LISI_perplexity
from BatchEffectMetrics import all_metrics
from ABaCo import abaco_run, abaco_recon


# AD count data
path = "data/dataset_ad.csv"
ad_count_batch_label = "batch"
ad_count_sample_label = "sample"
ad_count_bio_label = "trt"

ad_count_data = DataPreprocess(
    path, factors=[ad_count_sample_label, ad_count_batch_label, ad_count_bio_label]
)

# train DataLoader: [samples, ohe_batch]
ad_count_input_size = 567
ad_count_train_dataloader = DataLoader(
    TensorDataset(
        torch.tensor(
            ad_count_data.select_dtypes(include="number").values, dtype=torch.float32
        ),  # samples
        one_hot_encoding(ad_count_data[ad_count_batch_label])[
            0
        ],  # one hot encoded batch information
        one_hot_encoding(ad_count_data[ad_count_bio_label])[
            0
        ],  # one hot encoded biological information
    ),
    batch_size=1000,
)
ad_count_batch_size = 5
ad_count_bio_size = 2

# IBD data
path = "data/MGnify/IBD/IBD_dataset.csv"
ibd_batch_label = "project ID"
ibd_sample_label = "run ID"
ibd_bio_label = "associated phenotype"

ibd_data = DataPreprocess(
    path, factors=[ibd_sample_label, ibd_batch_label, ibd_bio_label]
)

# train DataLoader: [samples, ohe_batch]
ibd_input_size = 435
ibd_train_dataloader = DataLoader(
    TensorDataset(
        torch.tensor(
            ibd_data.select_dtypes(include="number").values, dtype=torch.float32
        ),  # samples
        one_hot_encoding(ibd_data[ibd_batch_label])[
            0
        ],  # one hot encoded batch information
        one_hot_encoding(ibd_data[ibd_bio_label])[
            0
        ],  # one hot encoded biological information
    ),
    batch_size=1000,
)
ibd_batch_size = 2
ibd_bio_size = 3

# DTU-GE data
path = "data/MGnify/DTU-GE/count/DTU-GE_phylum_count_data.csv"
dtu_batch_label = "pipeline"
dtu_sample_label = "accession"
dtu_bio_label = "location"

dtu_data = DataPreprocess(
    path, factors=[dtu_sample_label, dtu_batch_label, dtu_bio_label]
)

# filter data that location appears less than 15 times
country_counts = dtu_data["location"].value_counts()
keep = country_counts[(country_counts >= 15) & (country_counts < 30)].index
dtu_data = dtu_data[dtu_data["location"].isin(keep)]

# train DataLoader: [samples, ohe_batch]
dtu_input_size = 189
dtu_train_dataloader = DataLoader(
    TensorDataset(
        torch.tensor(
            dtu_data.select_dtypes(include="number").values, dtype=torch.float32
        ),  # samples
        one_hot_encoding(dtu_data[dtu_batch_label])[
            0
        ],  # one hot encoded batch information
        one_hot_encoding(dtu_data[dtu_bio_label])[
            0
        ],  # one hot encoded biological information
    ),
    batch_size=1000,
)
dtu_batch_size = 2
dtu_bio_size = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Set for n iteration - KL cycle VMM models
n = 50

performances_batch_ad = []
performances_bio_ad = []

performances_batch_ibd = []
performances_bio_ibd = []

performances_batch_dtu = []
performances_bio_dtu = []

for iter in range(n):

    # VMM models
    ad_count_vmm_kl_cycle = abaco_run(
        dataloader=ad_count_train_dataloader,
        n_batches=ad_count_batch_size,
        n_bios=ad_count_bio_size,
        input_size=ad_count_input_size,
        device=device,
        w_contra=100.0,
        kl_cycle=True,
        seed=None,
    )

    ibd_vmm_kl_cycle = abaco_run(
        dataloader=ibd_train_dataloader,
        n_batches=ibd_batch_size,
        n_bios=ibd_bio_size,
        input_size=ibd_input_size,
        device=device,
        w_contra=10.0,
        kl_cycle=True,
        seed=None,
    )

    dtu_vmm_kl_cycle = abaco_run(
        dataloader=dtu_train_dataloader,
        n_batches=dtu_batch_size,
        n_bios=dtu_bio_size,
        input_size=dtu_input_size,
        device=device,
        w_contra=10.0,
        kl_cycle=True,
        seed=None,
    )

    # Reconstruct data
    ad_recon_data = abaco_recon(
        model=ad_count_vmm_kl_cycle,
        device=device,
        data=ad_count_data,
        dataloader=ad_count_train_dataloader,
        sample_label=ad_count_sample_label,
        batch_label=ad_count_batch_label,
        bio_label=ad_count_bio_label,
        seed=None,
    )

    ibd_recon_data = abaco_recon(
        model=ibd_vmm_kl_cycle,
        device=device,
        data=ibd_data,
        dataloader=ibd_train_dataloader,
        sample_label=ibd_sample_label,
        batch_label=ibd_batch_label,
        bio_label=ibd_bio_label,
        seed=None,
    )

    dtu_recon_data = abaco_recon(
        model=dtu_vmm_kl_cycle,
        device=device,
        data=dtu_data,
        dataloader=dtu_train_dataloader,
        sample_label=dtu_sample_label,
        batch_label=dtu_batch_label,
        bio_label=dtu_bio_label,
        seed=None,
    )

    # Performance metrics
    norm_ad_recon_data = DataTransform(
        data=ad_recon_data,
        factors=[ad_count_sample_label, ad_count_batch_label, ad_count_bio_label],
        count=True,
    )

    performance_batch, performance_bio = all_metrics(
        data=norm_ad_recon_data,
        bio_label=ad_count_bio_label,
        batch_label=ad_count_batch_label,
    )
    performances_batch_ad.append(performance_batch)
    performances_bio_ad.append(performance_bio)

    norm_ibd_recon_data = DataTransform(
        data=ibd_recon_data,
        factors=[ibd_sample_label, ibd_batch_label, ibd_bio_label],
        count=True,
    )

    performance_batch, performance_bio = all_metrics(
        data=norm_ibd_recon_data, bio_label=ibd_bio_label, batch_label=ibd_batch_label
    )
    performances_batch_ibd.append(performance_batch)
    performances_bio_ibd.append(performance_bio)

    norm_dtu_recon_data = DataTransform(
        data=dtu_recon_data,
        factors=[dtu_sample_label, dtu_batch_label, dtu_bio_label],
        count=True,
    )

    performance_batch, performance_bio = all_metrics(
        data=norm_dtu_recon_data, bio_label=dtu_bio_label, batch_label=dtu_batch_label
    )
    performances_batch_dtu.append(performance_batch)
    performances_bio_dtu.append(performance_bio)

# Save performance to file
pd.DataFrame(performances_batch_ad).to_csv(
    "performance_metrics/AD_count/VMM_KL_cycle_batch.csv", index=False
)
pd.DataFrame(performances_bio_ad).to_csv(
    "performance_metrics/AD_count/VMM_KL_cycle_bio.csv", index=False
)

pd.DataFrame(performances_batch_ibd).to_csv(
    "performance_metrics/IBD/VMM_KL_cycle_batch.csv", index=False
)
pd.DataFrame(performances_bio_ibd).to_csv(
    "performance_metrics/IBD/VMM_KL_cycle_bio.csv", index=False
)

pd.DataFrame(performances_batch_dtu).to_csv(
    "performance_metrics/DTU-GE/VMM_KL_cycle_batch.csv", index=False
)
pd.DataFrame(performances_bio_dtu).to_csv(
    "performance_metrics/DTU-GE/VMM_KL_cycle_bio.csv", index=False
)

# Set seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Set for n iteration - KL cycle MoG models
n = 50

performances_batch_ad = []
performances_bio_ad = []

performances_batch_ibd = []
performances_bio_ibd = []

performances_batch_dtu = []
performances_bio_dtu = []

for iter in range(n):

    # VMM models
    ad_count_mog_kl_cycle = abaco_run(
        dataloader=ad_count_train_dataloader,
        n_batches=ad_count_batch_size,
        n_bios=ad_count_bio_size,
        input_size=ad_count_input_size,
        device=device,
        prior="MoG",
        w_contra=100.0,
        kl_cycle=True,
        seed=None,
    )

    ibd_mog_kl_cycle = abaco_run(
        dataloader=ibd_train_dataloader,
        n_batches=ibd_batch_size,
        n_bios=ibd_bio_size,
        input_size=ibd_input_size,
        device=device,
        prior="MoG",
        w_contra=10.0,
        kl_cycle=True,
        seed=None,
    )

    dtu_mog_kl_cycle = abaco_run(
        dataloader=dtu_train_dataloader,
        n_batches=dtu_batch_size,
        n_bios=dtu_bio_size,
        input_size=dtu_input_size,
        device=device,
        prior="MoG",
        w_contra=10.0,
        kl_cycle=True,
        seed=None,
    )

    # Reconstruct data
    ad_recon_data = abaco_recon(
        model=ad_count_mog_kl_cycle,
        device=device,
        data=ad_count_data,
        dataloader=ad_count_train_dataloader,
        sample_label=ad_count_sample_label,
        batch_label=ad_count_batch_label,
        bio_label=ad_count_bio_label,
        seed=None,
    )

    ibd_recon_data = abaco_recon(
        model=ibd_mog_kl_cycle,
        device=device,
        data=ibd_data,
        dataloader=ibd_train_dataloader,
        sample_label=ibd_sample_label,
        batch_label=ibd_batch_label,
        bio_label=ibd_bio_label,
        seed=None,
    )

    dtu_recon_data = abaco_recon(
        model=dtu_mog_kl_cycle,
        device=device,
        data=dtu_data,
        dataloader=dtu_train_dataloader,
        sample_label=dtu_sample_label,
        batch_label=dtu_batch_label,
        bio_label=dtu_bio_label,
        seed=None,
    )

    # Performance metrics
    norm_ad_recon_data = DataTransform(
        data=ad_recon_data,
        factors=[ad_count_sample_label, ad_count_batch_label, ad_count_bio_label],
        count=True,
    )

    performance_batch, performance_bio = all_metrics(
        data=norm_ad_recon_data,
        bio_label=ad_count_bio_label,
        batch_label=ad_count_batch_label,
    )
    performances_batch_ad.append(performance_batch)
    performances_bio_ad.append(performance_bio)

    norm_ibd_recon_data = DataTransform(
        data=ibd_recon_data,
        factors=[ibd_sample_label, ibd_batch_label, ibd_bio_label],
        count=True,
    )

    performance_batch, performance_bio = all_metrics(
        data=norm_ibd_recon_data, bio_label=ibd_bio_label, batch_label=ibd_batch_label
    )
    performances_batch_ibd.append(performance_batch)
    performances_bio_ibd.append(performance_bio)

    norm_dtu_recon_data = DataTransform(
        data=dtu_recon_data,
        factors=[dtu_sample_label, dtu_batch_label, dtu_bio_label],
        count=True,
    )

    performance_batch, performance_bio = all_metrics(
        data=norm_dtu_recon_data, bio_label=dtu_bio_label, batch_label=dtu_batch_label
    )
    performances_batch_dtu.append(performance_batch)
    performances_bio_dtu.append(performance_bio)

# Save performance to file
pd.DataFrame(performances_batch_ad).to_csv(
    "performance_metrics/AD_count/MoG_KL_cycle_batch.csv", index=False
)
pd.DataFrame(performances_bio_ad).to_csv(
    "performance_metrics/AD_count/MoG_KL_cycle_bio.csv", index=False
)

pd.DataFrame(performances_batch_ibd).to_csv(
    "performance_metrics/IBD/MoG_KL_cycle_batch.csv", index=False
)
pd.DataFrame(performances_bio_ibd).to_csv(
    "performance_metrics/IBD/MoG_KL_cycle_bio.csv", index=False
)

pd.DataFrame(performances_batch_dtu).to_csv(
    "performance_metrics/DTU-GE/MoG_KL_cycle_batch.csv", index=False
)
pd.DataFrame(performances_bio_dtu).to_csv(
    "performance_metrics/DTU-GE/MoG_KL_cycle_bio.csv", index=False
)

# Set seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Set for n iteration - KL cycle Normal models
n = 50

performances_batch_ad = []
performances_bio_ad = []

performances_batch_ibd = []
performances_bio_ibd = []

performances_batch_dtu = []
performances_bio_dtu = []

for iter in range(n):

    # VMM models
    ad_count_std_kl_cycle = abaco_run(
        dataloader=ad_count_train_dataloader,
        n_batches=ad_count_batch_size,
        n_bios=ad_count_bio_size,
        input_size=ad_count_input_size,
        device=device,
        prior="Normal",
        w_contra=100.0,
        kl_cycle=True,
        seed=None,
    )

    ibd_std_kl_cycle = abaco_run(
        dataloader=ibd_train_dataloader,
        n_batches=ibd_batch_size,
        n_bios=ibd_bio_size,
        input_size=ibd_input_size,
        device=device,
        prior="Normal",
        w_contra=10.0,
        kl_cycle=True,
        seed=None,
    )

    dtu_std_kl_cycle = abaco_run(
        dataloader=dtu_train_dataloader,
        n_batches=dtu_batch_size,
        n_bios=dtu_bio_size,
        input_size=dtu_input_size,
        device=device,
        prior="Normal",
        w_contra=10.0,
        kl_cycle=True,
        seed=None,
    )

    # Reconstruct data
    ad_recon_data = abaco_recon(
        model=ad_count_std_kl_cycle,
        device=device,
        data=ad_count_data,
        dataloader=ad_count_train_dataloader,
        sample_label=ad_count_sample_label,
        batch_label=ad_count_batch_label,
        bio_label=ad_count_bio_label,
        seed=None,
    )

    ibd_recon_data = abaco_recon(
        model=ibd_std_kl_cycle,
        device=device,
        data=ibd_data,
        dataloader=ibd_train_dataloader,
        sample_label=ibd_sample_label,
        batch_label=ibd_batch_label,
        bio_label=ibd_bio_label,
        seed=None,
    )

    dtu_recon_data = abaco_recon(
        model=dtu_std_kl_cycle,
        device=device,
        data=dtu_data,
        dataloader=dtu_train_dataloader,
        sample_label=dtu_sample_label,
        batch_label=dtu_batch_label,
        bio_label=dtu_bio_label,
        seed=None,
    )

    # Performance metrics
    norm_ad_recon_data = DataTransform(
        data=ad_recon_data,
        factors=[ad_count_sample_label, ad_count_batch_label, ad_count_bio_label],
        count=True,
    )

    performance_batch, performance_bio = all_metrics(
        data=norm_ad_recon_data,
        bio_label=ad_count_bio_label,
        batch_label=ad_count_batch_label,
    )
    performances_batch_ad.append(performance_batch)
    performances_bio_ad.append(performance_bio)

    norm_ibd_recon_data = DataTransform(
        data=ibd_recon_data,
        factors=[ibd_sample_label, ibd_batch_label, ibd_bio_label],
        count=True,
    )

    performance_batch, performance_bio = all_metrics(
        data=norm_ibd_recon_data, bio_label=ibd_bio_label, batch_label=ibd_batch_label
    )
    performances_batch_ibd.append(performance_batch)
    performances_bio_ibd.append(performance_bio)

    norm_dtu_recon_data = DataTransform(
        data=dtu_recon_data,
        factors=[dtu_sample_label, dtu_batch_label, dtu_bio_label],
        count=True,
    )

    performance_batch, performance_bio = all_metrics(
        data=norm_dtu_recon_data, bio_label=dtu_bio_label, batch_label=dtu_batch_label
    )
    performances_batch_dtu.append(performance_batch)
    performances_bio_dtu.append(performance_bio)

# Save performance to file
pd.DataFrame(performances_batch_ad).to_csv(
    "performance_metrics/AD_count/Std_KL_cycle_batch.csv", index=False
)
pd.DataFrame(performances_bio_ad).to_csv(
    "performance_metrics/AD_count/Std_KL_cycle_bio.csv", index=False
)

pd.DataFrame(performances_batch_ibd).to_csv(
    "performance_metrics/IBD/Std_KL_cycle_batch.csv", index=False
)
pd.DataFrame(performances_bio_ibd).to_csv(
    "performance_metrics/IBD/Std_KL_cycle_bio.csv", index=False
)

pd.DataFrame(performances_batch_dtu).to_csv(
    "performance_metrics/DTU-GE/Std_KL_cycle_batch.csv", index=False
)
pd.DataFrame(performances_bio_dtu).to_csv(
    "performance_metrics/DTU-GE/Std_KL_cycle_bio.csv", index=False
)


# Set seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Set for n iteration - No cycle VMM models
n = 50

performances_batch_ad = []
performances_bio_ad = []

performances_batch_ibd = []
performances_bio_ibd = []

performances_batch_dtu = []
performances_bio_dtu = []

for iter in range(n):

    # VMM models
    ad_count_vmm_no_cycle = abaco_run(
        dataloader=ad_count_train_dataloader,
        n_batches=ad_count_batch_size,
        n_bios=ad_count_bio_size,
        input_size=ad_count_input_size,
        device=device,
        w_contra=100.0,
        kl_cycle=False,
        seed=None,
    )

    ibd_vmm_no_cycle = abaco_run(
        dataloader=ibd_train_dataloader,
        n_batches=ibd_batch_size,
        n_bios=ibd_bio_size,
        input_size=ibd_input_size,
        device=device,
        w_contra=10.0,
        kl_cycle=False,
        seed=None,
    )

    dtu_vmm_no_cycle = abaco_run(
        dataloader=dtu_train_dataloader,
        n_batches=dtu_batch_size,
        n_bios=dtu_bio_size,
        input_size=dtu_input_size,
        device=device,
        w_contra=10.0,
        kl_cycle=False,
        seed=None,
    )

    # Reconstruct data
    ad_recon_data = abaco_recon(
        model=ad_count_vmm_no_cycle,
        device=device,
        data=ad_count_data,
        dataloader=ad_count_train_dataloader,
        sample_label=ad_count_sample_label,
        batch_label=ad_count_batch_label,
        bio_label=ad_count_bio_label,
        seed=None,
    )

    ibd_recon_data = abaco_recon(
        model=ibd_vmm_no_cycle,
        device=device,
        data=ibd_data,
        dataloader=ibd_train_dataloader,
        sample_label=ibd_sample_label,
        batch_label=ibd_batch_label,
        bio_label=ibd_bio_label,
        seed=None,
    )

    dtu_recon_data = abaco_recon(
        model=dtu_vmm_no_cycle,
        device=device,
        data=dtu_data,
        dataloader=dtu_train_dataloader,
        sample_label=dtu_sample_label,
        batch_label=dtu_batch_label,
        bio_label=dtu_bio_label,
        seed=None,
    )

    # Performance metrics
    norm_ad_recon_data = DataTransform(
        data=ad_recon_data,
        factors=[ad_count_sample_label, ad_count_batch_label, ad_count_bio_label],
        count=True,
    )

    performance_batch, performance_bio = all_metrics(
        data=norm_ad_recon_data,
        bio_label=ad_count_bio_label,
        batch_label=ad_count_batch_label,
    )
    performances_batch_ad.append(performance_batch)
    performances_bio_ad.append(performance_bio)

    norm_ibd_recon_data = DataTransform(
        data=ibd_recon_data,
        factors=[ibd_sample_label, ibd_batch_label, ibd_bio_label],
        count=True,
    )

    performance_batch, performance_bio = all_metrics(
        data=norm_ibd_recon_data, bio_label=ibd_bio_label, batch_label=ibd_batch_label
    )
    performances_batch_ibd.append(performance_batch)
    performances_bio_ibd.append(performance_bio)

    norm_dtu_recon_data = DataTransform(
        data=dtu_recon_data,
        factors=[dtu_sample_label, dtu_batch_label, dtu_bio_label],
        count=True,
    )

    performance_batch, performance_bio = all_metrics(
        data=norm_dtu_recon_data, bio_label=dtu_bio_label, batch_label=dtu_batch_label
    )
    performances_batch_dtu.append(performance_batch)
    performances_bio_dtu.append(performance_bio)

# Save performance to file
pd.DataFrame(performances_batch_ad).to_csv(
    "performance_metrics/AD_count/VMM_no_cycle_batch.csv", index=False
)
pd.DataFrame(performances_bio_ad).to_csv(
    "performance_metrics/AD_count/VMM_no_cycle_bio.csv", index=False
)

pd.DataFrame(performances_batch_ibd).to_csv(
    "performance_metrics/IBD/VMM_no_cycle_batch.csv", index=False
)
pd.DataFrame(performances_bio_ibd).to_csv(
    "performance_metrics/IBD/VMM_no_cycle_bio.csv", index=False
)

pd.DataFrame(performances_batch_dtu).to_csv(
    "performance_metrics/DTU-GE/VMM_no_cycle_batch.csv", index=False
)
pd.DataFrame(performances_bio_dtu).to_csv(
    "performance_metrics/DTU-GE/VMM_no_cycle_bio.csv", index=False
)

# Set seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Set for n iteration - No cycle MoG models
n = 50

performances_batch_ad = []
performances_bio_ad = []

performances_batch_ibd = []
performances_bio_ibd = []

performances_batch_dtu = []
performances_bio_dtu = []

for iter in range(n):

    # MoG models
    ad_count_mog_no_cycle = abaco_run(
        dataloader=ad_count_train_dataloader,
        n_batches=ad_count_batch_size,
        n_bios=ad_count_bio_size,
        input_size=ad_count_input_size,
        device=device,
        prior="MoG",
        w_contra=100.0,
        kl_cycle=False,
        seed=None,
    )

    ibd_mog_no_cycle = abaco_run(
        dataloader=ibd_train_dataloader,
        n_batches=ibd_batch_size,
        n_bios=ibd_bio_size,
        input_size=ibd_input_size,
        device=device,
        prior="MoG",
        w_contra=10.0,
        kl_cycle=False,
        seed=None,
    )

    dtu_mog_no_cycle = abaco_run(
        dataloader=dtu_train_dataloader,
        n_batches=dtu_batch_size,
        n_bios=dtu_bio_size,
        input_size=dtu_input_size,
        device=device,
        prior="MoG",
        w_contra=10.0,
        kl_cycle=False,
        seed=None,
    )

    # Reconstruct data
    ad_recon_data = abaco_recon(
        model=ad_count_mog_no_cycle,
        device=device,
        data=ad_count_data,
        dataloader=ad_count_train_dataloader,
        sample_label=ad_count_sample_label,
        batch_label=ad_count_batch_label,
        bio_label=ad_count_bio_label,
        seed=None,
    )

    ibd_recon_data = abaco_recon(
        model=ibd_mog_no_cycle,
        device=device,
        data=ibd_data,
        dataloader=ibd_train_dataloader,
        sample_label=ibd_sample_label,
        batch_label=ibd_batch_label,
        bio_label=ibd_bio_label,
        seed=None,
    )

    dtu_recon_data = abaco_recon(
        model=dtu_mog_no_cycle,
        device=device,
        data=dtu_data,
        dataloader=dtu_train_dataloader,
        sample_label=dtu_sample_label,
        batch_label=dtu_batch_label,
        bio_label=dtu_bio_label,
        seed=None,
    )

    # Performance metrics
    norm_ad_recon_data = DataTransform(
        data=ad_recon_data,
        factors=[ad_count_sample_label, ad_count_batch_label, ad_count_bio_label],
        count=True,
    )

    performance_batch, performance_bio = all_metrics(
        data=norm_ad_recon_data,
        bio_label=ad_count_bio_label,
        batch_label=ad_count_batch_label,
    )
    performances_batch_ad.append(performance_batch)
    performances_bio_ad.append(performance_bio)

    norm_ibd_recon_data = DataTransform(
        data=ibd_recon_data,
        factors=[ibd_sample_label, ibd_batch_label, ibd_bio_label],
        count=True,
    )

    performance_batch, performance_bio = all_metrics(
        data=norm_ibd_recon_data, bio_label=ibd_bio_label, batch_label=ibd_batch_label
    )
    performances_batch_ibd.append(performance_batch)
    performances_bio_ibd.append(performance_bio)

    norm_dtu_recon_data = DataTransform(
        data=dtu_recon_data,
        factors=[dtu_sample_label, dtu_batch_label, dtu_bio_label],
        count=True,
    )

    performance_batch, performance_bio = all_metrics(
        data=norm_dtu_recon_data, bio_label=dtu_bio_label, batch_label=dtu_batch_label
    )
    performances_batch_dtu.append(performance_batch)
    performances_bio_dtu.append(performance_bio)

# Save performance to file
pd.DataFrame(performances_batch_ad).to_csv(
    "performance_metrics/AD_count/MoG_no_cycle_batch.csv", index=False
)
pd.DataFrame(performances_bio_ad).to_csv(
    "performance_metrics/AD_count/MoG_no_cycle_bio.csv", index=False
)

pd.DataFrame(performances_batch_ibd).to_csv(
    "performance_metrics/IBD/MoG_no_cycle_batch.csv", index=False
)
pd.DataFrame(performances_bio_ibd).to_csv(
    "performance_metrics/IBD/MoG_no_cycle_bio.csv", index=False
)

pd.DataFrame(performances_batch_dtu).to_csv(
    "performance_metrics/DTU-GE/MoG_no_cycle_batch.csv", index=False
)
pd.DataFrame(performances_bio_dtu).to_csv(
    "performance_metrics/DTU-GE/MoG_no_cycle_bio.csv", index=False
)

# Set seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Set for n iteration - No cycle Normal models
n = 50

performances_batch_ad = []
performances_bio_ad = []

performances_batch_ibd = []
performances_bio_ibd = []

performances_batch_dtu = []
performances_bio_dtu = []

for iter in range(n):

    # VMM models
    ad_count_std_no_cycle = abaco_run(
        dataloader=ad_count_train_dataloader,
        n_batches=ad_count_batch_size,
        n_bios=ad_count_bio_size,
        input_size=ad_count_input_size,
        device=device,
        prior="Normal",
        w_contra=100.0,
        kl_cycle=False,
        seed=None,
    )

    ibd_std_no_cycle = abaco_run(
        dataloader=ibd_train_dataloader,
        n_batches=ibd_batch_size,
        n_bios=ibd_bio_size,
        input_size=ibd_input_size,
        device=device,
        prior="Normal",
        w_contra=10.0,
        kl_cycle=False,
        seed=None,
    )

    dtu_std_no_cycle = abaco_run(
        dataloader=dtu_train_dataloader,
        n_batches=dtu_batch_size,
        n_bios=dtu_bio_size,
        input_size=dtu_input_size,
        device=device,
        prior="Normal",
        w_contra=10.0,
        kl_cycle=False,
        seed=None,
    )

    # Reconstruct data
    ad_recon_data = abaco_recon(
        model=ad_count_std_no_cycle,
        device=device,
        data=ad_count_data,
        dataloader=ad_count_train_dataloader,
        sample_label=ad_count_sample_label,
        batch_label=ad_count_batch_label,
        bio_label=ad_count_bio_label,
        seed=None,
    )

    ibd_recon_data = abaco_recon(
        model=ibd_std_no_cycle,
        device=device,
        data=ibd_data,
        dataloader=ibd_train_dataloader,
        sample_label=ibd_sample_label,
        batch_label=ibd_batch_label,
        bio_label=ibd_bio_label,
        seed=None,
    )

    dtu_recon_data = abaco_recon(
        model=dtu_std_no_cycle,
        device=device,
        data=dtu_data,
        dataloader=dtu_train_dataloader,
        sample_label=dtu_sample_label,
        batch_label=dtu_batch_label,
        bio_label=dtu_bio_label,
        seed=None,
    )

    # Performance metrics
    norm_ad_recon_data = DataTransform(
        data=ad_recon_data,
        factors=[ad_count_sample_label, ad_count_batch_label, ad_count_bio_label],
        count=True,
    )

    performance_batch, performance_bio = all_metrics(
        data=norm_ad_recon_data,
        bio_label=ad_count_bio_label,
        batch_label=ad_count_batch_label,
    )
    performances_batch_ad.append(performance_batch)
    performances_bio_ad.append(performance_bio)

    norm_ibd_recon_data = DataTransform(
        data=ibd_recon_data,
        factors=[ibd_sample_label, ibd_batch_label, ibd_bio_label],
        count=True,
    )

    performance_batch, performance_bio = all_metrics(
        data=norm_ibd_recon_data, bio_label=ibd_bio_label, batch_label=ibd_batch_label
    )
    performances_batch_ibd.append(performance_batch)
    performances_bio_ibd.append(performance_bio)

    norm_dtu_recon_data = DataTransform(
        data=dtu_recon_data,
        factors=[dtu_sample_label, dtu_batch_label, dtu_bio_label],
        count=True,
    )

    performance_batch, performance_bio = all_metrics(
        data=norm_dtu_recon_data, bio_label=dtu_bio_label, batch_label=dtu_batch_label
    )
    performances_batch_dtu.append(performance_batch)
    performances_bio_dtu.append(performance_bio)

# Save performance to file
pd.DataFrame(performances_batch_ad).to_csv(
    "performance_metrics/AD_count/Std_no_cycle_batch.csv", index=False
)
pd.DataFrame(performances_bio_ad).to_csv(
    "performance_metrics/AD_count/Std_no_cycle_bio.csv", index=False
)

pd.DataFrame(performances_batch_ibd).to_csv(
    "performance_metrics/IBD/Std_no_cycle_batch.csv", index=False
)
pd.DataFrame(performances_bio_ibd).to_csv(
    "performance_metrics/IBD/Std_no_cycle_bio.csv", index=False
)

pd.DataFrame(performances_batch_dtu).to_csv(
    "performance_metrics/DTU-GE/Std_no_cycle_batch.csv", index=False
)
pd.DataFrame(performances_bio_dtu).to_csv(
    "performance_metrics/DTU-GE/Std_no_cycle_bio.csv", index=False
)
