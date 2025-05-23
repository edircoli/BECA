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
from BatchEffectMetrics import (
    all_metrics,
    pairwise_distance,
    PERMANOVA,
    pairwise_distance_std,
)
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------------------------

# Set seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Set for n iteration - KL cycle VMM models
n = 50

pairwise_distances = []
permanova_res_bc = []
permanova_res_ait = []

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

    # Normalized reconstructed data
    norm_ad_recon_data = DataTransform(
        data=ad_recon_data,
        factors=[ad_count_sample_label, ad_count_batch_label, ad_count_bio_label],
        count=True,
    )

    # Pairwise distance
    mean_all, mean_intra, mean_inter = pairwise_distance_std(
        data=ad_recon_data,
        sample_label=ad_count_sample_label,
        batch_label=ad_count_batch_label,
        bio_label=ad_count_bio_label,
    )
    pairwise_res = {
        "mean_all": mean_all,
        "mean_intra": mean_intra,
        "mean_inter": mean_inter,
    }
    # Add to list
    pairwise_distances.append(pairwise_res)

    # PERMANOVA
    bc, ait = PERMANOVA(
        ad_recon_data, ad_count_sample_label, ad_count_batch_label, ad_count_bio_label
    )
    permanova_bc = {
        "F-statistic": bc["test statistic"],
        "p-value": bc["p-value"],
        "R2": bc["R2"],
    }
    permanova_ait = {
        "F-statistic": ait["test statistic"],
        "p-value": ait["p-value"],
        "R2": ait["R2"],
    }
    # Add to list
    permanova_res_bc.append(permanova_bc)
    permanova_res_ait.append(permanova_ait)

# Save performance to file
pd.DataFrame(pairwise_distances).to_csv(
    "performance_metrics/AD_count/VMM_KL_cycle_pairwise_distances_std.csv", index=False
)
pd.DataFrame(permanova_res_bc).to_csv(
    "performance_metrics/AD_count/VMM_KL_cycle_permanova_braycurtis_2.csv", index=False
)
pd.DataFrame(permanova_res_ait).to_csv(
    "performance_metrics/AD_count/VMM_KL_cycle_permanova_aitchison_2.csv", index=False
)

# ------------------------------------------------------------------------------------

# Set seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Set for n iteration - KL cycle MoG models
n = 50

pairwise_distances = []
permanova_res_bc = []
permanova_res_ait = []

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

    # Normalized reconstructed data
    norm_ad_recon_data = DataTransform(
        data=ad_recon_data,
        factors=[ad_count_sample_label, ad_count_batch_label, ad_count_bio_label],
        count=True,
    )

    # Pairwise distance
    mean_all, mean_intra, mean_inter = pairwise_distance_std(
        data=ad_recon_data,
        sample_label=ad_count_sample_label,
        batch_label=ad_count_batch_label,
        bio_label=ad_count_bio_label,
    )
    pairwise_res = {
        "mean_all": mean_all,
        "mean_intra": mean_intra,
        "mean_inter": mean_inter,
    }
    # Add to list
    pairwise_distances.append(pairwise_res)

    # PERMANOVA
    bc, ait = PERMANOVA(
        ad_recon_data, ad_count_sample_label, ad_count_batch_label, ad_count_bio_label
    )
    permanova_bc = {
        "F-statistic": bc["test statistic"],
        "p-value": bc["p-value"],
        "R2": bc["R2"],
    }
    permanova_ait = {
        "F-statistic": ait["test statistic"],
        "p-value": ait["p-value"],
        "R2": ait["R2"],
    }
    # Add to list
    permanova_res_bc.append(permanova_bc)
    permanova_res_ait.append(permanova_ait)

# Save performance to file
pd.DataFrame(pairwise_distances).to_csv(
    "performance_metrics/AD_count/MoG_KL_cycle_pairwise_distances_std.csv", index=False
)
pd.DataFrame(permanova_res_bc).to_csv(
    "performance_metrics/AD_count/MoG_KL_cycle_permanova_braycurtis_2.csv", index=False
)
pd.DataFrame(permanova_res_ait).to_csv(
    "performance_metrics/AD_count/MoG_KL_cycle_permanova_aitchison_2.csv", index=False
)

# ------------------------------------------------------------------------------------

# Set seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Set for n iteration - KL cycle Normal models
n = 50

pairwise_distances = []
permanova_res_bc = []
permanova_res_ait = []

for iter in range(n):

    # VMM models
    ad_count_normal_kl_cycle = abaco_run(
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

    # Reconstruct data
    ad_recon_data = abaco_recon(
        model=ad_count_normal_kl_cycle,
        device=device,
        data=ad_count_data,
        dataloader=ad_count_train_dataloader,
        sample_label=ad_count_sample_label,
        batch_label=ad_count_batch_label,
        bio_label=ad_count_bio_label,
        seed=None,
    )

    # Normalized reconstructed data
    norm_ad_recon_data = DataTransform(
        data=ad_recon_data,
        factors=[ad_count_sample_label, ad_count_batch_label, ad_count_bio_label],
        count=True,
    )

    # Pairwise distance
    mean_all, mean_intra, mean_inter = pairwise_distance_std(
        data=ad_recon_data,
        sample_label=ad_count_sample_label,
        batch_label=ad_count_batch_label,
        bio_label=ad_count_bio_label,
    )
    pairwise_res = {
        "mean_all": mean_all,
        "mean_intra": mean_intra,
        "mean_inter": mean_inter,
    }
    # Add to list
    pairwise_distances.append(pairwise_res)

    # PERMANOVA
    bc, ait = PERMANOVA(
        ad_recon_data, ad_count_sample_label, ad_count_batch_label, ad_count_bio_label
    )
    permanova_bc = {
        "F-statistic": bc["test statistic"],
        "p-value": bc["p-value"],
        "R2": bc["R2"],
    }
    permanova_ait = {
        "F-statistic": ait["test statistic"],
        "p-value": ait["p-value"],
        "R2": ait["R2"],
    }
    # Add to list
    permanova_res_bc.append(permanova_bc)
    permanova_res_ait.append(permanova_ait)

# Save performance to file
pd.DataFrame(pairwise_distances).to_csv(
    "performance_metrics/AD_count/Std_KL_cycle_pairwise_distances_std.csv", index=False
)
pd.DataFrame(permanova_res_bc).to_csv(
    "performance_metrics/AD_count/Std_KL_cycle_permanova_braycurtis_2.csv", index=False
)
pd.DataFrame(permanova_res_ait).to_csv(
    "performance_metrics/AD_count/Std_KL_cycle_permanova_aitchison_2.csv", index=False
)

# ------------------------------------------------------------------------------------

# Set seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Set for n iteration - No cycle VMM models
n = 50

pairwise_distances = []
permanova_res_bc = []
permanova_res_ait = []

for iter in range(n):

    # VMM models
    ad_count_vmm_kl_cycle = abaco_run(
        dataloader=ad_count_train_dataloader,
        n_batches=ad_count_batch_size,
        n_bios=ad_count_bio_size,
        input_size=ad_count_input_size,
        device=device,
        w_contra=100.0,
        kl_cycle=False,
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

    # Normalized reconstructed data
    norm_ad_recon_data = DataTransform(
        data=ad_recon_data,
        factors=[ad_count_sample_label, ad_count_batch_label, ad_count_bio_label],
        count=True,
    )

    # Pairwise distance
    mean_all, mean_intra, mean_inter = pairwise_distance_std(
        data=ad_recon_data,
        sample_label=ad_count_sample_label,
        batch_label=ad_count_batch_label,
        bio_label=ad_count_bio_label,
    )
    pairwise_res = {
        "mean_all": mean_all,
        "mean_intra": mean_intra,
        "mean_inter": mean_inter,
    }
    # Add to list
    pairwise_distances.append(pairwise_res)

    # PERMANOVA
    bc, ait = PERMANOVA(
        ad_recon_data, ad_count_sample_label, ad_count_batch_label, ad_count_bio_label
    )
    permanova_bc = {
        "F-statistic": bc["test statistic"],
        "p-value": bc["p-value"],
        "R2": bc["R2"],
    }
    permanova_ait = {
        "F-statistic": ait["test statistic"],
        "p-value": ait["p-value"],
        "R2": ait["R2"],
    }
    # Add to list
    permanova_res_bc.append(permanova_bc)
    permanova_res_ait.append(permanova_ait)

# Save performance to file
pd.DataFrame(pairwise_distances).to_csv(
    "performance_metrics/AD_count/VMM_no_cycle_pairwise_distances_std.csv", index=False
)
pd.DataFrame(permanova_res_bc).to_csv(
    "performance_metrics/AD_count/VMM_no_cycle_permanova_braycurtis_2.csv", index=False
)
pd.DataFrame(permanova_res_ait).to_csv(
    "performance_metrics/AD_count/VMM_no_cycle_permanova_aitchison_2.csv", index=False
)

# ------------------------------------------------------------------------------------

# Set seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Set for n iteration - No cycle MoG models
n = 50

pairwise_distances = []
permanova_res_bc = []
permanova_res_ait = []

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
        kl_cycle=False,
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

    # Normalized reconstructed data
    norm_ad_recon_data = DataTransform(
        data=ad_recon_data,
        factors=[ad_count_sample_label, ad_count_batch_label, ad_count_bio_label],
        count=True,
    )

    # Pairwise distance
    mean_all, mean_intra, mean_inter = pairwise_distance_std(
        data=ad_recon_data,
        sample_label=ad_count_sample_label,
        batch_label=ad_count_batch_label,
        bio_label=ad_count_bio_label,
    )
    pairwise_res = {
        "mean_all": mean_all,
        "mean_intra": mean_intra,
        "mean_inter": mean_inter,
    }
    # Add to list
    pairwise_distances.append(pairwise_res)

    # PERMANOVA
    bc, ait = PERMANOVA(
        ad_recon_data, ad_count_sample_label, ad_count_batch_label, ad_count_bio_label
    )
    permanova_bc = {
        "F-statistic": bc["test statistic"],
        "p-value": bc["p-value"],
        "R2": bc["R2"],
    }
    permanova_ait = {
        "F-statistic": ait["test statistic"],
        "p-value": ait["p-value"],
        "R2": ait["R2"],
    }
    # Add to list
    permanova_res_bc.append(permanova_bc)
    permanova_res_ait.append(permanova_ait)

# Save performance to file
pd.DataFrame(pairwise_distances).to_csv(
    "performance_metrics/AD_count/MoG_no_cycle_pairwise_distances_std.csv", index=False
)
pd.DataFrame(permanova_res_bc).to_csv(
    "performance_metrics/AD_count/MoG_no_cycle_permanova_braycurtis_2.csv", index=False
)
pd.DataFrame(permanova_res_ait).to_csv(
    "performance_metrics/AD_count/MoG_no_cycle_permanova_aitchison_2.csv", index=False
)

# ------------------------------------------------------------------------------------

# Set seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Set for n iteration - KL cycle Normal models
n = 50

pairwise_distances = []
permanova_res_bc = []
permanova_res_ait = []

for iter in range(n):

    # VMM models
    ad_count_normal_kl_cycle = abaco_run(
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

    # Reconstruct data
    ad_recon_data = abaco_recon(
        model=ad_count_normal_kl_cycle,
        device=device,
        data=ad_count_data,
        dataloader=ad_count_train_dataloader,
        sample_label=ad_count_sample_label,
        batch_label=ad_count_batch_label,
        bio_label=ad_count_bio_label,
        seed=None,
    )

    # Normalized reconstructed data
    norm_ad_recon_data = DataTransform(
        data=ad_recon_data,
        factors=[ad_count_sample_label, ad_count_batch_label, ad_count_bio_label],
        count=True,
    )

    # Pairwise distance
    mean_all, mean_intra, mean_inter = pairwise_distance_std(
        data=ad_recon_data,
        sample_label=ad_count_sample_label,
        batch_label=ad_count_batch_label,
        bio_label=ad_count_bio_label,
    )
    pairwise_res = {
        "mean_all": mean_all,
        "mean_intra": mean_intra,
        "mean_inter": mean_inter,
    }
    # Add to list
    pairwise_distances.append(pairwise_res)

    # PERMANOVA
    bc, ait = PERMANOVA(
        ad_recon_data, ad_count_sample_label, ad_count_batch_label, ad_count_bio_label
    )
    permanova_bc = {
        "F-statistic": bc["test statistic"],
        "p-value": bc["p-value"],
        "R2": bc["R2"],
    }
    permanova_ait = {
        "F-statistic": ait["test statistic"],
        "p-value": ait["p-value"],
        "R2": ait["R2"],
    }
    # Add to list
    permanova_res_bc.append(permanova_bc)
    permanova_res_ait.append(permanova_ait)

# Save performance to file
pd.DataFrame(pairwise_distances).to_csv(
    "performance_metrics/AD_count/Std_no_cycle_pairwise_distances_std.csv", index=False
)
pd.DataFrame(permanova_res_bc).to_csv(
    "performance_metrics/AD_count/Std_no_cycle_permanova_braycurtis_2.csv", index=False
)
pd.DataFrame(permanova_res_ait).to_csv(
    "performance_metrics/AD_count/Std_no_cycle_permanova_aitchison_2.csv", index=False
)
