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
from src.ABaCo.BatchEffectDataLoader import (
    DataPreprocess,
    DataTransform,
    one_hot_encoding,
)
from src.ABaCo.BatchEffectCorrection import (
    correctCombat,
    correctLimma_rBE,
    correctBMC,
    correctPLSDAbatch_R,
    correctCombatSeq,
    correctConQuR,
)
from src.ABaCo.BatchEffectPlots import plotPCA, plotPCoA, plot_LISI_perplexity
from src.ABaCo.BatchEffectMetrics import (
    all_metrics,
    pairwise_distance,
    PERMANOVA,
    pairwise_distance_std,
    pairwise_distance_multi_run,
)
from src.ABaCo.ABaCo import (
    abaco_run,
    abaco_recon,
    abaco_run_ensemble,
    abaco_recon_ensemble,
)

# Number of iteration
n_iter = 50

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
n = n_iter

pairwise_distances = pd.DataFrame()
permanova_ait = pd.DataFrame()
permanova_bc = pd.DataFrame()

for iter in range(n):

    # VMM models
    ad_count_vmm_kl_cycle = abaco_run_ensemble(
        dataloader=ad_count_train_dataloader,
        n_batches=ad_count_batch_size,
        n_bios=ad_count_bio_size,
        input_size=ad_count_input_size,
        device=device,
        w_contra=100.0,
        kl_cycle=True,
        smooth_annealing=True,
        pre_epochs=2500,  # 2500 default
        post_epochs=5000,  # 5000 default
        vae_post_lr=2e-4,
        adv_lr=1e-7,
        disc_lr=1e-7,
        n_dec=5,
        seed=None,
    )

    # Reconstruct data
    ad_recon_data = abaco_recon_ensemble(
        model=ad_count_vmm_kl_cycle,
        device=device,
        data=ad_count_data,
        dataloader=ad_count_train_dataloader,
        sample_label=ad_count_sample_label,
        batch_label=ad_count_batch_label,
        bio_label=ad_count_bio_label,
        seed=None,
        monte_carlo=100,
    )

    # Pairwise distance
    run_pairwise_dist = pairwise_distance_multi_run(
        data=ad_recon_data,
        sample_label=ad_count_sample_label,
        batch_label=ad_count_batch_label,
        bio_label=ad_count_bio_label,
    )
    run_pairwise_dist["iter"] = iter

    # Add distance to dataframe
    pairwise_distances = pd.concat(
        [pairwise_distances, run_pairwise_dist], axis=0
    ).reset_index(drop=True)

    # Permanova
    bc, ait = PERMANOVA(
        ad_recon_data, ad_count_sample_label, ad_count_batch_label, ad_count_bio_label
    )

    bc = pd.DataFrame(
        {
            "R2": [bc["R2"]],
            "p-value": [bc["p-value"]],
            "test statistic": [bc["test statistic"]],
        }
    )
    ait = pd.DataFrame(
        {
            "R2": [ait["R2"]],
            "p-value": [ait["p-value"]],
            "test statistic": [ait["test statistic"]],
        }
    )

    permanova_bc = pd.concat([permanova_bc, bc], axis=0).reset_index(drop=True)

    permanova_ait = pd.concat([permanova_ait, ait], axis=0).reset_index(drop=True)


# Save performance to file
pd.DataFrame(pairwise_distances).to_csv(
    "performance_metrics/deterministic/AD_count/VMM_ensemble__KL_cycle_pairwise_distances.csv",
    index=False,
)
pd.DataFrame(permanova_bc).to_csv(
    "performance_metrics/deterministic/AD_count/VMM_ensemble_KL_cycle_permanova_braycurtis.csv",
    index=False,
)
pd.DataFrame(permanova_ait).to_csv(
    "performance_metrics/deterministic/AD_count/VMM_ensemble_KL_cycle_permanova_aitchison.csv",
    index=False,
)

# ------------------------------------------------------------------------------------
