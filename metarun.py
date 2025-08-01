# Essentials
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, ListedColormap
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from skbio.stats.ordination import pcoa
from skbio.stats.distance import DistanceMatrix, permanova
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
import scipy.stats as stats
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mannwhitneyu, kruskal, nbinom, multivariate_normal
from statsmodels.stats.multitest import multipletests
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import os
import re
import random
import statsmodels.api as sm
import seaborn as sns
import networkx as nx

# User libraries
from abaco.BatchEffectDataLoader import DataPreprocess, DataTransform, one_hot_encoding
from abaco.BatchEffectCorrection import (
    correctCombat,
    correctLimma_rBE,
    correctBMC,
    correctPLSDAbatch_R,
    correctCombatSeq,
    correctConQuR,
)
from abaco.BatchEffectPlots import plotPCA, plotPCoA, plot_LISI_perplexity
from abaco.BatchEffectMetrics import (
    all_metrics,
    pairwise_distance,
    pairwise_distance_std,
    PERMANOVA,
    pairwise_distance_multi_run,
    kBET,
    ARI,
    ASW,
    iLISI_norm,
)
from src.ABaCo.metaABaCo import metaABaCo

# ---- THE ORIGINAL FRAMEWORK WITH ZINB DISTRIBUTION ---- #

# # ----- AD COUNT DATA ----- #
# # Load AD count
# path = "data/dataset_ad.csv"
# ad_count_batch_label = "batch"
# ad_count_sample_label = "sample"
# ad_count_bio_label = "trt"

# ad_count_data = DataPreprocess(
#     path, factors=[ad_count_sample_label, ad_count_batch_label, ad_count_bio_label]
# )

# ad_count_batch_size = 5
# ad_count_bio_size = 2
# ad_count_input_size = ad_count_data.select_dtypes(include="number").shape[1]

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# seed = 42

# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)

# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)

# iter = 50

# for i in range(iter):

#     model = metaABaCo(
#         ad_count_data,
#         ad_count_bio_size,
#         ad_count_bio_label,
#         ad_count_batch_size,
#         ad_count_batch_label,
#         ad_count_input_size,
#         device,
#         prior="VMM",
#     )
#     model.correct()
#     recon_data = model.reconstruct()

#     recon_data.to_csv(
#         f"performance_metrics/meta_multi_runs/AD_count/vmm_model_{i}", index=False
#     )

#     if iter == 0:  # reset seed to 42
#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)

#         if torch.cuda.is_available():
#             torch.cuda.manual_seed_all(seed)

#     model = metaABaCo(
#         ad_count_data,
#         ad_count_bio_size,
#         ad_count_bio_label,
#         ad_count_batch_size,
#         ad_count_batch_label,
#         ad_count_input_size,
#         device,
#         prior="MoG",
#     )
#     model.correct()
#     recon_data = model.reconstruct()

#     recon_data.to_csv(
#         f"performance_metrics/meta_multi_runs/AD_count/mog_model_{i}", index=False
#     )

# # ----- IBD DATA ----- #
# # Load AD count
# path = "data/MGnify/IBD/IBD_dataset_genus.csv"
# ibd_batch_label = "project ID"
# ibd_sample_label = "run ID"
# ibd_bio_label = "associated phenotype"

# ibd_data = DataPreprocess(
#     path, factors=[ibd_sample_label, ibd_batch_label, ibd_bio_label]
# )

# ibd_batch_size = 2
# ibd_bio_size = 3
# ibd_input_size = ibd_data.select_dtypes(include="number").shape[1]

# seed = 42

# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)

# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)

# iter = 50

# for i in range(iter):

#     # Model setup
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = metaABaCo(
#         ibd_data,
#         ibd_bio_size,
#         ibd_bio_label,
#         ibd_batch_size,
#         ibd_batch_label,
#         ibd_input_size,
#         device,
#         prior="VMM",
#         epochs=[2000, 500, 1000],
#     )

#     # Model training
#     model.correct(
#         disc_lr=1e-7,
#         adv_lr=1e-7,
#         w_bio_penalty=100.0,
#         w_cluster_penalty=1.0,
#         phase_1_vae_lr=1e-4,
#         phase_2_vae_lr=1e-6,
#     )

#     # Model reconstruction
#     recon_data = model.reconstruct()

#     recon_data.to_csv(
#         f"performance_metrics/meta_multi_runs/IBD/vmm_model_{i}", index=False
#     )

#     # Model setup
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     model = metaABaCo(
#         ibd_data,
#         ibd_bio_size,
#         ibd_bio_label,
#         ibd_batch_size,
#         ibd_batch_label,
#         ibd_input_size,
#         device,
#         prior="VMM",
#         epochs=[2000, 500, 1000],
#     )

#     # Model training
#     model.correct(
#         disc_lr=1e-7,
#         adv_lr=1e-7,
#         w_bio_penalty=100.0,
#         w_cluster_penalty=1.0,
#         phase_1_vae_lr=1e-4,
#         phase_2_vae_lr=1e-6,
#     )

#     # Model reconstruction
#     recon_data = model.reconstruct()

#     recon_data.to_csv(
#         f"performance_metrics/meta_multi_runs/IBD/vmm_model_{i}", index=False
#     )

#     if iter == 0:  # reset seed to 42
#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)

#         if torch.cuda.is_available():
#             torch.cuda.manual_seed_all(seed)

#     model = metaABaCo(
#         ibd_data,
#         ibd_bio_size,
#         ibd_bio_label,
#         ibd_batch_size,
#         ibd_batch_label,
#         ibd_input_size,
#         device,
#         prior="MoG",
#         epochs=[2000, 500, 1000],
#     )

#     # Model training
#     model.correct(
#         disc_lr=1e-7,
#         adv_lr=1e-7,
#         w_bio_penalty=100.0,
#         w_cluster_penalty=1.0,
#         phase_1_vae_lr=1e-4,
#         phase_2_vae_lr=1e-6,
#     )

#     # Model reconstruction
#     recon_data = model.reconstruct()

#     recon_data.to_csv(
#         f"performance_metrics/meta_multi_runs/IBD/mog_model_{i}", index=False
#     )

# # ----- DTU-GE DATA ----- #
# # Load DTU-GE count
# path = "data/MGnify/DTU-GE/count/DTU-GE_phylum_count_data_filtered.csv"
# dtu_batch_label = "pipeline"
# dtu_sample_label = "accession"
# dtu_bio_label = "location"

# dtu_data = DataPreprocess(
#     path, factors=[dtu_sample_label, dtu_batch_label, dtu_bio_label]
# )

# dtu_batch_size = 2
# dtu_bio_size = 4
# dtu_input_size = dtu_data.select_dtypes(include="number").shape[1]

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# seed = 42

# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)

# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)

# iter = 50

# for i in range(iter):

#     # Model setup
#     model = metaABaCo(
#         dtu_data,
#         dtu_bio_size,
#         dtu_bio_label,
#         dtu_batch_size,
#         dtu_batch_label,
#         dtu_input_size,
#         device,
#         prior="VMM",
#         epochs=[2000, 500, 1000],
#     )

#     # Model training
#     model.correct(
#         adv_lr=1e-6,
#         disc_lr=1e-6,
#         w_bio_penalty=100.0,
#         phase_1_vae_lr=1e-4,
#         phase_2_vae_lr=1e-6,
#         phase_3_vae_lr=1e-7,
#     )

#     # Data reconstruction
#     recon_data = model.reconstruct()

#     recon_data.to_csv(
#         f"performance_metrics/meta_multi_runs/DTU-GE/vmm_model_{i}", index=False
#     )

#     if iter == 0:  # reset seed to 42
#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)

#         if torch.cuda.is_available():
#             torch.cuda.manual_seed_all(seed)

#     # Model setup
#     model = metaABaCo(
#         dtu_data,
#         dtu_bio_size,
#         dtu_bio_label,
#         dtu_batch_size,
#         dtu_batch_label,
#         dtu_input_size,
#         device,
#         prior="MoG",
#         epochs=[2000, 500, 1000],
#     )

#     # Model training
#     model.correct(
#         adv_lr=1e-6,
#         disc_lr=1e-6,
#         w_bio_penalty=100.0,
#         phase_1_vae_lr=1e-4,
#         phase_2_vae_lr=1e-6,
#         phase_3_vae_lr=1e-7,
#     )

#     # Data reconstruction
#     recon_data = model.reconstruct()

#     recon_data.to_csv(
#         f"performance_metrics/meta_multi_runs/DTU-GE/mog_model_{i}", index=False
#     )


# ---- FRAMEWORK WITH THE ZERO-INFLATED DIRICHLET DISTRIBUTION ---- #

# ----- AD COUNT DATA ----- #
# Load AD count
path = "data/dataset_ad.csv"
ad_count_batch_label = "batch"
ad_count_sample_label = "sample"
ad_count_bio_label = "trt"

ad_count_data = DataPreprocess(
    path, factors=[ad_count_sample_label, ad_count_batch_label, ad_count_bio_label]
)

ad_count_batch_size = 5
ad_count_bio_size = 2
ad_count_input_size = ad_count_data.select_dtypes(include="number").shape[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

iter = 50

for i in range(iter):

    model = metaABaCo(
        ad_count_data,
        ad_count_bio_size,
        ad_count_bio_label,
        ad_count_batch_size,
        ad_count_batch_label,
        ad_count_input_size,
        device,
        prior="VMM",
        pdist="ZIDM",
        epochs=[5000, 2000, 1000],
    )
    model.correct(
        adv_lr=1e-5,
        disc_lr=1e-5,
        w_elbo_nll=1e-2,
        phase_2_vae_lr=1e-5,
        phase_3_vae_lr=1e-6,
    )
    recon_data = model.reconstruct()

    recon_data.to_csv(
        f"performance_metrics/meta_multi_runs/ZIDM/AD_count/vmm_model_{i}", index=False
    )

    if i == 0:  # reset seed to 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    model = metaABaCo(
        ad_count_data,
        ad_count_bio_size,
        ad_count_bio_label,
        ad_count_batch_size,
        ad_count_batch_label,
        ad_count_input_size,
        device,
        prior="MoG",
        pdist="ZIDM",
        epochs=[5000, 2000, 1000],
    )
    model.correct(
        adv_lr=1e-5,
        disc_lr=1e-5,
        w_elbo_nll=1e-2,
        phase_2_vae_lr=1e-5,
        phase_3_vae_lr=1e-6,
    )
    recon_data = model.reconstruct()

    recon_data.to_csv(
        f"performance_metrics/meta_multi_runs/ZIDM/AD_count/mog_model_{i}", index=False
    )

# ----- IBD DATA ----- #
# Load AD count
path = "data/MGnify/IBD/IBD_dataset_genus.csv"
ibd_batch_label = "project ID"
ibd_sample_label = "run ID"
ibd_bio_label = "associated phenotype"

ibd_data = DataPreprocess(
    path, factors=[ibd_sample_label, ibd_batch_label, ibd_bio_label]
)

ibd_batch_size = 2
ibd_bio_size = 3
ibd_input_size = ibd_data.select_dtypes(include="number").shape[1]

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

iter = 50

for i in range(iter):

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = metaABaCo(
        ibd_data,
        ibd_bio_size,
        ibd_bio_label,
        ibd_batch_size,
        ibd_batch_label,
        ibd_input_size,
        device,
        prior="VMM",
        pdist="ZIDM",
        epochs=[3000, 1000, 1000],
    )
    model.correct(
        disc_lr=1e-7,
        adv_lr=1e-7,
        w_bio_penalty=100.0,
        w_cluster_penalty=1.0,
        phase_1_vae_lr=2e-4,
        phase_2_vae_lr=1e-6,
        w_elbo_nll=1e-2,
    )
    # Model reconstruction
    recon_data = model.reconstruct()

    recon_data.to_csv(
        f"performance_metrics/meta_multi_runs/ZIDM/IBD/vmm_model_{i}", index=False
    )

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if i == 0:  # reset seed to 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    model = metaABaCo(
        ibd_data,
        ibd_bio_size,
        ibd_bio_label,
        ibd_batch_size,
        ibd_batch_label,
        ibd_input_size,
        device,
        prior="MoG",
        pdist="ZIDM",
        epochs=[3000, 1000, 1000],
    )
    model.correct(
        disc_lr=1e-7,
        adv_lr=1e-7,
        w_bio_penalty=100.0,
        w_cluster_penalty=1.0,
        phase_1_vae_lr=2e-4,
        phase_2_vae_lr=1e-6,
        w_elbo_nll=1e-2,
    )

    # Model reconstruction
    recon_data = model.reconstruct()

    recon_data.to_csv(
        f"performance_metrics/meta_multi_runs/ZIDM/IBD/mog_model_{i}", index=False
    )

# ----- DTU-GE DATA ----- #
# Load DTU-GE count
path = "data/MGnify/DTU-GE/count/DTU-GE_phylum_count_data_filtered.csv"
dtu_batch_label = "pipeline"
dtu_sample_label = "accession"
dtu_bio_label = "location"

dtu_data = DataPreprocess(
    path, factors=[dtu_sample_label, dtu_batch_label, dtu_bio_label]
)

dtu_batch_size = 2
dtu_bio_size = 4
dtu_input_size = dtu_data.select_dtypes(include="number").shape[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

iter = 50

for i in range(iter):

    # Model setup
    model = metaABaCo(
        dtu_data,
        dtu_bio_size,
        dtu_bio_label,
        dtu_batch_size,
        dtu_batch_label,
        dtu_input_size,
        device,
        prior="VMM",
        pdist="ZIDM",
        epochs=[4000, 1000, 1000],
    )
    model.correct(
        adv_lr=1e-6,
        disc_lr=1e-6,
        w_bio_penalty=100.0,
        phase_1_vae_lr=2e-4,
        phase_2_vae_lr=1e-6,
        phase_3_vae_lr=1e-7,
        w_elbo_nll=1e-2,
    )

    # Data reconstruction
    recon_data = model.reconstruct()

    recon_data.to_csv(
        f"performance_metrics/meta_multi_runs/ZIDM/DTU-GE/vmm_model_{i}", index=False
    )

    if i == 0:  # reset seed to 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Model setup
    model = metaABaCo(
        dtu_data,
        dtu_bio_size,
        dtu_bio_label,
        dtu_batch_size,
        dtu_batch_label,
        dtu_input_size,
        device,
        prior="MoG",
        pdist="ZIDM",
        epochs=[4000, 1000, 1000],
    )
    model.correct(
        adv_lr=1e-6,
        disc_lr=1e-6,
        w_bio_penalty=100.0,
        phase_1_vae_lr=2e-4,
        phase_2_vae_lr=1e-6,
        phase_3_vae_lr=1e-7,
        w_elbo_nll=1e-2,
    )

    # Data reconstruction
    recon_data = model.reconstruct()

    recon_data.to_csv(
        f"performance_metrics/meta_multi_runs/ZIDM/DTU-GE/mog_model_{i}", index=False
    )

# ----- SIMULATED DATA ------ #

# Define random seed
np.random.seed(42)

n_samples = 100
n_features = 500
p_diff = 0.5
b_diff = 0.1

n_diff = int(n_features * p_diff)
b_diff = int(n_features * b_diff)

batches = ["Batch 1", "Batch 2"]
bios = ["A", "B"]

# Start several simulation data
sims = 50
sim_data_no_bt = []
sim_data_no_b = []
sim_data_no_t = []
sim_data_bt = []

for i in range(sims):

    # Build metadata
    metadata = pd.DataFrame(
        {
            "SampleID": [f"S{i+1}" for i in range(n_samples)],
            "Condition": np.random.choice(bios, size=n_samples, replace=True),
            "Batch": np.random.choice(batches, size=n_samples, replace=True),
        }
    )

    # Simulation settings per feature
    r = np.random.uniform(1.0, 3.0, size=n_features)  # Dispersion
    z_target = np.random.uniform(0.1, 0.5, n_features)  # e.g. 20–50% zeros

    # Baseline log-abundance and true effects
    baseline_log = np.random.normal(loc=2, scale=1, size=n_features)

    # compute per‐feature NB zero probability
    mu = np.exp(baseline_log)
    p = r / (r + mu)
    nb_zero = p**r

    # invert for p_zero
    p_zero = (z_target - nb_zero) / (1 - nb_zero)
    p_zero = np.clip(p_zero, 0, 1)

    # Biological effect
    effect_sizes = np.zeros(n_features)
    effect_idx = np.random.choice(n_features, n_diff, replace=False)
    effect_sizes[effect_idx] = np.random.normal(loc=2.0, scale=2.0, size=n_diff)

    # Batch effect
    batch_sizes = np.zeros(n_features)
    batch_idx = np.random.choice(n_features, b_diff, replace=False)
    batch_sizes[batch_idx] = np.random.normal(loc=1.0, scale=2.0, size=b_diff)

    # Simulating counts without biological effect
    counts = np.zeros((len(metadata), n_features), dtype=int)
    counts_batch = np.zeros((len(metadata), n_features), dtype=int)
    for i, row in metadata.iterrows():
        mu = baseline_log
        mu_batch = mu + (
            batch_sizes if row["Batch"] == batches[1] else 0
        )  # Batch shift on Batch 2
        p = r / (r + np.exp(mu))
        p_batch = r / (r + np.exp(mu_batch))
        is_zero = np.random.binomial(1, p_zero)
        counts_i = nbinom(r, p).rvs()
        counts[i, :] = np.where(is_zero == 1, 0, counts_i)
        counts_i_batch = nbinom(r, p_batch).rvs()
        counts_batch[i, :] = np.where(is_zero == 1, 0, counts_i_batch)

    # Case 1: No batch, no bio
    count_matrix_no_bt = pd.DataFrame(
        counts, columns=[f"OTU{j+1}" for j in range(n_features)]
    )
    count_matrix_no_bt = pd.concat([metadata, count_matrix_no_bt], axis=1)
    # Case 2: Batch, no bio
    count_matrix_no_t = pd.DataFrame(
        counts_batch, columns=[f"OTU{j+1}" for j in range(n_features)]
    )
    count_matrix_no_t = pd.concat([metadata, count_matrix_no_t], axis=1)

    sim_data_no_bt.append(count_matrix_no_bt)
    sim_data_no_t.append(count_matrix_no_t)

    # Simulating counts with biological effect
    counts = np.zeros((len(metadata), n_features), dtype=int)
    counts_batch = np.zeros((len(metadata), n_features), dtype=int)
    for i, row in metadata.iterrows():
        mu = baseline_log + (
            effect_sizes if row["Condition"] == bios[1] else 0
        )  # Biological shift on group B
        mu_batch = mu + (
            batch_sizes if row["Batch"] == batches[1] else 0
        )  # Batch shift on Batch 2
        p = r / (r + np.exp(mu))
        p_batch = r / (r + np.exp(mu_batch))
        is_zero = np.random.binomial(1, p_zero)
        counts_i = nbinom(r, p).rvs()
        counts[i, :] = np.where(is_zero == 1, 0, counts_i)
        counts_i_batch = nbinom(r, p_batch).rvs()
        counts_batch[i, :] = np.where(is_zero == 1, 0, counts_i_batch)

    # Case 3: No batch, bio
    count_matrix_no_b = pd.DataFrame(
        counts, columns=[f"OTU{j+1}" for j in range(n_features)]
    )
    count_matrix_no_b = pd.concat([metadata, count_matrix_no_b], axis=1)
    # Case 4: Batch, bio
    count_matrix_bt = pd.DataFrame(
        counts_batch, columns=[f"OTU{j+1}" for j in range(n_features)]
    )
    count_matrix_bt = pd.concat([metadata, count_matrix_bt], axis=1)

    sim_data_no_b.append(count_matrix_no_b)
    sim_data_bt.append(count_matrix_bt)

for i in range(sims):

    sim_data = sim_data_no_bt[i]
    model = metaABaCo(
        sim_data,
        2,
        "Condition",
        2,
        "Batch",
        500,
        device,
        prior="VMM",
        pdist="ZIDM",
        epochs=[5000, 1000, 1000],
    )
    model.correct(
        adv_lr=1e-5,
        disc_lr=1e-5,
        phase_1_vae_lr=1e-4,
        phase_2_vae_lr=1e-5,
        phase_3_vae_lr=1e-5,
        w_elbo_nll=1e-3,
        w_bio_penalty=100.0,
        w_cluster_penalty=100.0,
        seed=42,
    )
    recon_data = model.reconstruct(seed=42)

    recon_data.to_csv(f"simulated/ABaCo_ZIDM/vmm_data_no_bt_{i}")

    sim_data = sim_data_no_t[i]
    model = metaABaCo(
        sim_data,
        2,
        "Condition",
        2,
        "Batch",
        500,
        device,
        prior="VMM",
        pdist="ZIDM",
        epochs=[5000, 1000, 1000],
    )
    model.correct(
        adv_lr=1e-5,
        disc_lr=1e-5,
        phase_1_vae_lr=1e-4,
        phase_2_vae_lr=1e-5,
        phase_3_vae_lr=1e-5,
        w_elbo_nll=1e-3,
        w_bio_penalty=100.0,
        w_cluster_penalty=100.0,
        seed=42,
    )
    recon_data = model.reconstruct(seed=42)

    recon_data.to_csv(f"simulated/ABaCo_ZIDM/vmm_data_no_t_{i}")

    sim_data = sim_data_no_b[i]
    model = metaABaCo(
        sim_data,
        2,
        "Condition",
        2,
        "Batch",
        500,
        device,
        prior="VMM",
        pdist="ZIDM",
        epochs=[5000, 1000, 1000],
    )
    model.correct(
        adv_lr=1e-5,
        disc_lr=1e-5,
        phase_1_vae_lr=1e-4,
        phase_2_vae_lr=1e-5,
        phase_3_vae_lr=1e-5,
        w_elbo_nll=1e-3,
        w_bio_penalty=100.0,
        w_cluster_penalty=100.0,
        seed=42,
    )
    recon_data = model.reconstruct(seed=42)

    recon_data.to_csv(f"simulated/ABaCo_ZIDM/vmm_data_no_b_{i}")

    sim_data = sim_data_bt[i]
    model = metaABaCo(
        sim_data,
        2,
        "Condition",
        2,
        "Batch",
        500,
        device,
        prior="VMM",
        pdist="ZIDM",
        epochs=[5000, 1000, 1000],
    )
    model.correct(
        adv_lr=1e-5,
        disc_lr=1e-5,
        phase_1_vae_lr=1e-4,
        phase_2_vae_lr=1e-5,
        phase_3_vae_lr=1e-5,
        w_elbo_nll=1e-3,
        w_bio_penalty=100.0,
        w_cluster_penalty=100.0,
        seed=42,
    )
    recon_data = model.reconstruct(seed=42)

    recon_data.to_csv(f"simulated/ABaCo_ZIDM/vmm_data_bt_{i}")

    sim_data = sim_data_no_bt[i]
    model = metaABaCo(
        sim_data,
        2,
        "Condition",
        2,
        "Batch",
        500,
        device,
        prior="MoG",
        pdist="ZIDM",
        epochs=[5000, 1000, 1000],
    )
    model.correct(
        adv_lr=1e-5,
        disc_lr=1e-5,
        phase_1_vae_lr=1e-4,
        phase_2_vae_lr=1e-5,
        phase_3_vae_lr=1e-5,
        w_elbo_nll=1e-3,
        w_bio_penalty=100.0,
        w_cluster_penalty=100.0,
        seed=42,
    )
    recon_data = model.reconstruct(seed=42, mask=False)

    recon_data.to_csv(f"simulated/ABaCo_ZIDM/mog_data_no_bt_{i}")

    sim_data = sim_data_no_t[i]
    model = metaABaCo(
        sim_data,
        2,
        "Condition",
        2,
        "Batch",
        500,
        device,
        prior="MoG",
        pdist="ZIDM",
        epochs=[5000, 1000, 1000],
    )
    model.correct(
        adv_lr=1e-5,
        disc_lr=1e-5,
        phase_1_vae_lr=1e-4,
        phase_2_vae_lr=1e-5,
        phase_3_vae_lr=1e-5,
        w_elbo_nll=1e-3,
        w_bio_penalty=100.0,
        w_cluster_penalty=100.0,
        seed=42,
    )
    recon_data = model.reconstruct(seed=42)

    recon_data.to_csv(f"simulated/ABaCo_ZIDM/mog_data_no_t_{i}")

    sim_data = sim_data_no_b[i]
    model = metaABaCo(
        sim_data,
        2,
        "Condition",
        2,
        "Batch",
        500,
        device,
        prior="MoG",
        pdist="ZIDM",
        epochs=[5000, 1000, 1000],
    )
    model.correct(
        adv_lr=1e-5,
        disc_lr=1e-5,
        phase_1_vae_lr=1e-4,
        phase_2_vae_lr=1e-5,
        phase_3_vae_lr=1e-5,
        w_elbo_nll=1e-3,
        w_bio_penalty=100.0,
        w_cluster_penalty=100.0,
        seed=42,
    )
    recon_data = model.reconstruct(seed=42, mask=False)

    recon_data.to_csv(f"simulated/ABaCo_ZIDM/mog_data_no_b_{i}")

    sim_data = sim_data_bt[i]
    model = metaABaCo(
        sim_data,
        2,
        "Condition",
        2,
        "Batch",
        500,
        device,
        prior="MoG",
        pdist="ZIDM",
        epochs=[5000, 1000, 1000],
    )
    model.correct(
        adv_lr=1e-5,
        disc_lr=1e-5,
        phase_1_vae_lr=1e-4,
        phase_2_vae_lr=1e-5,
        phase_3_vae_lr=1e-5,
        w_elbo_nll=1e-3,
        w_bio_penalty=100.0,
        w_cluster_penalty=100.0,
        seed=42,
    )
    recon_data = model.reconstruct(seed=42)

    recon_data.to_csv(f"simulated/ABaCo_ZIDM/mog_data_bt_{i}")

for i in range(sims):

    sim_data = sim_data_no_bt[i]
    model = metaABaCo(
        sim_data,
        2,
        "Condition",
        2,
        "Batch",
        500,
        device,
        prior="VMM",
        pdist="ZINB",
        epochs=[5000, 1000, 1000],
    )
    model.correct(
        adv_lr=1e-5,
        disc_lr=1e-5,
        phase_1_vae_lr=1e-4,
        phase_2_vae_lr=1e-5,
        phase_3_vae_lr=1e-5,
        w_bio_penalty=100.0,
        w_cluster_penalty=100.0,
        seed=42,
    )
    recon_data = model.reconstruct(seed=42)

    recon_data.to_csv(f"simulated/ABaCo_ZINB/vmm_data_no_bt_{i}")

    sim_data = sim_data_no_t[i]
    model = metaABaCo(
        sim_data,
        2,
        "Condition",
        2,
        "Batch",
        500,
        device,
        prior="VMM",
        pdist="ZINB",
        epochs=[5000, 1000, 1000],
    )
    model.correct(
        adv_lr=1e-5,
        disc_lr=1e-5,
        phase_1_vae_lr=1e-4,
        phase_2_vae_lr=1e-5,
        phase_3_vae_lr=1e-5,
        w_bio_penalty=100.0,
        w_cluster_penalty=100.0,
        seed=42,
    )
    recon_data = model.reconstruct(seed=42)

    recon_data.to_csv(f"simulated/ABaCo_ZINB/vmm_data_no_t_{i}")

    sim_data = sim_data_no_b[i]
    model = metaABaCo(
        sim_data,
        2,
        "Condition",
        2,
        "Batch",
        500,
        device,
        prior="VMM",
        pdist="ZINB",
        epochs=[5000, 1000, 1000],
    )
    model.correct(
        adv_lr=1e-5,
        disc_lr=1e-5,
        phase_1_vae_lr=1e-4,
        phase_2_vae_lr=1e-5,
        phase_3_vae_lr=1e-5,
        w_bio_penalty=100.0,
        w_cluster_penalty=100.0,
        seed=42,
    )
    recon_data = model.reconstruct(seed=42)

    recon_data.to_csv(f"simulated/ABaCo_ZINB/vmm_data_no_b_{i}")

    sim_data = sim_data_bt[i]
    model = metaABaCo(
        sim_data,
        2,
        "Condition",
        2,
        "Batch",
        500,
        device,
        prior="VMM",
        pdist="ZINB",
        epochs=[5000, 1000, 1000],
    )
    model.correct(
        adv_lr=1e-5,
        disc_lr=1e-5,
        phase_1_vae_lr=1e-4,
        phase_2_vae_lr=1e-5,
        phase_3_vae_lr=1e-5,
        w_bio_penalty=100.0,
        w_cluster_penalty=100.0,
        seed=42,
    )
    recon_data = model.reconstruct(seed=42)

    recon_data.to_csv(f"simulated/ABaCo_ZINB/vmm_data_bt_{i}")

    sim_data = sim_data_no_bt[i]
    model = metaABaCo(
        sim_data,
        2,
        "Condition",
        2,
        "Batch",
        500,
        device,
        prior="MoG",
        pdist="ZINB",
        epochs=[5000, 1000, 1000],
    )
    model.correct(
        adv_lr=1e-5,
        disc_lr=1e-5,
        phase_1_vae_lr=1e-4,
        phase_2_vae_lr=1e-5,
        phase_3_vae_lr=1e-5,
        w_bio_penalty=100.0,
        w_cluster_penalty=100.0,
        seed=42,
    )
    recon_data = model.reconstruct(seed=42, mask=False)

    recon_data.to_csv(f"simulated/ABaCo_ZINB/mog_data_no_bt_{i}")

    sim_data = sim_data_no_t[i]
    model = metaABaCo(
        sim_data,
        2,
        "Condition",
        2,
        "Batch",
        500,
        device,
        prior="MoG",
        pdist="ZINB",
        epochs=[5000, 1000, 1000],
    )
    model.correct(
        adv_lr=1e-5,
        disc_lr=1e-5,
        phase_1_vae_lr=1e-4,
        phase_2_vae_lr=1e-5,
        phase_3_vae_lr=1e-5,
        w_bio_penalty=100.0,
        w_cluster_penalty=100.0,
        seed=42,
    )
    recon_data = model.reconstruct(seed=42)

    recon_data.to_csv(f"simulated/ABaCo_ZINB/mog_data_no_t_{i}")

    sim_data = sim_data_no_b[i]
    model = metaABaCo(
        sim_data,
        2,
        "Condition",
        2,
        "Batch",
        500,
        device,
        prior="MoG",
        pdist="ZINB",
        epochs=[5000, 1000, 1000],
    )
    model.correct(
        adv_lr=1e-5,
        disc_lr=1e-5,
        phase_1_vae_lr=1e-4,
        phase_2_vae_lr=1e-5,
        phase_3_vae_lr=1e-5,
        w_bio_penalty=100.0,
        w_cluster_penalty=100.0,
        seed=42,
    )
    recon_data = model.reconstruct(seed=42, mask=False)

    recon_data.to_csv(f"simulated/ABaCo_ZINB/mog_data_no_b_{i}")

    sim_data = sim_data_bt[i]
    model = metaABaCo(
        sim_data,
        2,
        "Condition",
        2,
        "Batch",
        500,
        device,
        prior="MoG",
        pdist="ZINB",
        epochs=[5000, 1000, 1000],
    )
    model.correct(
        adv_lr=1e-5,
        disc_lr=1e-5,
        phase_1_vae_lr=1e-4,
        phase_2_vae_lr=1e-5,
        phase_3_vae_lr=1e-5,
        w_bio_penalty=100.0,
        w_cluster_penalty=100.0,
        seed=42,
    )
    recon_data = model.reconstruct(seed=42)

    recon_data.to_csv(f"simulated/ABaCo_ZINB/mog_data_bt_{i}")

# # ------------- BATCH EFFECT CORRECTION USING OTHER METHODS ------------- #

# for i in range(sims):

#     df_no_bt = sim_data_no_bt[i]
#     df_no_t = sim_data_no_t[i]
#     df_no_b = sim_data_no_b[i]
#     df_bt = sim_data_bt[i]

#     # 1. Transform data to CLR space (for the corresponding)
#     df_no_bt_clr = DataTransform(
#         df_no_bt, factors=["SampleID", "Batch", "Condition"], count=True
#     )
#     df_no_t_clr = DataTransform(
#         df_no_t, factors=["SampleID", "Batch", "Condition"], count=True
#     )
#     df_no_b_clr = DataTransform(
#         df_no_b, factors=["SampleID", "Batch", "Condition"], count=True
#     )
#     df_bt_clr = DataTransform(
#         df_bt, factors=["SampleID", "Batch", "Condition"], count=True
#     )

#     # 2. Compute batch effect correction for other methods
#     bmc_no_bt = correctBMC(
#         df_no_bt_clr,
#         sample_label="SampleID",
#         batch_label="Batch",
#         exp_label="Condition",
#     )
#     bmc_no_t = correctBMC(
#         df_no_t_clr,
#         sample_label="SampleID",
#         batch_label="Batch",
#         exp_label="Condition",
#     )
#     bmc_no_b = correctBMC(
#         df_no_b_clr,
#         sample_label="SampleID",
#         batch_label="Batch",
#         exp_label="Condition",
#     )
#     bmc_bt = correctBMC(
#         df_bt_clr,
#         sample_label="SampleID",
#         batch_label="Batch",
#         exp_label="Condition",
#     )

#     combat_no_bt = correctCombat(
#         data=df_no_bt_clr,
#         sample_label="SampleID",
#         batch_label="Batch",
#         experiment_label="Condition",
#     )
#     combat_no_t = correctCombat(
#         data=df_no_t_clr,
#         sample_label="SampleID",
#         batch_label="Batch",
#         experiment_label="Condition",
#     )
#     combat_no_b = correctCombat(
#         data=df_no_b_clr,
#         sample_label="SampleID",
#         batch_label="Batch",
#         experiment_label="Condition",
#     )
#     combat_bt = correctCombat(
#         data=df_bt_clr,
#         sample_label="SampleID",
#         batch_label="Batch",
#         experiment_label="Condition",
#     )

#     limma_no_bt = correctLimma_rBE(
#         df_no_bt_clr,
#         sample_label="SampleID",
#         batch_label="Batch",
#         covariates_labels=["Condition"],
#     )
#     limma_no_t = correctLimma_rBE(
#         df_no_t_clr,
#         sample_label="SampleID",
#         batch_label="Batch",
#         covariates_labels=["Condition"],
#     )
#     limma_no_b = correctLimma_rBE(
#         df_no_b_clr,
#         sample_label="SampleID",
#         batch_label="Batch",
#         covariates_labels=["Condition"],
#     )
#     limma_bt = correctLimma_rBE(
#         df_bt_clr,
#         sample_label="SampleID",
#         batch_label="Batch",
#         covariates_labels=["Condition"],
#     )

#     plsda_no_bt = correctPLSDAbatch_R(
#         df_no_bt_clr,
#         sample_label="SampleID",
#         batch_label="Batch",
#         exp_label="Condition",
#         ncomp_bat=1,
#         ncomp_trt=1,
#     )
#     plsda_no_t = correctPLSDAbatch_R(
#         df_no_t_clr,
#         sample_label="SampleID",
#         batch_label="Batch",
#         exp_label="Condition",
#         ncomp_bat=1,
#         ncomp_trt=1,
#     )
#     plsda_no_b = correctPLSDAbatch_R(
#         df_no_b_clr,
#         sample_label="SampleID",
#         batch_label="Batch",
#         exp_label="Condition",
#         ncomp_bat=1,
#         ncomp_trt=1,
#     )
#     plsda_bt = correctPLSDAbatch_R(
#         df_bt_clr,
#         sample_label="SampleID",
#         batch_label="Batch",
#         exp_label="Condition",
#         ncomp_bat=1,
#         ncomp_trt=1,
#     )

#     combatseq_no_bt = correctCombatSeq(
#         df_no_bt,
#         sample_label="SampleID",
#         batch_label="Batch",
#         condition_label="Condition",
#     )
#     combatseq_no_t = correctCombatSeq(
#         df_no_t,
#         sample_label="SampleID",
#         batch_label="Batch",
#         condition_label="Condition",
#     )
#     combatseq_no_b = correctCombatSeq(
#         df_no_b,
#         sample_label="SampleID",
#         batch_label="Batch",
#         condition_label="Condition",
#     )
#     combatseq_bt = correctCombatSeq(
#         df_bt,
#         sample_label="SampleID",
#         batch_label="Batch",
#         condition_label="Condition",
#     )

#     conqur_no_bt = correctConQuR(
#         df_no_bt,
#         batch_cols=["Batch"],
#         covariate_cols=["Condition"],
#     )
#     conqur_no_t = correctConQuR(
#         df_no_t,
#         batch_cols=["Batch"],
#         covariate_cols=["Condition"],
#     )
#     conqur_no_b = correctConQuR(
#         df_no_b,
#         batch_cols=["Batch"],
#         covariate_cols=["Condition"],
#     )
#     conqur_bt = correctConQuR(
#         df_bt,
#         batch_cols=["Batch"],
#         covariate_cols=["Condition"],
#     )

#     # 3. Save count data to folder
#     bmc_no_bt.to_csv(f"simulated/BMC/data_no_bt_{i}", index=False)
#     bmc_no_t.to_csv(f"simulated/BMC/data_no_t_{i}", index=False)
#     bmc_no_b.to_csv(f"simulated/BMC/data_no_b_{i}", index=False)
#     bmc_bt.to_csv(f"simulated/BMC/data_bt_{i}", index=False)

#     combat_no_bt.to_csv(f"simulated/ComBat/data_no_bt_{i}", index=False)
#     combat_no_t.to_csv(f"simulated/ComBat/data_no_t_{i}", index=False)
#     combat_no_b.to_csv(f"simulated/ComBat/data_no_b_{i}", index=False)
#     combat_bt.to_csv(f"simulated/ComBat/data_bt_{i}", index=False)

#     limma_no_bt.to_csv(f"simulated/limma/data_no_bt_{i}", index=False)
#     limma_no_t.to_csv(f"simulated/limma/data_no_t_{i}", index=False)
#     limma_no_b.to_csv(f"simulated/limma/data_no_b_{i}", index=False)
#     limma_bt.to_csv(f"simulated/limma/data_bt_{i}", index=False)

#     plsda_no_bt.to_csv(f"simulated/PLSDA/data_no_bt_{i}", index=False)
#     plsda_no_t.to_csv(f"simulated/PLSDA/data_no_t_{i}", index=False)
#     plsda_no_b.to_csv(f"simulated/PLSDA/data_no_b_{i}", index=False)
#     plsda_bt.to_csv(f"simulated/PLSDA/data_bt_{i}", index=False)

#     combatseq_no_bt.to_csv(f"simulated/ComBat-seq/data_no_bt_{i}", index=False)
#     combatseq_no_t.to_csv(f"simulated/ComBat-seq/data_no_t_{i}", index=False)
#     combatseq_no_b.to_csv(f"simulated/ComBat-seq/data_no_b_{i}", index=False)
#     combatseq_bt.to_csv(f"simulated/ComBat-seq/data_bt_{i}", index=False)

#     conqur_no_bt.to_csv(f"simulated/ConQuR/data_no_bt_{i}", index=False)
#     conqur_no_t.to_csv(f"simulated/ConQuR/data_no_t_{i}", index=False)
#     conqur_no_b.to_csv(f"simulated/ConQuR/data_no_b_{i}", index=False)
#     conqur_bt.to_csv(f"simulated/ConQuR/data_bt_{i}", index=False)
