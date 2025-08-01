# Essentials
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
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
    pairwise_distance_std,
    PERMANOVA,
    pairwise_distance_multi_run,
    kBET,
    ARI,
    ASW,
    iLISI_norm,
)
from src.ABaCo.ABaCo import (
    abaco_run,
    abaco_recon,
    contour_plot,
    abaco_run_ensemble,
    abaco_recon_ensemble,
)

# ------------- CREATING SIMULATED DATASETS ------------- #
# Define random seed
np.random.seed(42)

n_samples = 100
n_features = 200
p_diff = 0.75
b_diff = 0.95

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
    p_zero = np.random.uniform(0.0, 0.1, size=n_features)  # Probability of zero count

    # Baseline log-abundance and true effects
    baseline_log = np.random.normal(loc=2, scale=1, size=n_features)

    # Biological effect
    effect_sizes = np.zeros(n_features)
    effect_idx = np.random.choice(n_features, n_diff, replace=False)
    effect_sizes[effect_idx] = np.random.normal(loc=2.0, scale=4.0, size=n_diff)

    # Batch effect
    batch_sizes = np.zeros(n_features)
    batch_idx = np.random.choice(n_features, b_diff, replace=False)
    batch_sizes[batch_idx] = np.random.normal(loc=5.0, scale=2.0, size=b_diff)

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

# ------------- BATCH EFFECT CORRECTION USING OTHER METHODS ------------- #

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


# ------------- BATCH EFFECT CORRECTION USING ABACO MODELS ------------- #
# Restart seed for ABaCo
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for i in range(sims):

    df_no_bt = sim_data_no_bt[i]
    df_no_t = sim_data_no_t[i]
    df_no_b = sim_data_no_b[i]
    df_bt = sim_data_bt[i]

    # 1. Construct DataLoader object for ABaCo computation
    df_no_bt_dataloader = DataLoader(
        TensorDataset(
            torch.tensor(
                df_no_bt.select_dtypes(include="number").values, dtype=torch.float32
            ),  # samples
            one_hot_encoding(df_no_bt["Batch"])[0],  # one hot encoded batch information
            one_hot_encoding(df_no_bt["Condition"])[
                0
            ],  # one hot encoded biological information
        ),
        batch_size=1000,
    )
    df_no_t_dataloader = DataLoader(
        TensorDataset(
            torch.tensor(
                df_no_t.select_dtypes(include="number").values, dtype=torch.float32
            ),  # samples
            one_hot_encoding(df_no_t["Batch"])[0],  # one hot encoded batch information
            one_hot_encoding(df_no_t["Condition"])[
                0
            ],  # one hot encoded biological information
        ),
        batch_size=1000,
    )
    df_no_b_dataloader = DataLoader(
        TensorDataset(
            torch.tensor(
                df_no_b.select_dtypes(include="number").values, dtype=torch.float32
            ),  # samples
            one_hot_encoding(df_no_b["Batch"])[0],  # one hot encoded batch information
            one_hot_encoding(df_no_b["Condition"])[
                0
            ],  # one hot encoded biological information
        ),
        batch_size=1000,
    )
    df_bt_dataloader = DataLoader(
        TensorDataset(
            torch.tensor(
                df_bt.select_dtypes(include="number").values, dtype=torch.float32
            ),  # samples
            one_hot_encoding(df_bt["Batch"])[0],  # one hot encoded batch information
            one_hot_encoding(df_bt["Condition"])[
                0
            ],  # one hot encoded biological information
        ),
        batch_size=1000,
    )

    # 2. Run ABaCo on datasets
    # abaco_no_bt = abaco_run(
    #     dataloader=df_no_bt_dataloader,
    #     prior="Normal",
    #     n_batches=2,
    #     n_bios=2,
    #     input_size=n_features,
    #     device=device,
    #     w_contra=20.0,  # Modifiable, rather keep it on 100 as baseline
    #     kl_cycle=True,
    #     w_cycle=1e-4,
    #     seed=None,
    #     smooth_annealing=True,
    #     pre_epochs=4000,
    #     post_epochs=4000,
    #     vae_post_lr=1e-4,
    #     adv_lr=1e-6,
    #     disc_lr=1e-6,
    # )
    # abaco_no_t = abaco_run(
    #     dataloader=df_no_t_dataloader,
    #     prior="Normal",
    #     n_batches=2,
    #     n_bios=2,
    #     input_size=n_features,
    #     device=device,
    #     w_contra=20.0,  # Modifiable, rather keep it on 100 as baseline
    #     kl_cycle=True,
    #     w_cycle=1e-4,
    #     seed=None,
    #     smooth_annealing=True,
    #     pre_epochs=4000,
    #     post_epochs=4000,
    #     vae_post_lr=1e-4,
    #     adv_lr=1e-5,
    #     disc_lr=1e-5,
    # )
    # abaco_no_b = abaco_run(
    #     dataloader=df_no_b_dataloader,
    #     prior="Normal",
    #     n_batches=2,
    #     n_bios=2,
    #     input_size=n_features,
    #     device=device,
    #     w_contra=100.0,  # Modifiable, rather keep it on 100 as baseline
    #     kl_cycle=False,
    #     w_cycle=1e-4,
    #     seed=None,
    #     smooth_annealing=True,
    #     pre_epochs=4000,
    #     post_epochs=4000,
    #     vae_post_lr=1e-4,
    #     adv_lr=1e-6,
    #     disc_lr=1e-6,
    # )
    abaco_bt = abaco_run(
        dataloader=df_bt_dataloader,
        prior="Normal",
        n_batches=2,
        n_bios=2,
        input_size=n_features,
        device=device,
        w_contra=2000.0,  # Modifiable, rather keep it on 100 as baseline
        kl_cycle=False,
        w_cycle=1e-4,
        seed=None,
        smooth_annealing=True,
        pre_epochs=4000,
        post_epochs=4000,
        vae_post_lr=2e-4,
        adv_lr=5e-5,
        disc_lr=5e-5,
    )

    # 3. Reconstruct data with trained ABaCo models
    # no_bt_recon_data = abaco_recon(
    #     model=abaco_no_bt,
    #     device=device,
    #     data=df_no_bt,
    #     dataloader=df_no_bt_dataloader,
    #     sample_label="SampleID",
    #     batch_label="Batch",
    #     bio_label="Condition",
    #     seed=None,
    #     monte_carlo=1,
    # )
    # no_t_recon_data = abaco_recon(
    #     model=abaco_no_t,
    #     device=device,
    #     data=df_no_t,
    #     dataloader=df_no_t_dataloader,
    #     sample_label="SampleID",
    #     batch_label="Batch",
    #     bio_label="Condition",
    #     seed=None,
    #     monte_carlo=1,
    # )
    # no_b_recon_data = abaco_recon(
    #     model=abaco_no_b,
    #     device=device,
    #     data=df_no_b,
    #     dataloader=df_no_b_dataloader,
    #     sample_label="SampleID",
    #     batch_label="Batch",
    #     bio_label="Condition",
    #     seed=None,
    #     monte_carlo=1,
    # )
    bt_recon_data = abaco_recon(
        model=abaco_bt,
        device=device,
        data=df_bt,
        dataloader=df_bt_dataloader,
        sample_label="SampleID",
        batch_label="Batch",
        bio_label="Condition",
        seed=None,
        monte_carlo=1,
    )

    # 4. Save reconstructed datasets
    # no_bt_recon_data.to_csv(f"simulated/ABaCo/data_no_bt{i}", index=False)
    # no_t_recon_data.to_csv(f"simulated/ABaCo/data_no_t{i}", index=False)
    # no_b_recon_data.to_csv(f"simulated/ABaCo/data_no_b{i}", index=False)
    bt_recon_data.to_csv(f"simulated/ABaCo/data_bt{i}", index=False)
