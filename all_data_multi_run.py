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
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import time
import psutil
import GPUtil

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
    cLISI_full_rank,
    iLISI_full_rank,
    PERMANOVA,
)
from src.ABaCo.ABaCo import abaco_run, abaco_recon


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
path = "data/MGnify/IBD/IBD_dataset_genus.csv"
ibd_batch_label = "project ID"
ibd_sample_label = "run ID"
ibd_bio_label = "associated phenotype"

ibd_data = DataPreprocess(
    path, factors=[ibd_sample_label, ibd_batch_label, ibd_bio_label]
)

# train DataLoader: [samples, ohe_batch]
ibd_input_size = 193
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
path = "data/MGnify/DTU-GE/count/DTU-GE_phylum_count_data_filtered.csv"
dtu_batch_label = "pipeline"
dtu_sample_label = "accession"
dtu_bio_label = "location"

dtu_data = DataPreprocess(
    path, factors=[dtu_sample_label, dtu_batch_label, dtu_bio_label]
)

# train DataLoader: [samples, ohe_batch]
dtu_input_size = 162
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
dtu_bio_size = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The real loop - over all ABaCo models
n = 50

for iter in range(n):

    # Set seed
    if iter == 0:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    performances_batch_ad = []
    performances_ilisi_ad = pd.DataFrame()
    performances_clisi_ad = pd.DataFrame()
    permanova_ait_ad = pd.DataFrame()
    permanova_bc_ad = pd.DataFrame()

    performances_batch_ibd = []
    performances_ilisi_ibd = pd.DataFrame()
    performances_clisi_ibd = pd.DataFrame()
    permanova_ait_ibd = pd.DataFrame()
    permanova_bc_ibd = pd.DataFrame()

    performances_batch_dtu = []
    performances_ilisi_dtu = pd.DataFrame()
    performances_clisi_dtu = pd.DataFrame()
    permanova_ait_dtu = pd.DataFrame()
    permanova_bc_dtu = pd.DataFrame()

    performances_mannwhit_ad = pd.DataFrame()
    performances_mannwhit_ibd = pd.DataFrame()
    performances_mannwhit_dtu = pd.DataFrame()

    # VMM models

    # for performance metrics
    # proc = psutil.Process()
    # # --- pre‐run snapshots ---
    # t0_wall = time.time()
    # t0_cpu = sum(proc.cpu_times()[:2])  # user + sys CPU s
    # mem0 = proc.memory_info().rss  # resident set size
    # torch.cuda.reset_peak_memory_stats()  # clear PyTorch peak

    # ad_count_vmm_kl_cycle = abaco_run(
    #     dataloader=ad_count_train_dataloader,
    #     n_batches=ad_count_batch_size,
    #     n_bios=ad_count_bio_size,
    #     input_size=ad_count_input_size,
    #     device=device,
    #     w_contra=100.0,
    #     kl_cycle=True,
    #     seed=None,
    #     smooth_annealing=True,
    #     pre_epochs=2500,
    #     post_epochs=5000,
    #     vae_post_lr=2e-4,
    #     adv_lr=1e-7,
    #     disc_lr=1e-7,
    # )

    # # --- post‐run snapshots ---
    # t1_wall = time.time()
    # t1_cpu = sum(proc.cpu_times()[:2])
    # mem1 = proc.memory_info().rss
    # peak_gpu = torch.cuda.max_memory_allocated()

    # # --- GPU‐side current status (optional) ---
    # gpus = GPUtil.getGPUs()

    # # --- print a neat summary ---
    # print("===== AD count - VMM KL cycle Performance =====")
    # print(f"Wall‐clock time  : {t1_wall - t0_wall:.1f} s")
    # print(f"CPU time         : {t1_cpu  - t0_cpu :.1f} s")
    # print(f"RAM change       : {(mem1 - mem0)/1024**3:+.2f} GiB")
    # print(f"Peak GPU memory : {peak_gpu/1024**3:.2f} GiB")
    # for g in gpus:
    #     print(
    #         f" GPU {g.id}: util={g.load*100:.1f}%, mem={g.memoryUsed/1024**3:.1f}/{g.memoryTotal/1024**3:.1f} GiB"
    #     )

    if iter == 0:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # for performance metrics
    proc = psutil.Process()
    # --- pre‐run snapshots ---
    t0_wall = time.time()
    t0_cpu = sum(proc.cpu_times()[:2])  # user + sys CPU s
    mem0 = proc.memory_info().rss  # resident set size
    torch.cuda.reset_peak_memory_stats()  # clear PyTorch peak

    ibd_vmm_kl_cycle = abaco_run(
        dataloader=ibd_train_dataloader,
        n_batches=ibd_batch_size,
        n_bios=ibd_bio_size,
        input_size=ibd_input_size,
        device=device,
        w_contra=1000.0,
        kl_cycle=True,
        w_cycle=1e-3,
        seed=None,
        smooth_annealing=True,
        pre_epochs=5000,
        post_epochs=2500,
        vae_pre_lr=1e-4,
        vae_post_lr=1e-4,
        encoder_net=[2048, 1024, 512, 256],
        decoder_net=[256, 512, 1024, 2048],
        adv_lr=1e-6,
        disc_lr=1e-6,
        new_pre_train=True,
    )

    # --- post‐run snapshots ---
    t1_wall = time.time()
    t1_cpu = sum(proc.cpu_times()[:2])
    mem1 = proc.memory_info().rss
    peak_gpu = torch.cuda.max_memory_allocated()

    # --- GPU‐side current status (optional) ---
    gpus = GPUtil.getGPUs()

    # --- print a neat summary ---
    print("===== IBD - VMM KL cycle Performance =====")
    print(f"Wall‐clock time  : {t1_wall - t0_wall:.1f} s")
    print(f"CPU time         : {t1_cpu  - t0_cpu :.1f} s")
    print(f"RAM change       : {(mem1 - mem0)/1024**3:+.2f} GiB")
    print(f"Peak GPU memory : {peak_gpu/1024**3:.2f} GiB")
    for g in gpus:
        print(
            f" GPU {g.id}: util={g.load*100:.1f}%, mem={g.memoryUsed/1024**3:.1f}/{g.memoryTotal/1024**3:.1f} GiB"
        )

    if iter == 0:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # # for performance metrics
    # proc = psutil.Process()
    # # --- pre‐run snapshots ---
    # t0_wall = time.time()
    # t0_cpu = sum(proc.cpu_times()[:2])  # user + sys CPU s
    # mem0 = proc.memory_info().rss  # resident set size
    # torch.cuda.reset_peak_memory_stats()  # clear PyTorch peak

    # dtu_vmm_kl_cycle = abaco_run(
    #     dataloader=dtu_train_dataloader,
    #     n_batches=dtu_batch_size,
    #     n_bios=dtu_bio_size,
    #     input_size=dtu_input_size,
    #     device=device,
    #     w_contra=1000.0,
    #     kl_cycle=True,
    #     seed=None,
    #     smooth_annealing=True,
    #     pre_epochs=5000,
    #     post_epochs=2500,
    #     vae_pre_lr=1e-4,
    #     vae_post_lr=2e-4,
    #     adv_lr=1e-6,
    #     disc_lr=1e-6,
    #     w_cycle=1e-3,
    # )

    # # --- post‐run snapshots ---
    # t1_wall = time.time()
    # t1_cpu = sum(proc.cpu_times()[:2])
    # mem1 = proc.memory_info().rss
    # peak_gpu = torch.cuda.max_memory_allocated()

    # # --- GPU‐side current status (optional) ---
    # gpus = GPUtil.getGPUs()

    # # --- print a neat summary ---
    # print("===== DTU-GE - VMM KL cycle Performance =====")
    # print(f"Wall‐clock time  : {t1_wall - t0_wall:.1f} s")
    # print(f"CPU time         : {t1_cpu  - t0_cpu :.1f} s")
    # print(f"RAM change       : {(mem1 - mem0)/1024**3:+.2f} GiB")
    # print(f"Peak GPU memory : {peak_gpu/1024**3:.2f} GiB")
    # for g in gpus:
    #     print(
    #         f" GPU {g.id}: util={g.load*100:.1f}%, mem={g.memoryUsed/1024**3:.1f}/{g.memoryTotal/1024**3:.1f} GiB"
    #     )

    if iter == 0:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Reconstruct data
    # ad_recon_data = abaco_recon(
    #     model=ad_count_vmm_kl_cycle,
    #     device=device,
    #     data=ad_count_data,
    #     dataloader=ad_count_train_dataloader,
    #     sample_label=ad_count_sample_label,
    #     batch_label=ad_count_batch_label,
    #     bio_label=ad_count_bio_label,
    #     seed=None,
    #     monte_carlo=1,
    # )

    if iter == 0:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    ibd_recon_data = abaco_recon(
        model=ibd_vmm_kl_cycle,
        device=device,
        data=ibd_data,
        dataloader=ibd_train_dataloader,
        sample_label=ibd_sample_label,
        batch_label=ibd_batch_label,
        bio_label=ibd_bio_label,
        seed=None,
        monte_carlo=1,
    )

    if iter == 0:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # dtu_recon_data = abaco_recon(
    #     model=dtu_vmm_kl_cycle,
    #     device=device,
    #     data=dtu_data,
    #     dataloader=dtu_train_dataloader,
    #     sample_label=dtu_sample_label,
    #     batch_label=dtu_batch_label,
    #     bio_label=dtu_bio_label,
    #     seed=None,
    #     monte_carlo=1,
    # )

    # ad_recon_data.to_csv(
    #     f"performance_metrics/multi_runs/AD_count/vmm_kl_recon_{iter}", index=False
    # )
    ibd_recon_data.to_csv(
        f"performance_metrics/multi_runs/IBD/vmm_kl_recon_new_{iter}",
        index=False,
    )
    # dtu_recon_data.to_csv(
    #     f"performance_metrics/multi_runs/DTU-GE/vmm_kl_recon_kl_annealing_{iter}",
    #     index=False,
    # )

    # # LISI tables
    # clisi_ad = cLISI_full_rank(ad_recon_data, ad_count_bio_label)
    # ilisi_ad = iLISI_full_rank(ad_recon_data, ad_count_batch_label)

    # clisi_ad["iter"] = iter
    # ilisi_ad["iter"] = iter

    # performances_clisi_ad = pd.concat(
    #     [performances_clisi_ad, clisi_ad], axis=0
    # ).reset_index(drop=True)
    # performances_ilisi_ad = pd.concat(
    #     [performances_ilisi_ad, ilisi_ad], axis=0
    # ).reset_index(drop=True)

    # clisi_ibd = cLISI_full_rank(ibd_recon_data, ibd_bio_label)
    # ilisi_ibd = iLISI_full_rank(ibd_recon_data, ibd_batch_label)

    # clisi_ibd["iter"] = iter
    # ilisi_ibd["iter"] = iter

    # performances_clisi_ibd = pd.concat(
    #     [performances_clisi_ibd, clisi_ibd], axis=0
    # ).reset_index(drop=True)
    # performances_ilisi_ibd = pd.concat(
    #     [performances_ilisi_ibd, ilisi_ibd], axis=0
    # ).reset_index(drop=True)

    # clisi_dtu = cLISI_full_rank(dtu_recon_data, dtu_bio_label)
    # ilisi_dtu = iLISI_full_rank(dtu_recon_data, dtu_batch_label)

    # clisi_dtu["iter"] = iter
    # ilisi_dtu["iter"] = iter

    # performances_clisi_dtu = pd.concat(
    #     [performances_clisi_dtu, clisi_dtu], axis=0
    # ).reset_index(drop=True)
    # performances_ilisi_dtu = pd.concat(
    #     [performances_ilisi_dtu, ilisi_dtu], axis=0
    # ).reset_index(drop=True)

    # # PERMANOVA
    # bc, ait = PERMANOVA(
    #     ad_recon_data, ad_count_sample_label, ad_count_batch_label, ad_count_bio_label
    # )

    # bc = pd.DataFrame(
    #     {
    #         "R2": [bc["R2"]],
    #         "p-value": [bc["p-value"]],
    #         "test statistic": [bc["test statistic"]],
    #     }
    # )
    # ait = pd.DataFrame(
    #     {
    #         "R2": [ait["R2"]],
    #         "p-value": [ait["p-value"]],
    #         "test statistic": [ait["test statistic"]],
    #     }
    # )

    # permanova_bc_ad = pd.concat([permanova_bc_ad, bc], axis=0).reset_index(drop=True)

    # permanova_ait_ad = pd.concat([permanova_ait_ad, ait], axis=0).reset_index(drop=True)

    # bc, ait = PERMANOVA(
    #     ibd_recon_data, ibd_sample_label, ibd_batch_label, ibd_bio_label
    # )

    # bc = pd.DataFrame(
    #     {
    #         "R2": [bc["R2"]],
    #         "p-value": [bc["p-value"]],
    #         "test statistic": [bc["test statistic"]],
    #     }
    # )
    # ait = pd.DataFrame(
    #     {
    #         "R2": [ait["R2"]],
    #         "p-value": [ait["p-value"]],
    #         "test statistic": [ait["test statistic"]],
    #     }
    # )

    # permanova_bc_ibd = pd.concat([permanova_bc_ibd, bc], axis=0).reset_index(drop=True)

    # permanova_ait_ibd = pd.concat([permanova_ait_ibd, ait], axis=0).reset_index(
    #     drop=True
    # )

    # bc, ait = PERMANOVA(
    #     dtu_recon_data, dtu_sample_label, dtu_batch_label, dtu_bio_label
    # )

    # bc = pd.DataFrame(
    #     {
    #         "R2": [bc["R2"]],
    #         "p-value": [bc["p-value"]],
    #         "test statistic": [bc["test statistic"]],
    #     }
    # )
    # ait = pd.DataFrame(
    #     {
    #         "R2": [ait["R2"]],
    #         "p-value": [ait["p-value"]],
    #         "test statistic": [ait["test statistic"]],
    #     }
    # )

    # permanova_bc_dtu = pd.concat([permanova_bc_dtu, bc], axis=0).reset_index(drop=True)

    # permanova_ait_dtu = pd.concat([permanova_ait_dtu, ait], axis=0).reset_index(
    #     drop=True
    # )

    # # Performance metrics
    # norm_ad_recon_data = DataTransform(
    #     data=ad_recon_data,
    #     factors=[ad_count_sample_label, ad_count_batch_label, ad_count_bio_label],
    #     count=True,
    # )

    # performance_batch, _ = all_metrics(
    #     data=norm_ad_recon_data,
    #     bio_label=ad_count_bio_label,
    #     batch_label=ad_count_batch_label,
    # )
    # performances_batch_ad.append(performance_batch)

    # norm_ibd_recon_data = DataTransform(
    #     data=ibd_recon_data,
    #     factors=[ibd_sample_label, ibd_batch_label, ibd_bio_label],
    #     count=True,
    # )

    # performance_batch, _ = all_metrics(
    #     data=norm_ibd_recon_data, bio_label=ibd_bio_label, batch_label=ibd_batch_label
    # )
    # performances_batch_ibd.append(performance_batch)

    # norm_dtu_recon_data = DataTransform(
    #     data=dtu_recon_data,
    #     factors=[dtu_sample_label, dtu_batch_label, dtu_bio_label],
    #     count=True,
    # )

    # performance_batch, _ = all_metrics(
    #     data=norm_dtu_recon_data, bio_label=dtu_bio_label, batch_label=dtu_batch_label
    # )
    # performances_batch_dtu.append(performance_batch)

    # # MANN-WHITNEY U TEST - AD COUNT
    # # 1. Compute relative abundances for raw and corrected
    # count_raw = ad_count_data.select_dtypes(include="number")
    # rel_raw = count_raw.div(count_raw.sum(axis=1), axis=0)

    # count_corr = ad_recon_data.select_dtypes(include="number")
    # rel_corr = count_corr.div(count_corr.sum(axis=1), axis=0)

    # # 2. Identify top 10 OTUs by mean relative abundance in the raw data
    # top10 = rel_raw.mean().sort_values(ascending=False).head(10).index

    # # 3. Subset and reshape both to long form, adding a 'Dataset' column
    # df_raw = (
    #     rel_raw[top10]
    #     .reset_index()
    #     .melt(
    #         id_vars="index", value_vars=top10, var_name="OTU", value_name="RelAbundance"
    #     )
    # )
    # df_raw["Dataset"] = "Raw"

    # df_corr = (
    #     rel_corr[top10]
    #     .reset_index()
    #     .melt(
    #         id_vars="index", value_vars=top10, var_name="OTU", value_name="RelAbundance"
    #     )
    # )
    # df_corr["Dataset"] = "Corrected"

    # # 4. Concatenate
    # df_all = pd.concat([df_raw, df_corr], ignore_index=True)
    # df_all.rename(columns={"index": "Sample"}, inplace=True)

    # # 5. Compute Mann-whitney U test for top 10 taxonomic groups
    # stat_results = []

    # for otu in top10:
    #     raw_values = df_all[(df_all["OTU"] == otu) & (df_all["Dataset"] == "Raw")][
    #         "RelAbundance"
    #     ]
    #     corr_values = df_all[
    #         (df_all["OTU"] == otu) & (df_all["Dataset"] == "Corrected")
    #     ]["RelAbundance"]

    #     # Mann-whitney U test
    #     stat, p = mannwhitneyu(
    #         raw_values, corr_values, alternative="two-sided"
    #     )  # Ha two-sided: the distributions aren't equal / there are significant differences
    #     stat_results.append({"OTU": otu, "U-stat": stat, "p-value": p})

    # df_stats = pd.DataFrame(stat_results)
    # df_stats["adj p-value"] = multipletests(df_stats["p-value"], method="fdr_bh")[1]
    # df_stats["iter"] = iter

    # performances_mannwhit_ad = pd.concat(
    #     [performances_mannwhit_ad, df_stats], axis=0
    # ).reset_index(drop=True)

    # # MANN-WHITNEY U TEST - IBD
    # # 1. Compute relative abundances for raw and corrected
    # count_raw = ibd_data.select_dtypes(include="number")
    # rel_raw = count_raw.div(count_raw.sum(axis=1), axis=0)

    # count_corr = ibd_recon_data.select_dtypes(include="number")
    # rel_corr = count_corr.div(count_corr.sum(axis=1), axis=0)

    # # 2. Identify top 10 OTUs by mean relative abundance in the raw data
    # top10 = rel_raw.mean().sort_values(ascending=False).head(10).index

    # # 3. Subset and reshape both to long form, adding a 'Dataset' column
    # df_raw = (
    #     rel_raw[top10]
    #     .reset_index()
    #     .melt(
    #         id_vars="index", value_vars=top10, var_name="OTU", value_name="RelAbundance"
    #     )
    # )
    # df_raw["Dataset"] = "Raw"

    # df_corr = (
    #     rel_corr[top10]
    #     .reset_index()
    #     .melt(
    #         id_vars="index", value_vars=top10, var_name="OTU", value_name="RelAbundance"
    #     )
    # )
    # df_corr["Dataset"] = "Corrected"

    # # 4. Concatenate
    # df_all = pd.concat([df_raw, df_corr], ignore_index=True)
    # df_all.rename(columns={"index": "Sample"}, inplace=True)

    # # 5. Compute Mann-whitney U test for top 10 taxonomic groups
    # stat_results = []

    # for otu in top10:
    #     raw_values = df_all[(df_all["OTU"] == otu) & (df_all["Dataset"] == "Raw")][
    #         "RelAbundance"
    #     ]
    #     corr_values = df_all[
    #         (df_all["OTU"] == otu) & (df_all["Dataset"] == "Corrected")
    #     ]["RelAbundance"]

    #     # Mann-whitney U test
    #     stat, p = mannwhitneyu(
    #         raw_values, corr_values, alternative="two-sided"
    #     )  # Ha two-sided: the distributions aren't equal / there are significant differences
    #     stat_results.append({"OTU": otu, "U-stat": stat, "p-value": p})

    # df_stats = pd.DataFrame(stat_results)
    # df_stats["adj p-value"] = multipletests(df_stats["p-value"], method="fdr_bh")[1]
    # df_stats["iter"] = iter

    # performances_mannwhit_ibd = pd.concat(
    #     [performances_mannwhit_ibd, df_stats], axis=0
    # ).reset_index(drop=True)

    # # MANN-WHITNEY U TEST - DTU-GE
    # # 1. Compute relative abundances for raw and corrected
    # count_raw = dtu_data.select_dtypes(include="number")
    # rel_raw = count_raw.div(count_raw.sum(axis=1), axis=0)

    # count_corr = dtu_recon_data.select_dtypes(include="number")
    # rel_corr = count_corr.div(count_corr.sum(axis=1), axis=0)

    # # 2. Identify top 10 OTUs by mean relative abundance in the raw data
    # top10 = rel_raw.mean().sort_values(ascending=False).head(10).index

    # # 3. Subset and reshape both to long form, adding a 'Dataset' column
    # df_raw = (
    #     rel_raw[top10]
    #     .reset_index()
    #     .melt(
    #         id_vars="index", value_vars=top10, var_name="OTU", value_name="RelAbundance"
    #     )
    # )
    # df_raw["Dataset"] = "Raw"

    # df_corr = (
    #     rel_corr[top10]
    #     .reset_index()
    #     .melt(
    #         id_vars="index", value_vars=top10, var_name="OTU", value_name="RelAbundance"
    #     )
    # )
    # df_corr["Dataset"] = "Corrected"

    # # 4. Concatenate
    # df_all = pd.concat([df_raw, df_corr], ignore_index=True)
    # df_all.rename(columns={"index": "Sample"}, inplace=True)

    # # 5. Compute Mann-whitney U test for top 10 taxonomic groups
    # stat_results = []

    # for otu in top10:
    #     raw_values = df_all[(df_all["OTU"] == otu) & (df_all["Dataset"] == "Raw")][
    #         "RelAbundance"
    #     ]
    #     corr_values = df_all[
    #         (df_all["OTU"] == otu) & (df_all["Dataset"] == "Corrected")
    #     ]["RelAbundance"]

    #     # Mann-whitney U test
    #     stat, p = mannwhitneyu(
    #         raw_values, corr_values, alternative="two-sided"
    #     )  # Ha two-sided: the distributions aren't equal / there are significant differences
    #     stat_results.append({"OTU": otu, "U-stat": stat, "p-value": p})

    # df_stats = pd.DataFrame(stat_results)
    # df_stats["adj p-value"] = multipletests(df_stats["p-value"], method="fdr_bh")[1]
    # df_stats["iter"] = iter

    # performances_mannwhit_dtu = pd.concat(
    #     [performances_mannwhit_dtu, df_stats], axis=0
    # ).reset_index(drop=True)

    # Save performance to file
    # pd.DataFrame(performances_batch_ad).to_csv(
    #     "performance_metrics/deterministic/AD_count/VMM_KL_cycle_batch.csv", index=False
    # )
    # pd.DataFrame(performances_clisi_ad).to_csv(
    #     "performance_metrics/deterministic/AD_count/VMM_KL_cycle_clisi.csv", index=False
    # )
    # pd.DataFrame(performances_ilisi_ad).to_csv(
    #     "performance_metrics/deterministic/AD_count/VMM_KL_cycle_ilisi.csv", index=False
    # )
    # pd.DataFrame(permanova_ait_ad).to_csv(
    #     "performance_metrics/deterministic/AD_count/VMM_KL_cycle_perma_ait.csv", index=False
    # )
    # pd.DataFrame(permanova_bc_ad).to_csv(
    #     "performance_metrics/deterministic/AD_count/VMM_KL_cycle_perma_bc.csv", index=False
    # )
    # pd.DataFrame(performances_mannwhit_ad).to_csv(
    #     "performance_metrics/deterministic/AD_count/VMM_KL_cycle_mannwhit.csv", index=False
    # )

    # pd.DataFrame(performances_batch_ibd).to_csv(
    #     "performance_metrics/deterministic/IBD/VMM_KL_cycle_batch.csv", index=False
    # )
    # pd.DataFrame(performances_clisi_ibd).to_csv(
    #     "performance_metrics/deterministic/IBD/VMM_KL_cycle_clisi.csv", index=False
    # )
    # pd.DataFrame(performances_ilisi_ibd).to_csv(
    #     "performance_metrics/deterministic/IBD/VMM_KL_cycle_ilisi.csv", index=False
    # )
    # pd.DataFrame(permanova_ait_ibd).to_csv(
    #     "performance_metrics/deterministic/IBD/VMM_KL_cycle_perma_ait.csv", index=False
    # )
    # pd.DataFrame(permanova_bc_ibd).to_csv(
    #     "performance_metrics/deterministic/IBD/VMM_KL_cycle_perma_bc.csv", index=False
    # )
    # pd.DataFrame(performances_mannwhit_ibd).to_csv(
    #     "performance_metrics/deterministic/IBD/VMM_KL_cycle_mannwhit.csv", index=False
    # )

    # pd.DataFrame(performances_batch_dtu).to_csv(
    #     "performance_metrics/deterministic/DTU-GE/VMM_KL_cycle_batch.csv", index=False
    # )
    # pd.DataFrame(performances_clisi_dtu).to_csv(
    #     "performance_metrics/deterministic/DTU-GE/VMM_KL_cycle_clisi.csv", index=False
    # )
    # pd.DataFrame(performances_ilisi_dtu).to_csv(
    #     "performance_metrics/deterministic/DTU-GE/VMM_KL_cycle_ilisi.csv", index=False
    # )
    # pd.DataFrame(permanova_ait_dtu).to_csv(
    #     "performance_metrics/deterministic/DTU-GE/VMM_KL_cycle_perma_ait.csv", index=False
    # )
    # pd.DataFrame(permanova_bc_dtu).to_csv(
    #     "performance_metrics/deterministic/DTU-GE/VMM_KL_cycle_perma_bc.csv", index=False
    # )
    # pd.DataFrame(performances_mannwhit_dtu).to_csv(
    #     "performance_metrics/deterministic/DTU-GE/VMM_KL_cycle_mannwhit.csv", index=False
    # )

    # Set seed
    if iter == 0:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    performances_batch_ad = []
    performances_ilisi_ad = pd.DataFrame()
    performances_clisi_ad = pd.DataFrame()
    permanova_ait_ad = pd.DataFrame()
    permanova_bc_ad = pd.DataFrame()

    performances_batch_ibd = []
    performances_ilisi_ibd = pd.DataFrame()
    performances_clisi_ibd = pd.DataFrame()
    permanova_ait_ibd = pd.DataFrame()
    permanova_bc_ibd = pd.DataFrame()

    performances_batch_dtu = []
    performances_ilisi_dtu = pd.DataFrame()
    performances_clisi_dtu = pd.DataFrame()
    permanova_ait_dtu = pd.DataFrame()
    permanova_bc_dtu = pd.DataFrame()

    performances_mannwhit_ad = pd.DataFrame()
    performances_mannwhit_ibd = pd.DataFrame()
    performances_mannwhit_dtu = pd.DataFrame()

    # # for performance metrics
    # proc = psutil.Process()
    # # --- pre‐run snapshots ---
    # t0_wall = time.time()
    # t0_cpu = sum(proc.cpu_times()[:2])  # user + sys CPU s
    # mem0 = proc.memory_info().rss  # resident set size
    # torch.cuda.reset_peak_memory_stats()  # clear PyTorch peak

    # # MoG models
    # ad_count_mog_kl_cycle = abaco_run(
    #     dataloader=ad_count_train_dataloader,
    #     n_batches=ad_count_batch_size,
    #     n_bios=ad_count_bio_size,
    #     input_size=ad_count_input_size,
    #     device=device,
    #     prior="MoG",
    #     w_contra=100.0,
    #     kl_cycle=True,
    #     seed=None,
    #     smooth_annealing=True,
    #     pre_epochs=2500,
    #     post_epochs=5000,
    #     vae_post_lr=2e-4,
    #     adv_lr=1e-7,
    #     disc_lr=1e-7,
    # )

    # # --- post‐run snapshots ---
    # t1_wall = time.time()
    # t1_cpu = sum(proc.cpu_times()[:2])
    # mem1 = proc.memory_info().rss
    # peak_gpu = torch.cuda.max_memory_allocated()

    # # --- GPU‐side current status (optional) ---
    # gpus = GPUtil.getGPUs()

    # # --- print a neat summary ---
    # print("===== AD count - MoG KL cycle Performance =====")
    # print(f"Wall‐clock time  : {t1_wall - t0_wall:.1f} s")
    # print(f"CPU time         : {t1_cpu  - t0_cpu :.1f} s")
    # print(f"RAM change       : {(mem1 - mem0)/1024**3:+.2f} GiB")
    # print(f"Peak GPU memory : {peak_gpu/1024**3:.2f} GiB")
    # for g in gpus:
    #     print(
    #         f" GPU {g.id}: util={g.load*100:.1f}%, mem={g.memoryUsed/1024**3:.1f}/{g.memoryTotal/1024**3:.1f} GiB"
    #     )

    if iter == 0:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # # for performance metrics
    # proc = psutil.Process()
    # # --- pre‐run snapshots ---
    # t0_wall = time.time()
    # t0_cpu = sum(proc.cpu_times()[:2])  # user + sys CPU s
    # mem0 = proc.memory_info().rss  # resident set size
    # torch.cuda.reset_peak_memory_stats()  # clear PyTorch peak

    # ibd_mog_kl_cycle = abaco_run(
    #     dataloader=ibd_train_dataloader,
    #     n_batches=ibd_batch_size,
    #     n_bios=ibd_bio_size,
    #     input_size=ibd_input_size,
    #     device=device,
    #     prior="MoG",
    #     w_contra=1000.0,
    #     kl_cycle=True,
    #     w_cycle=1e-3,
    #     seed=None,
    #     smooth_annealing=True,
    #     pre_epochs=5000,
    #     post_epochs=2500,
    #     vae_post_lr=1e-4,
    #     encoder_net=[2048, 1024, 512, 256],
    #     decoder_net=[256, 512, 1024, 2048],
    #     adv_lr=1e-6,
    #     disc_lr=1e-6,
    # )

    # # --- post‐run snapshots ---
    # t1_wall = time.time()
    # t1_cpu = sum(proc.cpu_times()[:2])
    # mem1 = proc.memory_info().rss
    # peak_gpu = torch.cuda.max_memory_allocated()

    # # --- GPU‐side current status (optional) ---
    # gpus = GPUtil.getGPUs()

    # # --- print a neat summary ---
    # print("===== IBD - MoG KL cycle Performance =====")
    # print(f"Wall‐clock time  : {t1_wall - t0_wall:.1f} s")
    # print(f"CPU time         : {t1_cpu  - t0_cpu :.1f} s")
    # print(f"RAM change       : {(mem1 - mem0)/1024**3:+.2f} GiB")
    # print(f"Peak GPU memory : {peak_gpu/1024**3:.2f} GiB")
    # for g in gpus:
    #     print(
    #         f" GPU {g.id}: util={g.load*100:.1f}%, mem={g.memoryUsed/1024**3:.1f}/{g.memoryTotal/1024**3:.1f} GiB"
    #     )

    if iter == 0:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # # for performance metrics
    # proc = psutil.Process()
    # # --- pre‐run snapshots ---
    # t0_wall = time.time()
    # t0_cpu = sum(proc.cpu_times()[:2])  # user + sys CPU s
    # mem0 = proc.memory_info().rss  # resident set size
    # torch.cuda.reset_peak_memory_stats()  # clear PyTorch peak

    # dtu_mog_kl_cycle = abaco_run(
    #     dataloader=dtu_train_dataloader,
    #     n_batches=dtu_batch_size,
    #     n_bios=dtu_bio_size,
    #     input_size=dtu_input_size,
    #     device=device,
    #     prior="MoG",
    #     w_contra=1000.0,
    #     kl_cycle=True,
    #     seed=None,
    #     smooth_annealing=True,
    #     pre_epochs=2500,
    #     post_epochs=5000,
    #     vae_post_lr=2e-4,
    #     adv_lr=1e-5,
    #     disc_lr=1e-5,
    # )

    # # --- post‐run snapshots ---
    # t1_wall = time.time()
    # t1_cpu = sum(proc.cpu_times()[:2])
    # mem1 = proc.memory_info().rss
    # peak_gpu = torch.cuda.max_memory_allocated()

    # # --- GPU‐side current status (optional) ---
    # gpus = GPUtil.getGPUs()

    # # --- print a neat summary ---
    # print("===== DTU-GE - MoG KL cycle Performance =====")
    # print(f"Wall‐clock time  : {t1_wall - t0_wall:.1f} s")
    # print(f"CPU time         : {t1_cpu  - t0_cpu :.1f} s")
    # print(f"RAM change       : {(mem1 - mem0)/1024**3:+.2f} GiB")
    # print(f"Peak GPU memory : {peak_gpu/1024**3:.2f} GiB")
    # for g in gpus:
    #     print(
    #         f" GPU {g.id}: util={g.load*100:.1f}%, mem={g.memoryUsed/1024**3:.1f}/{g.memoryTotal/1024**3:.1f} GiB"
    #     )

    if iter == 0:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Reconstruct data
    # ad_recon_data = abaco_recon(
    #     model=ad_count_mog_kl_cycle,
    #     device=device,
    #     data=ad_count_data,
    #     dataloader=ad_count_train_dataloader,
    #     sample_label=ad_count_sample_label,
    #     batch_label=ad_count_batch_label,
    #     bio_label=ad_count_bio_label,
    #     seed=None,
    #     monte_carlo=1,
    # )

    if iter == 0:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ibd_recon_data = abaco_recon(
    #     model=ibd_mog_kl_cycle,
    #     device=device,
    #     data=ibd_data,
    #     dataloader=ibd_train_dataloader,
    #     sample_label=ibd_sample_label,
    #     batch_label=ibd_batch_label,
    #     bio_label=ibd_bio_label,
    #     seed=None,
    #     monte_carlo=1,
    # )

    if iter == 0:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # dtu_recon_data = abaco_recon(
    #     model=dtu_mog_kl_cycle,
    #     device=device,
    #     data=dtu_data,
    #     dataloader=dtu_train_dataloader,
    #     sample_label=dtu_sample_label,
    #     batch_label=dtu_batch_label,
    #     bio_label=dtu_bio_label,
    #     seed=None,
    #     monte_carlo=1,
    # )

    # ad_recon_data.to_csv(
    #     f"performance_metrics/multi_runs/AD_count/mog_kl_recon_{iter}", index=False
    # )
    # ibd_recon_data.to_csv(
    #     f"performance_metrics/multi_runs/IBD/mog_kl_recon_contra_1000_2_{iter}",
    #     index=False,
    # )
    # dtu_recon_data.to_csv(
    #     f"performance_metrics/multi_runs/DTU-GE/mog_kl_recon_contra_1000_{iter}",
    #     index=False,
    # )

    # # LISI tables
    # clisi_ad = cLISI_full_rank(ad_recon_data, ad_count_bio_label)
    # ilisi_ad = iLISI_full_rank(ad_recon_data, ad_count_batch_label)

    # clisi_ad["iter"] = iter
    # ilisi_ad["iter"] = iter

    # performances_clisi_ad = pd.concat(
    #     [performances_clisi_ad, clisi_ad], axis=0
    # ).reset_index(drop=True)
    # performances_ilisi_ad = pd.concat(
    #     [performances_ilisi_ad, ilisi_ad], axis=0
    # ).reset_index(drop=True)

    # clisi_ibd = cLISI_full_rank(ibd_recon_data, ibd_bio_label)
    # ilisi_ibd = iLISI_full_rank(ibd_recon_data, ibd_batch_label)

    # clisi_ibd["iter"] = iter
    # ilisi_ibd["iter"] = iter

    # performances_clisi_ibd = pd.concat(
    #     [performances_clisi_ibd, clisi_ibd], axis=0
    # ).reset_index(drop=True)
    # performances_ilisi_ibd = pd.concat(
    #     [performances_ilisi_ibd, ilisi_ibd], axis=0
    # ).reset_index(drop=True)

    # clisi_dtu = cLISI_full_rank(dtu_recon_data, dtu_bio_label)
    # ilisi_dtu = iLISI_full_rank(dtu_recon_data, dtu_batch_label)

    # clisi_dtu["iter"] = iter
    # ilisi_dtu["iter"] = iter

    # performances_clisi_dtu = pd.concat(
    #     [performances_clisi_dtu, clisi_dtu], axis=0
    # ).reset_index(drop=True)
    # performances_ilisi_dtu = pd.concat(
    #     [performances_ilisi_dtu, ilisi_dtu], axis=0
    # ).reset_index(drop=True)

    # # PERMANOVA
    # bc, ait = PERMANOVA(
    #     ad_recon_data, ad_count_sample_label, ad_count_batch_label, ad_count_bio_label
    # )

    # bc = pd.DataFrame(
    #     {
    #         "R2": [bc["R2"]],
    #         "p-value": [bc["p-value"]],
    #         "test statistic": [bc["test statistic"]],
    #     }
    # )
    # ait = pd.DataFrame(
    #     {
    #         "R2": [ait["R2"]],
    #         "p-value": [ait["p-value"]],
    #         "test statistic": [ait["test statistic"]],
    #     }
    # )

    # permanova_bc_ad = pd.concat([permanova_bc_ad, bc], axis=0).reset_index(drop=True)

    # permanova_ait_ad = pd.concat([permanova_ait_ad, ait], axis=0).reset_index(drop=True)

    # bc, ait = PERMANOVA(
    #     ibd_recon_data, ibd_sample_label, ibd_batch_label, ibd_bio_label
    # )

    # bc = pd.DataFrame(
    #     {
    #         "R2": [bc["R2"]],
    #         "p-value": [bc["p-value"]],
    #         "test statistic": [bc["test statistic"]],
    #     }
    # )
    # ait = pd.DataFrame(
    #     {
    #         "R2": [ait["R2"]],
    #         "p-value": [ait["p-value"]],
    #         "test statistic": [ait["test statistic"]],
    #     }
    # )

    # permanova_bc_ibd = pd.concat([permanova_bc_ibd, bc], axis=0).reset_index(drop=True)

    # permanova_ait_ibd = pd.concat([permanova_ait_ibd, ait], axis=0).reset_index(
    #     drop=True
    # )

    # bc, ait = PERMANOVA(
    #     dtu_recon_data, dtu_sample_label, dtu_batch_label, dtu_bio_label
    # )

    # bc = pd.DataFrame(
    #     {
    #         "R2": [bc["R2"]],
    #         "p-value": [bc["p-value"]],
    #         "test statistic": [bc["test statistic"]],
    #     }
    # )
    # ait = pd.DataFrame(
    #     {
    #         "R2": [ait["R2"]],
    #         "p-value": [ait["p-value"]],
    #         "test statistic": [ait["test statistic"]],
    #     }
    # )

    # permanova_bc_dtu = pd.concat([permanova_bc_dtu, bc], axis=0).reset_index(drop=True)

    # permanova_ait_dtu = pd.concat([permanova_ait_dtu, ait], axis=0).reset_index(
    #     drop=True
    # )

    # # Performance metrics
    # norm_ad_recon_data = DataTransform(
    #     data=ad_recon_data,
    #     factors=[ad_count_sample_label, ad_count_batch_label, ad_count_bio_label],
    #     count=True,
    # )

    # performance_batch, _ = all_metrics(
    #     data=norm_ad_recon_data,
    #     bio_label=ad_count_bio_label,
    #     batch_label=ad_count_batch_label,
    # )
    # performances_batch_ad.append(performance_batch)

    # norm_ibd_recon_data = DataTransform(
    #     data=ibd_recon_data,
    #     factors=[ibd_sample_label, ibd_batch_label, ibd_bio_label],
    #     count=True,
    # )

    # performance_batch, _ = all_metrics(
    #     data=norm_ibd_recon_data, bio_label=ibd_bio_label, batch_label=ibd_batch_label
    # )
    # performances_batch_ibd.append(performance_batch)

    # norm_dtu_recon_data = DataTransform(
    #     data=dtu_recon_data,
    #     factors=[dtu_sample_label, dtu_batch_label, dtu_bio_label],
    #     count=True,
    # )

    # performance_batch, _ = all_metrics(
    #     data=norm_dtu_recon_data, bio_label=dtu_bio_label, batch_label=dtu_batch_label
    # )
    # performances_batch_dtu.append(performance_batch)

    # # MANN-WHITNEY U TEST - AD COUNT
    # # 1. Compute relative abundances for raw and corrected
    # count_raw = ad_count_data.select_dtypes(include="number")
    # rel_raw = count_raw.div(count_raw.sum(axis=1), axis=0)

    # count_corr = ad_recon_data.select_dtypes(include="number")
    # rel_corr = count_corr.div(count_corr.sum(axis=1), axis=0)

    # # 2. Identify top 10 OTUs by mean relative abundance in the raw data
    # top10 = rel_raw.mean().sort_values(ascending=False).head(10).index

    # # 3. Subset and reshape both to long form, adding a 'Dataset' column
    # df_raw = (
    #     rel_raw[top10]
    #     .reset_index()
    #     .melt(
    #         id_vars="index", value_vars=top10, var_name="OTU", value_name="RelAbundance"
    #     )
    # )
    # df_raw["Dataset"] = "Raw"

    # df_corr = (
    #     rel_corr[top10]
    #     .reset_index()
    #     .melt(
    #         id_vars="index", value_vars=top10, var_name="OTU", value_name="RelAbundance"
    #     )
    # )
    # df_corr["Dataset"] = "Corrected"

    # # 4. Concatenate
    # df_all = pd.concat([df_raw, df_corr], ignore_index=True)
    # df_all.rename(columns={"index": "Sample"}, inplace=True)

    # # 5. Compute Mann-whitney U test for top 10 taxonomic groups
    # stat_results = []

    # for otu in top10:
    #     raw_values = df_all[(df_all["OTU"] == otu) & (df_all["Dataset"] == "Raw")][
    #         "RelAbundance"
    #     ]
    #     corr_values = df_all[
    #         (df_all["OTU"] == otu) & (df_all["Dataset"] == "Corrected")
    #     ]["RelAbundance"]

    #     # Mann-whitney U test
    #     stat, p = mannwhitneyu(
    #         raw_values, corr_values, alternative="two-sided"
    #     )  # Ha two-sided: the distributions aren't equal / there are significant differences
    #     stat_results.append({"OTU": otu, "U-stat": stat, "p-value": p})

    # df_stats = pd.DataFrame(stat_results)
    # df_stats["adj p-value"] = multipletests(df_stats["p-value"], method="fdr_bh")[1]
    # df_stats["iter"] = iter

    # performances_mannwhit_ad = pd.concat(
    #     [performances_mannwhit_ad, df_stats], axis=0
    # ).reset_index(drop=True)

    # # MANN-WHITNEY U TEST - IBD
    # # 1. Compute relative abundances for raw and corrected
    # count_raw = ibd_data.select_dtypes(include="number")
    # rel_raw = count_raw.div(count_raw.sum(axis=1), axis=0)

    # count_corr = ibd_recon_data.select_dtypes(include="number")
    # rel_corr = count_corr.div(count_corr.sum(axis=1), axis=0)

    # # 2. Identify top 10 OTUs by mean relative abundance in the raw data
    # top10 = rel_raw.mean().sort_values(ascending=False).head(10).index

    # # 3. Subset and reshape both to long form, adding a 'Dataset' column
    # df_raw = (
    #     rel_raw[top10]
    #     .reset_index()
    #     .melt(
    #         id_vars="index", value_vars=top10, var_name="OTU", value_name="RelAbundance"
    #     )
    # )
    # df_raw["Dataset"] = "Raw"

    # df_corr = (
    #     rel_corr[top10]
    #     .reset_index()
    #     .melt(
    #         id_vars="index", value_vars=top10, var_name="OTU", value_name="RelAbundance"
    #     )
    # )
    # df_corr["Dataset"] = "Corrected"

    # # 4. Concatenate
    # df_all = pd.concat([df_raw, df_corr], ignore_index=True)
    # df_all.rename(columns={"index": "Sample"}, inplace=True)

    # # 5. Compute Mann-whitney U test for top 10 taxonomic groups
    # stat_results = []

    # for otu in top10:
    #     raw_values = df_all[(df_all["OTU"] == otu) & (df_all["Dataset"] == "Raw")][
    #         "RelAbundance"
    #     ]
    #     corr_values = df_all[
    #         (df_all["OTU"] == otu) & (df_all["Dataset"] == "Corrected")
    #     ]["RelAbundance"]

    #     # Mann-whitney U test
    #     stat, p = mannwhitneyu(
    #         raw_values, corr_values, alternative="two-sided"
    #     )  # Ha two-sided: the distributions aren't equal / there are significant differences
    #     stat_results.append({"OTU": otu, "U-stat": stat, "p-value": p})

    # df_stats = pd.DataFrame(stat_results)
    # df_stats["adj p-value"] = multipletests(df_stats["p-value"], method="fdr_bh")[1]
    # df_stats["iter"] = iter

    # performances_mannwhit_ibd = pd.concat(
    #     [performances_mannwhit_ibd, df_stats], axis=0
    # ).reset_index(drop=True)

    # # MANN-WHITNEY U TEST - DTU-GE
    # # 1. Compute relative abundances for raw and corrected
    # count_raw = dtu_data.select_dtypes(include="number")
    # rel_raw = count_raw.div(count_raw.sum(axis=1), axis=0)

    # count_corr = dtu_recon_data.select_dtypes(include="number")
    # rel_corr = count_corr.div(count_corr.sum(axis=1), axis=0)

    # # 2. Identify top 10 OTUs by mean relative abundance in the raw data
    # top10 = rel_raw.mean().sort_values(ascending=False).head(10).index

    # # 3. Subset and reshape both to long form, adding a 'Dataset' column
    # df_raw = (
    #     rel_raw[top10]
    #     .reset_index()
    #     .melt(
    #         id_vars="index", value_vars=top10, var_name="OTU", value_name="RelAbundance"
    #     )
    # )
    # df_raw["Dataset"] = "Raw"

    # df_corr = (
    #     rel_corr[top10]
    #     .reset_index()
    #     .melt(
    #         id_vars="index", value_vars=top10, var_name="OTU", value_name="RelAbundance"
    #     )
    # )
    # df_corr["Dataset"] = "Corrected"

    # # 4. Concatenate
    # df_all = pd.concat([df_raw, df_corr], ignore_index=True)
    # df_all.rename(columns={"index": "Sample"}, inplace=True)

    # # 5. Compute Mann-whitney U test for top 10 taxonomic groups
    # stat_results = []

    # for otu in top10:
    #     raw_values = df_all[(df_all["OTU"] == otu) & (df_all["Dataset"] == "Raw")][
    #         "RelAbundance"
    #     ]
    #     corr_values = df_all[
    #         (df_all["OTU"] == otu) & (df_all["Dataset"] == "Corrected")
    #     ]["RelAbundance"]

    #     # Mann-whitney U test
    #     stat, p = mannwhitneyu(
    #         raw_values, corr_values, alternative="two-sided"
    #     )  # Ha two-sided: the distributions aren't equal / there are significant differences
    #     stat_results.append({"OTU": otu, "U-stat": stat, "p-value": p})

    # df_stats = pd.DataFrame(stat_results)
    # df_stats["adj p-value"] = multipletests(df_stats["p-value"], method="fdr_bh")[1]
    # df_stats["iter"] = iter

    # performances_mannwhit_dtu = pd.concat(
    #     [performances_mannwhit_dtu, df_stats], axis=0
    # ).reset_index(drop=True)

    # Save performance to file
    # pd.DataFrame(performances_batch_ad).to_csv(
    #     "performance_metrics/deterministic/AD_count/MoG_KL_cycle_batch.csv", index=False
    # )
    # pd.DataFrame(performances_clisi_ad).to_csv(
    #     "performance_metrics/deterministic/AD_count/MoG_KL_cycle_clisi.csv", index=False
    # )
    # pd.DataFrame(performances_ilisi_ad).to_csv(
    #     "performance_metrics/deterministic/AD_count/MoG_KL_cycle_ilisi.csv", index=False
    # )
    # pd.DataFrame(permanova_ait_ad).to_csv(
    #     "performance_metrics/deterministic/AD_count/MoG_KL_cycle_perma_ait.csv", index=False
    # )
    # pd.DataFrame(permanova_bc_ad).to_csv(
    #     "performance_metrics/deterministic/AD_count/MoG_KL_cycle_perma_bc.csv", index=False
    # )
    # pd.DataFrame(performances_mannwhit_ad).to_csv(
    #     "performance_metrics/deterministic/AD_count/MoG_KL_cycle_mannwhit.csv", index=False
    # )

    # pd.DataFrame(performances_batch_ibd).to_csv(
    #     "performance_metrics/deterministic/IBD/MoG_KL_cycle_batch.csv", index=False
    # )
    # pd.DataFrame(performances_clisi_ibd).to_csv(
    #     "performance_metrics/deterministic/IBD/MoG_KL_cycle_clisi.csv", index=False
    # )
    # pd.DataFrame(performances_ilisi_ibd).to_csv(
    #     "performance_metrics/deterministic/IBD/MoG_KL_cycle_ilisi.csv", index=False
    # )
    # pd.DataFrame(permanova_ait_ibd).to_csv(
    #     "performance_metrics/deterministic/IBD/MoG_KL_cycle_perma_ait.csv", index=False
    # )
    # pd.DataFrame(permanova_bc_ibd).to_csv(
    #     "performance_metrics/deterministic/IBD/MoG_KL_cycle_perma_bc.csv", index=False
    # )
    # pd.DataFrame(performances_mannwhit_ibd).to_csv(
    #     "performance_metrics/deterministic/IBD/MoG_KL_cycle_mannwhit.csv", index=False
    # )

    # pd.DataFrame(performances_batch_dtu).to_csv(
    #     "performance_metrics/deterministic/DTU-GE/MoG_KL_cycle_batch.csv", index=False
    # )
    # pd.DataFrame(performances_clisi_dtu).to_csv(
    #     "performance_metrics/deterministic/DTU-GE/MoG_KL_cycle_clisi.csv", index=False
    # )
    # pd.DataFrame(performances_ilisi_dtu).to_csv(
    #     "performance_metrics/deterministic/DTU-GE/MoG_KL_cycle_ilisi.csv", index=False
    # )
    # pd.DataFrame(permanova_ait_dtu).to_csv(
    #     "performance_metrics/deterministic/DTU-GE/MoG_KL_cycle_perma_ait.csv", index=False
    # )
    # pd.DataFrame(permanova_bc_dtu).to_csv(
    #     "performance_metrics/deterministic/DTU-GE/MoG_KL_cycle_perma_bc.csv", index=False
    # )
    # pd.DataFrame(performances_mannwhit_dtu).to_csv(
    #     "performance_metrics/deterministic/DTU-GE/MoG_KL_cycle_mannwhit.csv", index=False
    # )

    # Set seed
    if iter == 0:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    performances_batch_ad = []
    performances_ilisi_ad = pd.DataFrame()
    performances_clisi_ad = pd.DataFrame()
    permanova_ait_ad = pd.DataFrame()
    permanova_bc_ad = pd.DataFrame()

    performances_batch_ibd = []
    performances_ilisi_ibd = pd.DataFrame()
    performances_clisi_ibd = pd.DataFrame()
    permanova_ait_ibd = pd.DataFrame()
    permanova_bc_ibd = pd.DataFrame()

    performances_batch_dtu = []
    performances_ilisi_dtu = pd.DataFrame()
    performances_clisi_dtu = pd.DataFrame()
    permanova_ait_dtu = pd.DataFrame()
    permanova_bc_dtu = pd.DataFrame()

    performances_mannwhit_ad = pd.DataFrame()
    performances_mannwhit_ibd = pd.DataFrame()
    performances_mannwhit_dtu = pd.DataFrame()

    # # for performance metrics
    # proc = psutil.Process()
    # # --- pre‐run snapshots ---
    # t0_wall = time.time()
    # t0_cpu = sum(proc.cpu_times()[:2])  # user + sys CPU s
    # mem0 = proc.memory_info().rss  # resident set size
    # torch.cuda.reset_peak_memory_stats()  # clear PyTorch peak

    # # Normal models
    # ad_count_std_kl_cycle = abaco_run(
    #     dataloader=ad_count_train_dataloader,
    #     n_batches=ad_count_batch_size,
    #     n_bios=ad_count_bio_size,
    #     input_size=ad_count_input_size,
    #     device=device,
    #     prior="Normal",
    #     w_contra=100.0,
    #     kl_cycle=True,
    #     seed=None,
    #     smooth_annealing=True,
    #     pre_epochs=2500,
    #     post_epochs=5000,
    #     vae_post_lr=2e-4,
    #     adv_lr=1e-7,
    #     disc_lr=1e-7,
    # )

    # # --- post‐run snapshots ---
    # t1_wall = time.time()
    # t1_cpu = sum(proc.cpu_times()[:2])
    # mem1 = proc.memory_info().rss
    # peak_gpu = torch.cuda.max_memory_allocated()

    # # --- GPU‐side current status (optional) ---
    # gpus = GPUtil.getGPUs()

    # # --- print a neat summary ---
    # print("===== AD count - Std KL cycle Performance =====")
    # print(f"Wall‐clock time  : {t1_wall - t0_wall:.1f} s")
    # print(f"CPU time         : {t1_cpu  - t0_cpu :.1f} s")
    # print(f"RAM change       : {(mem1 - mem0)/1024**3:+.2f} GiB")
    # print(f"Peak GPU memory : {peak_gpu/1024**3:.2f} GiB")
    # for g in gpus:
    #     print(
    #         f" GPU {g.id}: util={g.load*100:.1f}%, mem={g.memoryUsed/1024**3:.1f}/{g.memoryTotal/1024**3:.1f} GiB"
    #     )

    if iter == 0:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # # for performance metrics
    # proc = psutil.Process()
    # # --- pre‐run snapshots ---
    # t0_wall = time.time()
    # t0_cpu = sum(proc.cpu_times()[:2])  # user + sys CPU s
    # mem0 = proc.memory_info().rss  # resident set size
    # torch.cuda.reset_peak_memory_stats()  # clear PyTorch peak

    # ibd_std_kl_cycle = abaco_run(
    #     dataloader=ibd_train_dataloader,
    #     n_batches=ibd_batch_size,
    #     n_bios=ibd_bio_size,
    #     input_size=ibd_input_size,
    #     device=device,
    #     prior="Normal",
    #     w_contra=1000.0,
    #     kl_cycle=True,
    #     w_cycle=1e-3,
    #     seed=None,
    #     smooth_annealing=True,
    #     pre_epochs=5000,
    #     post_epochs=2500,
    #     vae_post_lr=1e-4,
    #     encoder_net=[2048, 1024, 512, 256],
    #     decoder_net=[256, 512, 1024, 2048],
    #     adv_lr=1e-6,
    #     disc_lr=1e-6,
    # )

    # # --- post‐run snapshots ---
    # t1_wall = time.time()
    # t1_cpu = sum(proc.cpu_times()[:2])
    # mem1 = proc.memory_info().rss
    # peak_gpu = torch.cuda.max_memory_allocated()

    # # --- GPU‐side current status (optional) ---
    # gpus = GPUtil.getGPUs()

    # # --- print a neat summary ---
    # print("===== IBD - Std KL cycle Performance =====")
    # print(f"Wall‐clock time  : {t1_wall - t0_wall:.1f} s")
    # print(f"CPU time         : {t1_cpu  - t0_cpu :.1f} s")
    # print(f"RAM change       : {(mem1 - mem0)/1024**3:+.2f} GiB")
    # print(f"Peak GPU memory : {peak_gpu/1024**3:.2f} GiB")
    # for g in gpus:
    #     print(
    #         f" GPU {g.id}: util={g.load*100:.1f}%, mem={g.memoryUsed/1024**3:.1f}/{g.memoryTotal/1024**3:.1f} GiB"
    #     )

    if iter == 0:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # # for performance metrics
    # proc = psutil.Process()
    # # --- pre‐run snapshots ---
    # t0_wall = time.time()
    # t0_cpu = sum(proc.cpu_times()[:2])  # user + sys CPU s
    # mem0 = proc.memory_info().rss  # resident set size
    # torch.cuda.reset_peak_memory_stats()  # clear PyTorch peak

    # dtu_std_kl_cycle = abaco_run(
    #     dataloader=dtu_train_dataloader,
    #     n_batches=dtu_batch_size,
    #     n_bios=dtu_bio_size,
    #     input_size=dtu_input_size,
    #     device=device,
    #     prior="Normal",
    #     w_contra=1000.0,
    #     kl_cycle=True,
    #     seed=None,
    #     smooth_annealing=True,
    #     pre_epochs=2500,
    #     post_epochs=5000,
    #     vae_post_lr=2e-4,
    #     adv_lr=1e-5,
    #     disc_lr=1e-5,
    # )

    # # --- post‐run snapshots ---
    # t1_wall = time.time()
    # t1_cpu = sum(proc.cpu_times()[:2])
    # mem1 = proc.memory_info().rss
    # peak_gpu = torch.cuda.max_memory_allocated()

    # # --- GPU‐side current status (optional) ---
    # gpus = GPUtil.getGPUs()

    # # --- print a neat summary ---
    # print("===== DTU-GE - Std KL cycle Performance =====")
    # print(f"Wall‐clock time  : {t1_wall - t0_wall:.1f} s")
    # print(f"CPU time         : {t1_cpu  - t0_cpu :.1f} s")
    # print(f"RAM change       : {(mem1 - mem0)/1024**3:+.2f} GiB")
    # print(f"Peak GPU memory : {peak_gpu/1024**3:.2f} GiB")
    # for g in gpus:
    #     print(
    #         f" GPU {g.id}: util={g.load*100:.1f}%, mem={g.memoryUsed/1024**3:.1f}/{g.memoryTotal/1024**3:.1f} GiB"
    #     )

    if iter == 0:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Reconstruct data
    # ad_recon_data = abaco_recon(
    #     model=ad_count_std_kl_cycle,
    #     device=device,
    #     data=ad_count_data,
    #     dataloader=ad_count_train_dataloader,
    #     sample_label=ad_count_sample_label,
    #     batch_label=ad_count_batch_label,
    #     bio_label=ad_count_bio_label,
    #     seed=None,
    #     monte_carlo=1,
    # )

    if iter == 0:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ibd_recon_data = abaco_recon(
    #     model=ibd_std_kl_cycle,
    #     device=device,
    #     data=ibd_data,
    #     dataloader=ibd_train_dataloader,
    #     sample_label=ibd_sample_label,
    #     batch_label=ibd_batch_label,
    #     bio_label=ibd_bio_label,
    #     seed=None,
    #     monte_carlo=1,
    # )

    if iter == 0:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # dtu_recon_data = abaco_recon(
    #     model=dtu_std_kl_cycle,
    #     device=device,
    #     data=dtu_data,
    #     dataloader=dtu_train_dataloader,
    #     sample_label=dtu_sample_label,
    #     batch_label=dtu_batch_label,
    #     bio_label=dtu_bio_label,
    #     seed=None,
    #     monte_carlo=1,
    # )

    # ad_recon_data.to_csv(
    #     f"performance_metrics/multi_runs/AD_count/std_kl_recon_{iter}", index=False
    # )
    # ibd_recon_data.to_csv(
    #     f"performance_metrics/multi_runs/IBD/std_kl_recon_contra_1000_2_{iter}",
    #     index=False,
    # )
    # dtu_recon_data.to_csv(
    #     f"performance_metrics/multi_runs/DTU-GE/std_kl_recon_contra_1000_{iter}",
    #     index=False,
    # )

    # # LISI tables
    # clisi_ad = cLISI_full_rank(ad_recon_data, ad_count_bio_label)
    # ilisi_ad = iLISI_full_rank(ad_recon_data, ad_count_batch_label)

    # clisi_ad["iter"] = iter
    # ilisi_ad["iter"] = iter

    # performances_clisi_ad = pd.concat(
    #     [performances_clisi_ad, clisi_ad], axis=0
    # ).reset_index(drop=True)
    # performances_ilisi_ad = pd.concat(
    #     [performances_ilisi_ad, ilisi_ad], axis=0
    # ).reset_index(drop=True)

    # clisi_ibd = cLISI_full_rank(ibd_recon_data, ibd_bio_label)
    # ilisi_ibd = iLISI_full_rank(ibd_recon_data, ibd_batch_label)

    # clisi_ibd["iter"] = iter
    # ilisi_ibd["iter"] = iter

    # performances_clisi_ibd = pd.concat(
    #     [performances_clisi_ibd, clisi_ibd], axis=0
    # ).reset_index(drop=True)
    # performances_ilisi_ibd = pd.concat(
    #     [performances_ilisi_ibd, ilisi_ibd], axis=0
    # ).reset_index(drop=True)

    # clisi_dtu = cLISI_full_rank(dtu_recon_data, dtu_bio_label)
    # ilisi_dtu = iLISI_full_rank(dtu_recon_data, dtu_batch_label)

    # clisi_dtu["iter"] = iter
    # ilisi_dtu["iter"] = iter

    # performances_clisi_dtu = pd.concat(
    #     [performances_clisi_dtu, clisi_dtu], axis=0
    # ).reset_index(drop=True)
    # performances_ilisi_dtu = pd.concat(
    #     [performances_ilisi_dtu, ilisi_dtu], axis=0
    # ).reset_index(drop=True)

    # # PERMANOVA
    # bc, ait = PERMANOVA(
    #     ad_recon_data, ad_count_sample_label, ad_count_batch_label, ad_count_bio_label
    # )

    # bc = pd.DataFrame(
    #     {
    #         "R2": [bc["R2"]],
    #         "p-value": [bc["p-value"]],
    #         "test statistic": [bc["test statistic"]],
    #     }
    # )
    # ait = pd.DataFrame(
    #     {
    #         "R2": [ait["R2"]],
    #         "p-value": [ait["p-value"]],
    #         "test statistic": [ait["test statistic"]],
    #     }
    # )

    # permanova_bc_ad = pd.concat([permanova_bc_ad, bc], axis=0).reset_index(drop=True)

    # permanova_ait_ad = pd.concat([permanova_ait_ad, ait], axis=0).reset_index(drop=True)

    # bc, ait = PERMANOVA(
    #     ibd_recon_data, ibd_sample_label, ibd_batch_label, ibd_bio_label
    # )

    # bc = pd.DataFrame(
    #     {
    #         "R2": [bc["R2"]],
    #         "p-value": [bc["p-value"]],
    #         "test statistic": [bc["test statistic"]],
    #     }
    # )
    # ait = pd.DataFrame(
    #     {
    #         "R2": [ait["R2"]],
    #         "p-value": [ait["p-value"]],
    #         "test statistic": [ait["test statistic"]],
    #     }
    # )

    # permanova_bc_ibd = pd.concat([permanova_bc_ibd, bc], axis=0).reset_index(drop=True)

    # permanova_ait_ibd = pd.concat([permanova_ait_ibd, ait], axis=0).reset_index(
    #     drop=True
    # )

    # bc, ait = PERMANOVA(
    #     dtu_recon_data, dtu_sample_label, dtu_batch_label, dtu_bio_label
    # )

    # bc = pd.DataFrame(
    #     {
    #         "R2": [bc["R2"]],
    #         "p-value": [bc["p-value"]],
    #         "test statistic": [bc["test statistic"]],
    #     }
    # )
    # ait = pd.DataFrame(
    #     {
    #         "R2": [ait["R2"]],
    #         "p-value": [ait["p-value"]],
    #         "test statistic": [ait["test statistic"]],
    #     }
    # )

    # permanova_bc_dtu = pd.concat([permanova_bc_dtu, bc], axis=0).reset_index(drop=True)

    # permanova_ait_dtu = pd.concat([permanova_ait_dtu, ait], axis=0).reset_index(
    #     drop=True
    # )

    # # Performance metrics
    # norm_ad_recon_data = DataTransform(
    #     data=ad_recon_data,
    #     factors=[ad_count_sample_label, ad_count_batch_label, ad_count_bio_label],
    #     count=True,
    # )

    # performance_batch, _ = all_metrics(
    #     data=norm_ad_recon_data,
    #     bio_label=ad_count_bio_label,
    #     batch_label=ad_count_batch_label,
    # )
    # performances_batch_ad.append(performance_batch)

    # norm_ibd_recon_data = DataTransform(
    #     data=ibd_recon_data,
    #     factors=[ibd_sample_label, ibd_batch_label, ibd_bio_label],
    #     count=True,
    # )

    # performance_batch, _ = all_metrics(
    #     data=norm_ibd_recon_data, bio_label=ibd_bio_label, batch_label=ibd_batch_label
    # )
    # performances_batch_ibd.append(performance_batch)

    # norm_dtu_recon_data = DataTransform(
    #     data=dtu_recon_data,
    #     factors=[dtu_sample_label, dtu_batch_label, dtu_bio_label],
    #     count=True,
    # )

    # performance_batch, _ = all_metrics(
    #     data=norm_dtu_recon_data, bio_label=dtu_bio_label, batch_label=dtu_batch_label
    # )
    # performances_batch_dtu.append(performance_batch)

    # MANN-WHITNEY U TEST - AD COUNT
    # 1. Compute relative abundances for raw and corrected
    # count_raw = ad_count_data.select_dtypes(include="number")
    # rel_raw = count_raw.div(count_raw.sum(axis=1), axis=0)

    # count_corr = ad_recon_data.select_dtypes(include="number")
    # rel_corr = count_corr.div(count_corr.sum(axis=1), axis=0)

    # # 2. Identify top 10 OTUs by mean relative abundance in the raw data
    # top10 = rel_raw.mean().sort_values(ascending=False).head(10).index

    # # 3. Subset and reshape both to long form, adding a 'Dataset' column
    # df_raw = (
    #     rel_raw[top10]
    #     .reset_index()
    #     .melt(
    #         id_vars="index", value_vars=top10, var_name="OTU", value_name="RelAbundance"
    #     )
    # )
    # df_raw["Dataset"] = "Raw"

    # df_corr = (
    #     rel_corr[top10]
    #     .reset_index()
    #     .melt(
    #         id_vars="index", value_vars=top10, var_name="OTU", value_name="RelAbundance"
    #     )
    # )
    # df_corr["Dataset"] = "Corrected"

    # # 4. Concatenate
    # df_all = pd.concat([df_raw, df_corr], ignore_index=True)
    # df_all.rename(columns={"index": "Sample"}, inplace=True)

    # # 5. Compute Mann-whitney U test for top 10 taxonomic groups
    # stat_results = []

    # for otu in top10:
    #     raw_values = df_all[(df_all["OTU"] == otu) & (df_all["Dataset"] == "Raw")][
    #         "RelAbundance"
    #     ]
    #     corr_values = df_all[
    #         (df_all["OTU"] == otu) & (df_all["Dataset"] == "Corrected")
    #     ]["RelAbundance"]

    #     # Mann-whitney U test
    #     stat, p = mannwhitneyu(
    #         raw_values, corr_values, alternative="two-sided"
    #     )  # Ha two-sided: the distributions aren't equal / there are significant differences
    #     stat_results.append({"OTU": otu, "U-stat": stat, "p-value": p})

    # df_stats = pd.DataFrame(stat_results)
    # df_stats["adj p-value"] = multipletests(df_stats["p-value"], method="fdr_bh")[1]
    # df_stats["iter"] = iter

    # performances_mannwhit_ad = pd.concat(
    #     [performances_mannwhit_ad, df_stats], axis=0
    # ).reset_index(drop=True)

    # # MANN-WHITNEY U TEST - IBD
    # # 1. Compute relative abundances for raw and corrected
    # count_raw = ibd_data.select_dtypes(include="number")
    # rel_raw = count_raw.div(count_raw.sum(axis=1), axis=0)

    # count_corr = ibd_recon_data.select_dtypes(include="number")
    # rel_corr = count_corr.div(count_corr.sum(axis=1), axis=0)

    # # 2. Identify top 10 OTUs by mean relative abundance in the raw data
    # top10 = rel_raw.mean().sort_values(ascending=False).head(10).index

    # # 3. Subset and reshape both to long form, adding a 'Dataset' column
    # df_raw = (
    #     rel_raw[top10]
    #     .reset_index()
    #     .melt(
    #         id_vars="index", value_vars=top10, var_name="OTU", value_name="RelAbundance"
    #     )
    # )
    # df_raw["Dataset"] = "Raw"

    # df_corr = (
    #     rel_corr[top10]
    #     .reset_index()
    #     .melt(
    #         id_vars="index", value_vars=top10, var_name="OTU", value_name="RelAbundance"
    #     )
    # )
    # df_corr["Dataset"] = "Corrected"

    # # 4. Concatenate
    # df_all = pd.concat([df_raw, df_corr], ignore_index=True)
    # df_all.rename(columns={"index": "Sample"}, inplace=True)

    # # 5. Compute Mann-whitney U test for top 10 taxonomic groups
    # stat_results = []

    # for otu in top10:
    #     raw_values = df_all[(df_all["OTU"] == otu) & (df_all["Dataset"] == "Raw")][
    #         "RelAbundance"
    #     ]
    #     corr_values = df_all[
    #         (df_all["OTU"] == otu) & (df_all["Dataset"] == "Corrected")
    #     ]["RelAbundance"]

    #     # Mann-whitney U test
    #     stat, p = mannwhitneyu(
    #         raw_values, corr_values, alternative="two-sided"
    #     )  # Ha two-sided: the distributions aren't equal / there are significant differences
    #     stat_results.append({"OTU": otu, "U-stat": stat, "p-value": p})

    # df_stats = pd.DataFrame(stat_results)
    # df_stats["adj p-value"] = multipletests(df_stats["p-value"], method="fdr_bh")[1]
    # df_stats["iter"] = iter

    # performances_mannwhit_ibd = pd.concat(
    #     [performances_mannwhit_ibd, df_stats], axis=0
    # ).reset_index(drop=True)

    # # MANN-WHITNEY U TEST - DTU-GE
    # # 1. Compute relative abundances for raw and corrected
    # count_raw = dtu_data.select_dtypes(include="number")
    # rel_raw = count_raw.div(count_raw.sum(axis=1), axis=0)

    # count_corr = dtu_recon_data.select_dtypes(include="number")
    # rel_corr = count_corr.div(count_corr.sum(axis=1), axis=0)

    # # 2. Identify top 10 OTUs by mean relative abundance in the raw data
    # top10 = rel_raw.mean().sort_values(ascending=False).head(10).index

    # # 3. Subset and reshape both to long form, adding a 'Dataset' column
    # df_raw = (
    #     rel_raw[top10]
    #     .reset_index()
    #     .melt(
    #         id_vars="index", value_vars=top10, var_name="OTU", value_name="RelAbundance"
    #     )
    # )
    # df_raw["Dataset"] = "Raw"

    # df_corr = (
    #     rel_corr[top10]
    #     .reset_index()
    #     .melt(
    #         id_vars="index", value_vars=top10, var_name="OTU", value_name="RelAbundance"
    #     )
    # )
    # df_corr["Dataset"] = "Corrected"

    # # 4. Concatenate
    # df_all = pd.concat([df_raw, df_corr], ignore_index=True)
    # df_all.rename(columns={"index": "Sample"}, inplace=True)

    # # 5. Compute Mann-whitney U test for top 10 taxonomic groups
    # stat_results = []

    # for otu in top10:
    #     raw_values = df_all[(df_all["OTU"] == otu) & (df_all["Dataset"] == "Raw")][
    #         "RelAbundance"
    #     ]
    #     corr_values = df_all[
    #         (df_all["OTU"] == otu) & (df_all["Dataset"] == "Corrected")
    #     ]["RelAbundance"]

    #     # Mann-whitney U test
    #     stat, p = mannwhitneyu(
    #         raw_values, corr_values, alternative="two-sided"
    #     )  # Ha two-sided: the distributions aren't equal / there are significant differences
    #     stat_results.append({"OTU": otu, "U-stat": stat, "p-value": p})

    # df_stats = pd.DataFrame(stat_results)
    # df_stats["adj p-value"] = multipletests(df_stats["p-value"], method="fdr_bh")[1]
    # df_stats["iter"] = iter

    # performances_mannwhit_dtu = pd.concat(
    #     [performances_mannwhit_dtu, df_stats], axis=0
    # ).reset_index(drop=True)

    # Save performance to file
    # Save performance to file
    # pd.DataFrame(performances_batch_ad).to_csv(
    #     "performance_metrics/deterministic/AD_count/Std_KL_cycle_batch.csv", index=False
    # )
    # pd.DataFrame(performances_clisi_ad).to_csv(
    #     "performance_metrics/deterministic/AD_count/Std_KL_cycle_clisi.csv", index=False
    # )
    # pd.DataFrame(performances_ilisi_ad).to_csv(
    #     "performance_metrics/deterministic/AD_count/Std_KL_cycle_ilisi.csv", index=False
    # )
    # pd.DataFrame(permanova_ait_ad).to_csv(
    #     "performance_metrics/deterministic/AD_count/Std_KL_cycle_perma_ait.csv", index=False
    # )
    # pd.DataFrame(permanova_bc_ad).to_csv(
    #     "performance_metrics/deterministic/AD_count/Std_KL_cycle_perma_bc.csv", index=False
    # )
    # pd.DataFrame(performances_mannwhit_ad).to_csv(
    #     "performance_metrics/deterministic/AD_count/Std_KL_cycle_mannwhit.csv", index=False
    # )

    # pd.DataFrame(performances_batch_ibd).to_csv(
    #     "performance_metrics/deterministic/IBD/Std_KL_cycle_batch.csv", index=False
    # )
    # pd.DataFrame(performances_clisi_ibd).to_csv(
    #     "performance_metrics/deterministic/IBD/Std_KL_cycle_clisi.csv", index=False
    # )
    # pd.DataFrame(performances_ilisi_ibd).to_csv(
    #     "performance_metrics/deterministic/IBD/Std_KL_cycle_ilisi.csv", index=False
    # )
    # pd.DataFrame(permanova_ait_ibd).to_csv(
    #     "performance_metrics/deterministic/IBD/Std_KL_cycle_perma_ait.csv", index=False
    # )
    # pd.DataFrame(permanova_bc_ibd).to_csv(
    #     "performance_metrics/deterministic/IBD/Std_KL_cycle_perma_bc.csv", index=False
    # )
    # pd.DataFrame(performances_mannwhit_ibd).to_csv(
    #     "performance_metrics/deterministic/IBD/Std_KL_cycle_mannwhit.csv", index=False
    # )

    # pd.DataFrame(performances_batch_dtu).to_csv(
    #     "performance_metrics/deterministic/DTU-GE/Std_KL_cycle_batch.csv", index=False
    # )
    # pd.DataFrame(performances_clisi_dtu).to_csv(
    #     "performance_metrics/deterministic/DTU-GE/Std_KL_cycle_clisi.csv", index=False
    # )
    # pd.DataFrame(performances_ilisi_dtu).to_csv(
    #     "performance_metrics/deterministic/DTU-GE/Std_KL_cycle_ilisi.csv", index=False
    # )
    # pd.DataFrame(permanova_ait_dtu).to_csv(
    #     "performance_metrics/deterministic/DTU-GE/Std_KL_cycle_perma_ait.csv", index=False
    # )
    # pd.DataFrame(permanova_bc_dtu).to_csv(
    #     "performance_metrics/deterministic/DTU-GE/Std_KL_cycle_perma_bc.csv", index=False
    # )
    # pd.DataFrame(performances_mannwhit_dtu).to_csv(
    #     "performance_metrics/deterministic/DTU-GE/Std_KL_cycle_mannwhit.csv", index=False
    # )

    # Set seed
    if iter == 0:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    performances_batch_ad = []
    performances_ilisi_ad = pd.DataFrame()
    performances_clisi_ad = pd.DataFrame()
    permanova_ait_ad = pd.DataFrame()
    permanova_bc_ad = pd.DataFrame()

    performances_batch_ibd = []
    performances_ilisi_ibd = pd.DataFrame()
    performances_clisi_ibd = pd.DataFrame()
    permanova_ait_ibd = pd.DataFrame()
    permanova_bc_ibd = pd.DataFrame()

    performances_batch_dtu = []
    performances_ilisi_dtu = pd.DataFrame()
    performances_clisi_dtu = pd.DataFrame()
    permanova_ait_dtu = pd.DataFrame()
    permanova_bc_dtu = pd.DataFrame()

    performances_mannwhit_ad = pd.DataFrame()
    performances_mannwhit_ibd = pd.DataFrame()
    performances_mannwhit_dtu = pd.DataFrame()

    # # for performance metrics
    # proc = psutil.Process()
    # # --- pre‐run snapshots ---
    # t0_wall = time.time()
    # t0_cpu = sum(proc.cpu_times()[:2])  # user + sys CPU s
    # mem0 = proc.memory_info().rss  # resident set size
    # torch.cuda.reset_peak_memory_stats()  # clear PyTorch peak

    # # VMM models
    # ad_count_vmm_no_cycle = abaco_run(
    #     dataloader=ad_count_train_dataloader,
    #     n_batches=ad_count_batch_size,
    #     n_bios=ad_count_bio_size,
    #     input_size=ad_count_input_size,
    #     device=device,
    #     w_contra=100.0,
    #     kl_cycle=False,
    #     seed=None,
    #     smooth_annealing=True,
    #     pre_epochs=2500,
    #     post_epochs=5000,
    #     vae_post_lr=2e-4,
    #     adv_lr=1e-7,
    #     disc_lr=1e-7,
    # )

    # # --- post‐run snapshots ---
    # t1_wall = time.time()
    # t1_cpu = sum(proc.cpu_times()[:2])
    # mem1 = proc.memory_info().rss
    # peak_gpu = torch.cuda.max_memory_allocated()

    # # --- GPU‐side current status (optional) ---
    # gpus = GPUtil.getGPUs()

    # # --- print a neat summary ---
    # print("===== AD count - VMM no cycle Performance =====")
    # print(f"Wall‐clock time  : {t1_wall - t0_wall:.1f} s")
    # print(f"CPU time         : {t1_cpu  - t0_cpu :.1f} s")
    # print(f"RAM change       : {(mem1 - mem0)/1024**3:+.2f} GiB")
    # print(f"Peak GPU memory : {peak_gpu/1024**3:.2f} GiB")
    # for g in gpus:
    #     print(
    #         f" GPU {g.id}: util={g.load*100:.1f}%, mem={g.memoryUsed/1024**3:.1f}/{g.memoryTotal/1024**3:.1f} GiB"
    #     )

    if iter == 0:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # for performance metrics
    proc = psutil.Process()
    # --- pre‐run snapshots ---
    t0_wall = time.time()
    t0_cpu = sum(proc.cpu_times()[:2])  # user + sys CPU s
    mem0 = proc.memory_info().rss  # resident set size
    torch.cuda.reset_peak_memory_stats()  # clear PyTorch peak

    ibd_vmm_no_cycle = abaco_run(
        dataloader=ibd_train_dataloader,
        n_batches=ibd_batch_size,
        n_bios=ibd_bio_size,
        input_size=ibd_input_size,
        device=device,
        w_contra=1000.0,
        kl_cycle=False,
        seed=None,
        smooth_annealing=True,
        pre_epochs=5000,
        post_epochs=2500,
        vae_pre_lr=1e-4,
        vae_post_lr=1e-4,
        encoder_net=[2048, 1024, 512, 256],
        decoder_net=[256, 512, 1024, 2048],
        adv_lr=1e-6,
        disc_lr=1e-6,
        new_pre_train=True,
    )

    # --- post‐run snapshots ---
    t1_wall = time.time()
    t1_cpu = sum(proc.cpu_times()[:2])
    mem1 = proc.memory_info().rss
    peak_gpu = torch.cuda.max_memory_allocated()

    # --- GPU‐side current status (optional) ---
    gpus = GPUtil.getGPUs()

    # --- print a neat summary ---
    print("===== IBD - VMM no cycle Performance =====")
    print(f"Wall‐clock time  : {t1_wall - t0_wall:.1f} s")
    print(f"CPU time         : {t1_cpu  - t0_cpu :.1f} s")
    print(f"RAM change       : {(mem1 - mem0)/1024**3:+.2f} GiB")
    print(f"Peak GPU memory : {peak_gpu/1024**3:.2f} GiB")
    for g in gpus:
        print(
            f" GPU {g.id}: util={g.load*100:.1f}%, mem={g.memoryUsed/1024**3:.1f}/{g.memoryTotal/1024**3:.1f} GiB"
        )

    if iter == 0:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # # for performance metrics
    # proc = psutil.Process()
    # # --- pre‐run snapshots ---
    # t0_wall = time.time()
    # t0_cpu = sum(proc.cpu_times()[:2])  # user + sys CPU s
    # mem0 = proc.memory_info().rss  # resident set size
    # torch.cuda.reset_peak_memory_stats()  # clear PyTorch peak

    # dtu_vmm_no_cycle = abaco_run(
    #     dataloader=dtu_train_dataloader,
    #     n_batches=dtu_batch_size,
    #     n_bios=dtu_bio_size,
    #     input_size=dtu_input_size,
    #     device=device,
    #     w_contra=1000.0,
    #     kl_cycle=False,
    #     seed=None,
    #     smooth_annealing=True,
    #     pre_epochs=5000,
    #     post_epochs=2500,
    #     vae_pre_lr=1e-4,
    #     vae_post_lr=2e-4,
    #     adv_lr=1e-6,
    #     disc_lr=1e-6,
    # )

    # # --- post‐run snapshots ---
    # t1_wall = time.time()
    # t1_cpu = sum(proc.cpu_times()[:2])
    # mem1 = proc.memory_info().rss
    # peak_gpu = torch.cuda.max_memory_allocated()

    # # --- GPU‐side current status (optional) ---
    # gpus = GPUtil.getGPUs()

    # # --- print a neat summary ---
    # print("===== DTU-GE - VMM no cycle Performance =====")
    # print(f"Wall‐clock time  : {t1_wall - t0_wall:.1f} s")
    # print(f"CPU time         : {t1_cpu  - t0_cpu :.1f} s")
    # print(f"RAM change       : {(mem1 - mem0)/1024**3:+.2f} GiB")
    # print(f"Peak GPU memory : {peak_gpu/1024**3:.2f} GiB")
    # for g in gpus:
    #     print(
    #         f" GPU {g.id}: util={g.load*100:.1f}%, mem={g.memoryUsed/1024**3:.1f}/{g.memoryTotal/1024**3:.1f} GiB"
    #     )

    # if iter == 0:
    #     seed = 42
    #     random.seed(seed)
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)

    #     if torch.cuda.is_available():
    #         torch.cuda.manual_seed_all(seed)

    # Reconstruct data
    # ad_recon_data = abaco_recon(
    #     model=ad_count_vmm_no_cycle,
    #     device=device,
    #     data=ad_count_data,
    #     dataloader=ad_count_train_dataloader,
    #     sample_label=ad_count_sample_label,
    #     batch_label=ad_count_batch_label,
    #     bio_label=ad_count_bio_label,
    #     seed=None,
    #     monte_carlo=1,
    # )

    if iter == 0:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    ibd_recon_data = abaco_recon(
        model=ibd_vmm_no_cycle,
        device=device,
        data=ibd_data,
        dataloader=ibd_train_dataloader,
        sample_label=ibd_sample_label,
        batch_label=ibd_batch_label,
        bio_label=ibd_bio_label,
        seed=None,
        monte_carlo=1,
    )

    if iter == 0:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # dtu_recon_data = abaco_recon(
    #     model=dtu_vmm_no_cycle,
    #     device=device,
    #     data=dtu_data,
    #     dataloader=dtu_train_dataloader,
    #     sample_label=dtu_sample_label,
    #     batch_label=dtu_batch_label,
    #     bio_label=dtu_bio_label,
    #     seed=None,
    #     monte_carlo=1,
    # )

    # ad_recon_data.to_csv(
    #     f"performance_metrics/multi_runs/AD_count/vmm_no_recon_{iter}", index=False
    # )
    ibd_recon_data.to_csv(
        f"performance_metrics/multi_runs/IBD/vmm_no_recon_new_{iter}",
        index=False,
    )
    # dtu_recon_data.to_csv(
    #     f"performance_metrics/multi_runs/DTU-GE/vmm_no_recon_kl_annealing_{iter}",
    #     index=False,
    # )

    # # LISI tables
    # clisi_ad = cLISI_full_rank(ad_recon_data, ad_count_bio_label)
    # ilisi_ad = iLISI_full_rank(ad_recon_data, ad_count_batch_label)

    # clisi_ad["iter"] = iter
    # ilisi_ad["iter"] = iter

    # performances_clisi_ad = pd.concat(
    #     [performances_clisi_ad, clisi_ad], axis=0
    # ).reset_index(drop=True)
    # performances_ilisi_ad = pd.concat(
    #     [performances_ilisi_ad, ilisi_ad], axis=0
    # ).reset_index(drop=True)

    # clisi_ibd = cLISI_full_rank(ibd_recon_data, ibd_bio_label)
    # ilisi_ibd = iLISI_full_rank(ibd_recon_data, ibd_batch_label)

    # clisi_ibd["iter"] = iter
    # ilisi_ibd["iter"] = iter

    # performances_clisi_ibd = pd.concat(
    #     [performances_clisi_ibd, clisi_ibd], axis=0
    # ).reset_index(drop=True)
    # performances_ilisi_ibd = pd.concat(
    #     [performances_ilisi_ibd, ilisi_ibd], axis=0
    # ).reset_index(drop=True)

    # clisi_dtu = cLISI_full_rank(dtu_recon_data, dtu_bio_label)
    # ilisi_dtu = iLISI_full_rank(dtu_recon_data, dtu_batch_label)

    # clisi_dtu["iter"] = iter
    # ilisi_dtu["iter"] = iter

    # performances_clisi_dtu = pd.concat(
    #     [performances_clisi_dtu, clisi_dtu], axis=0
    # ).reset_index(drop=True)
    # performances_ilisi_dtu = pd.concat(
    #     [performances_ilisi_dtu, ilisi_dtu], axis=0
    # ).reset_index(drop=True)

    # # PERMANOVA
    # bc, ait = PERMANOVA(
    #     ad_recon_data, ad_count_sample_label, ad_count_batch_label, ad_count_bio_label
    # )

    # bc = pd.DataFrame(
    #     {
    #         "R2": [bc["R2"]],
    #         "p-value": [bc["p-value"]],
    #         "test statistic": [bc["test statistic"]],
    #     }
    # )
    # ait = pd.DataFrame(
    #     {
    #         "R2": [ait["R2"]],
    #         "p-value": [ait["p-value"]],
    #         "test statistic": [ait["test statistic"]],
    #     }
    # )

    # permanova_bc_ad = pd.concat([permanova_bc_ad, bc], axis=0).reset_index(drop=True)

    # permanova_ait_ad = pd.concat([permanova_ait_ad, ait], axis=0).reset_index(drop=True)

    # bc, ait = PERMANOVA(
    #     ibd_recon_data, ibd_sample_label, ibd_batch_label, ibd_bio_label
    # )

    # bc = pd.DataFrame(
    #     {
    #         "R2": [bc["R2"]],
    #         "p-value": [bc["p-value"]],
    #         "test statistic": [bc["test statistic"]],
    #     }
    # )
    # ait = pd.DataFrame(
    #     {
    #         "R2": [ait["R2"]],
    #         "p-value": [ait["p-value"]],
    #         "test statistic": [ait["test statistic"]],
    #     }
    # )

    # permanova_bc_ibd = pd.concat([permanova_bc_ibd, bc], axis=0).reset_index(drop=True)

    # permanova_ait_ibd = pd.concat([permanova_ait_ibd, ait], axis=0).reset_index(
    #     drop=True
    # )

    # bc, ait = PERMANOVA(
    #     dtu_recon_data, dtu_sample_label, dtu_batch_label, dtu_bio_label
    # )

    # bc = pd.DataFrame(
    #     {
    #         "R2": [bc["R2"]],
    #         "p-value": [bc["p-value"]],
    #         "test statistic": [bc["test statistic"]],
    #     }
    # )
    # ait = pd.DataFrame(
    #     {
    #         "R2": [ait["R2"]],
    #         "p-value": [ait["p-value"]],
    #         "test statistic": [ait["test statistic"]],
    #     }
    # )

    # permanova_bc_dtu = pd.concat([permanova_bc_dtu, bc], axis=0).reset_index(drop=True)

    # permanova_ait_dtu = pd.concat([permanova_ait_dtu, ait], axis=0).reset_index(
    #     drop=True
    # )

    # # Performance metrics
    # norm_ad_recon_data = DataTransform(
    #     data=ad_recon_data,
    #     factors=[ad_count_sample_label, ad_count_batch_label, ad_count_bio_label],
    #     count=True,
    # )

    # performance_batch, _ = all_metrics(
    #     data=norm_ad_recon_data,
    #     bio_label=ad_count_bio_label,
    #     batch_label=ad_count_batch_label,
    # )
    # performances_batch_ad.append(performance_batch)

    # norm_ibd_recon_data = DataTransform(
    #     data=ibd_recon_data,
    #     factors=[ibd_sample_label, ibd_batch_label, ibd_bio_label],
    #     count=True,
    # )

    # performance_batch, _ = all_metrics(
    #     data=norm_ibd_recon_data, bio_label=ibd_bio_label, batch_label=ibd_batch_label
    # )
    # performances_batch_ibd.append(performance_batch)

    # norm_dtu_recon_data = DataTransform(
    #     data=dtu_recon_data,
    #     factors=[dtu_sample_label, dtu_batch_label, dtu_bio_label],
    #     count=True,
    # )

    # performance_batch, _ = all_metrics(
    #     data=norm_dtu_recon_data, bio_label=dtu_bio_label, batch_label=dtu_batch_label
    # )
    # performances_batch_dtu.append(performance_batch)

    # MANN-WHITNEY U TEST - AD COUNT
    # 1. Compute relative abundances for raw and corrected
    # count_raw = ad_count_data.select_dtypes(include="number")
    # rel_raw = count_raw.div(count_raw.sum(axis=1), axis=0)

    # count_corr = ad_recon_data.select_dtypes(include="number")
    # rel_corr = count_corr.div(count_corr.sum(axis=1), axis=0)

    # # 2. Identify top 10 OTUs by mean relative abundance in the raw data
    # top10 = rel_raw.mean().sort_values(ascending=False).head(10).index

    # # 3. Subset and reshape both to long form, adding a 'Dataset' column
    # df_raw = (
    #     rel_raw[top10]
    #     .reset_index()
    #     .melt(
    #         id_vars="index", value_vars=top10, var_name="OTU", value_name="RelAbundance"
    #     )
    # )
    # df_raw["Dataset"] = "Raw"

    # df_corr = (
    #     rel_corr[top10]
    #     .reset_index()
    #     .melt(
    #         id_vars="index", value_vars=top10, var_name="OTU", value_name="RelAbundance"
    #     )
    # )
    # df_corr["Dataset"] = "Corrected"

    # # 4. Concatenate
    # df_all = pd.concat([df_raw, df_corr], ignore_index=True)
    # df_all.rename(columns={"index": "Sample"}, inplace=True)

    # # 5. Compute Mann-whitney U test for top 10 taxonomic groups
    # stat_results = []

    # for otu in top10:
    #     raw_values = df_all[(df_all["OTU"] == otu) & (df_all["Dataset"] == "Raw")][
    #         "RelAbundance"
    #     ]
    #     corr_values = df_all[
    #         (df_all["OTU"] == otu) & (df_all["Dataset"] == "Corrected")
    #     ]["RelAbundance"]

    #     # Mann-whitney U test
    #     stat, p = mannwhitneyu(
    #         raw_values, corr_values, alternative="two-sided"
    #     )  # Ha two-sided: the distributions aren't equal / there are significant differences
    #     stat_results.append({"OTU": otu, "U-stat": stat, "p-value": p})

    # df_stats = pd.DataFrame(stat_results)
    # df_stats["adj p-value"] = multipletests(df_stats["p-value"], method="fdr_bh")[1]
    # df_stats["iter"] = iter

    # performances_mannwhit_ad = pd.concat(
    #     [performances_mannwhit_ad, df_stats], axis=0
    # ).reset_index(drop=True)

    # # MANN-WHITNEY U TEST - IBD
    # # 1. Compute relative abundances for raw and corrected
    # count_raw = ibd_data.select_dtypes(include="number")
    # rel_raw = count_raw.div(count_raw.sum(axis=1), axis=0)

    # count_corr = ibd_recon_data.select_dtypes(include="number")
    # rel_corr = count_corr.div(count_corr.sum(axis=1), axis=0)

    # # 2. Identify top 10 OTUs by mean relative abundance in the raw data
    # top10 = rel_raw.mean().sort_values(ascending=False).head(10).index

    # # 3. Subset and reshape both to long form, adding a 'Dataset' column
    # df_raw = (
    #     rel_raw[top10]
    #     .reset_index()
    #     .melt(
    #         id_vars="index", value_vars=top10, var_name="OTU", value_name="RelAbundance"
    #     )
    # )
    # df_raw["Dataset"] = "Raw"

    # df_corr = (
    #     rel_corr[top10]
    #     .reset_index()
    #     .melt(
    #         id_vars="index", value_vars=top10, var_name="OTU", value_name="RelAbundance"
    #     )
    # )
    # df_corr["Dataset"] = "Corrected"

    # # 4. Concatenate
    # df_all = pd.concat([df_raw, df_corr], ignore_index=True)
    # df_all.rename(columns={"index": "Sample"}, inplace=True)

    # # 5. Compute Mann-whitney U test for top 10 taxonomic groups
    # stat_results = []

    # for otu in top10:
    #     raw_values = df_all[(df_all["OTU"] == otu) & (df_all["Dataset"] == "Raw")][
    #         "RelAbundance"
    #     ]
    #     corr_values = df_all[
    #         (df_all["OTU"] == otu) & (df_all["Dataset"] == "Corrected")
    #     ]["RelAbundance"]

    #     # Mann-whitney U test
    #     stat, p = mannwhitneyu(
    #         raw_values, corr_values, alternative="two-sided"
    #     )  # Ha two-sided: the distributions aren't equal / there are significant differences
    #     stat_results.append({"OTU": otu, "U-stat": stat, "p-value": p})

    # df_stats = pd.DataFrame(stat_results)
    # df_stats["adj p-value"] = multipletests(df_stats["p-value"], method="fdr_bh")[1]
    # df_stats["iter"] = iter

    # performances_mannwhit_ibd = pd.concat(
    #     [performances_mannwhit_ibd, df_stats], axis=0
    # ).reset_index(drop=True)

    # # MANN-WHITNEY U TEST - DTU-GE
    # # 1. Compute relative abundances for raw and corrected
    # count_raw = dtu_data.select_dtypes(include="number")
    # rel_raw = count_raw.div(count_raw.sum(axis=1), axis=0)

    # count_corr = dtu_recon_data.select_dtypes(include="number")
    # rel_corr = count_corr.div(count_corr.sum(axis=1), axis=0)

    # # 2. Identify top 10 OTUs by mean relative abundance in the raw data
    # top10 = rel_raw.mean().sort_values(ascending=False).head(10).index

    # # 3. Subset and reshape both to long form, adding a 'Dataset' column
    # df_raw = (
    #     rel_raw[top10]
    #     .reset_index()
    #     .melt(
    #         id_vars="index", value_vars=top10, var_name="OTU", value_name="RelAbundance"
    #     )
    # )
    # df_raw["Dataset"] = "Raw"

    # df_corr = (
    #     rel_corr[top10]
    #     .reset_index()
    #     .melt(
    #         id_vars="index", value_vars=top10, var_name="OTU", value_name="RelAbundance"
    #     )
    # )
    # df_corr["Dataset"] = "Corrected"

    # # 4. Concatenate
    # df_all = pd.concat([df_raw, df_corr], ignore_index=True)
    # df_all.rename(columns={"index": "Sample"}, inplace=True)

    # # 5. Compute Mann-whitney U test for top 10 taxonomic groups
    # stat_results = []

    # for otu in top10:
    #     raw_values = df_all[(df_all["OTU"] == otu) & (df_all["Dataset"] == "Raw")][
    #         "RelAbundance"
    #     ]
    #     corr_values = df_all[
    #         (df_all["OTU"] == otu) & (df_all["Dataset"] == "Corrected")
    #     ]["RelAbundance"]

    #     # Mann-whitney U test
    #     stat, p = mannwhitneyu(
    #         raw_values, corr_values, alternative="two-sided"
    #     )  # Ha two-sided: the distributions aren't equal / there are significant differences
    #     stat_results.append({"OTU": otu, "U-stat": stat, "p-value": p})

    # df_stats = pd.DataFrame(stat_results)
    # df_stats["adj p-value"] = multipletests(df_stats["p-value"], method="fdr_bh")[1]
    # df_stats["iter"] = iter

    # performances_mannwhit_dtu = pd.concat(
    #     [performances_mannwhit_dtu, df_stats], axis=0
    # ).reset_index(drop=True)

    # Save performance to file
    # pd.DataFrame(performances_batch_ad).to_csv(
    #     "performance_metrics/deterministic/AD_count/VMM_no_cycle_batch.csv", index=False
    # )
    # pd.DataFrame(performances_clisi_ad).to_csv(
    #     "performance_metrics/deterministic/AD_count/VMM_no_cycle_clisi.csv", index=False
    # )
    # pd.DataFrame(performances_ilisi_ad).to_csv(
    #     "performance_metrics/deterministic/AD_count/VMM_no_cycle_ilisi.csv", index=False
    # )
    # pd.DataFrame(permanova_ait_ad).to_csv(
    #     "performance_metrics/deterministic/AD_count/VMM_no_cycle_perma_ait.csv", index=False
    # )
    # pd.DataFrame(permanova_bc_ad).to_csv(
    #     "performance_metrics/deterministic/AD_count/VMM_no_cycle_perma_bc.csv", index=False
    # )
    # pd.DataFrame(performances_mannwhit_ad).to_csv(
    #     "performance_metrics/deterministic/AD_count/VMM_no_cycle_mannwhit.csv", index=False
    # )

    # pd.DataFrame(performances_batch_ibd).to_csv(
    #     "performance_metrics/deterministic/IBD/VMM_no_cycle_batch.csv", index=False
    # )
    # pd.DataFrame(performances_clisi_ibd).to_csv(
    #     "performance_metrics/deterministic/IBD/VMM_no_cycle_clisi.csv", index=False
    # )
    # pd.DataFrame(performances_ilisi_ibd).to_csv(
    #     "performance_metrics/deterministic/IBD/VMM_no_cycle_ilisi.csv", index=False
    # )
    # pd.DataFrame(permanova_ait_ibd).to_csv(
    #     "performance_metrics/deterministic/IBD/VMM_no_cycle_perma_ait.csv", index=False
    # )
    # pd.DataFrame(permanova_bc_ibd).to_csv(
    #     "performance_metrics/deterministic/IBD/VMM_no_cycle_perma_bc.csv", index=False
    # )
    # pd.DataFrame(performances_mannwhit_ibd).to_csv(
    #     "performance_metrics/deterministic/IBD/VMM_no_cycle_mannwhit.csv", index=False
    # )

    # pd.DataFrame(performances_batch_dtu).to_csv(
    #     "performance_metrics/deterministic/DTU-GE/VMM_no_cycle_batch.csv", index=False
    # )
    # pd.DataFrame(performances_clisi_dtu).to_csv(
    #     "performance_metrics/deterministic/DTU-GE/VMM_no_cycle_clisi.csv", index=False
    # )
    # pd.DataFrame(performances_ilisi_dtu).to_csv(
    #     "performance_metrics/deterministic/DTU-GE/VMM_no_cycle_ilisi.csv", index=False
    # )
    # pd.DataFrame(permanova_ait_dtu).to_csv(
    #     "performance_metrics/deterministic/DTU-GE/VMM_no_cycle_perma_ait.csv", index=False
    # )
    # pd.DataFrame(permanova_bc_dtu).to_csv(
    #     "performance_metrics/deterministic/DTU-GE/VMM_no_cycle_perma_bc.csv", index=False
    # )
    # pd.DataFrame(performances_mannwhit_dtu).to_csv(
    #     "performance_metrics/deterministic/DTU-GE/VMM_no_cycle_mannwhit.csv", index=False
    # )

    # Set seed
    if iter == 0:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    performances_batch_ad = []
    performances_ilisi_ad = pd.DataFrame()
    performances_clisi_ad = pd.DataFrame()
    permanova_ait_ad = pd.DataFrame()
    permanova_bc_ad = pd.DataFrame()

    performances_batch_ibd = []
    performances_ilisi_ibd = pd.DataFrame()
    performances_clisi_ibd = pd.DataFrame()
    permanova_ait_ibd = pd.DataFrame()
    permanova_bc_ibd = pd.DataFrame()

    performances_batch_dtu = []
    performances_ilisi_dtu = pd.DataFrame()
    performances_clisi_dtu = pd.DataFrame()
    permanova_ait_dtu = pd.DataFrame()
    permanova_bc_dtu = pd.DataFrame()

    performances_mannwhit_ad = pd.DataFrame()
    performances_mannwhit_ibd = pd.DataFrame()
    performances_mannwhit_dtu = pd.DataFrame()

    # # for performance metrics
    # proc = psutil.Process()
    # # --- pre‐run snapshots ---
    # t0_wall = time.time()
    # t0_cpu = sum(proc.cpu_times()[:2])  # user + sys CPU s
    # mem0 = proc.memory_info().rss  # resident set size
    # torch.cuda.reset_peak_memory_stats()  # clear PyTorch peak

    # # MoG models
    # ad_count_mog_no_cycle = abaco_run(
    #     dataloader=ad_count_train_dataloader,
    #     n_batches=ad_count_batch_size,
    #     n_bios=ad_count_bio_size,
    #     input_size=ad_count_input_size,
    #     device=device,
    #     prior="MoG",
    #     w_contra=100.0,
    #     kl_cycle=False,
    #     seed=None,
    #     smooth_annealing=True,
    #     pre_epochs=2500,
    #     post_epochs=5000,
    #     vae_post_lr=2e-4,
    #     adv_lr=1e-7,
    #     disc_lr=1e-7,
    # )

    # # --- post‐run snapshots ---
    # t1_wall = time.time()
    # t1_cpu = sum(proc.cpu_times()[:2])
    # mem1 = proc.memory_info().rss
    # peak_gpu = torch.cuda.max_memory_allocated()

    # # --- GPU‐side current status (optional) ---
    # gpus = GPUtil.getGPUs()

    # # --- print a neat summary ---
    # print("===== AD count - MoG no cycle Performance =====")
    # print(f"Wall‐clock time  : {t1_wall - t0_wall:.1f} s")
    # print(f"CPU time         : {t1_cpu  - t0_cpu :.1f} s")
    # print(f"RAM change       : {(mem1 - mem0)/1024**3:+.2f} GiB")
    # print(f"Peak GPU memory : {peak_gpu/1024**3:.2f} GiB")
    # for g in gpus:
    #     print(
    #         f" GPU {g.id}: util={g.load*100:.1f}%, mem={g.memoryUsed/1024**3:.1f}/{g.memoryTotal/1024**3:.1f} GiB"
    #     )

    if iter == 0:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # # for performance metrics
    # proc = psutil.Process()
    # # --- pre‐run snapshots ---
    # t0_wall = time.time()
    # t0_cpu = sum(proc.cpu_times()[:2])  # user + sys CPU s
    # mem0 = proc.memory_info().rss  # resident set size
    # torch.cuda.reset_peak_memory_stats()  # clear PyTorch peak

    # ibd_mog_no_cycle = abaco_run(
    #     dataloader=ibd_train_dataloader,
    #     n_batches=ibd_batch_size,
    #     n_bios=ibd_bio_size,
    #     input_size=ibd_input_size,
    #     device=device,
    #     prior="MoG",
    #     w_contra=1000.0,
    #     kl_cycle=False,
    #     seed=None,
    #     smooth_annealing=True,
    #     pre_epochs=5000,
    #     post_epochs=2500,
    #     vae_post_lr=1e-4,
    #     encoder_net=[2048, 1024, 512, 256],
    #     decoder_net=[256, 512, 1024, 2048],
    #     adv_lr=1e-6,
    #     disc_lr=1e-6,
    # )

    # # --- post‐run snapshots ---
    # t1_wall = time.time()
    # t1_cpu = sum(proc.cpu_times()[:2])
    # mem1 = proc.memory_info().rss
    # peak_gpu = torch.cuda.max_memory_allocated()

    # # --- GPU‐side current status (optional) ---
    # gpus = GPUtil.getGPUs()

    # # --- print a neat summary ---
    # print("===== IBD - MoG no cycle Performance =====")
    # print(f"Wall‐clock time  : {t1_wall - t0_wall:.1f} s")
    # print(f"CPU time         : {t1_cpu  - t0_cpu :.1f} s")
    # print(f"RAM change       : {(mem1 - mem0)/1024**3:+.2f} GiB")
    # print(f"Peak GPU memory : {peak_gpu/1024**3:.2f} GiB")
    # for g in gpus:
    #     print(
    #         f" GPU {g.id}: util={g.load*100:.1f}%, mem={g.memoryUsed/1024**3:.1f}/{g.memoryTotal/1024**3:.1f} GiB"
    #     )

    if iter == 0:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # # for performance metrics
    # proc = psutil.Process()
    # # --- pre‐run snapshots ---
    # t0_wall = time.time()
    # t0_cpu = sum(proc.cpu_times()[:2])  # user + sys CPU s
    # mem0 = proc.memory_info().rss  # resident set size
    # torch.cuda.reset_peak_memory_stats()  # clear PyTorch peak

    # dtu_mog_no_cycle = abaco_run(
    #     dataloader=dtu_train_dataloader,
    #     n_batches=dtu_batch_size,
    #     n_bios=dtu_bio_size,
    #     input_size=dtu_input_size,
    #     device=device,
    #     prior="MoG",
    #     w_contra=1000.0,
    #     kl_cycle=False,
    #     seed=None,
    #     smooth_annealing=True,
    #     pre_epochs=2500,
    #     post_epochs=5000,
    #     vae_post_lr=2e-4,
    #     adv_lr=1e-5,
    #     disc_lr=1e-5,
    # )

    # # --- post‐run snapshots ---
    # t1_wall = time.time()
    # t1_cpu = sum(proc.cpu_times()[:2])
    # mem1 = proc.memory_info().rss
    # peak_gpu = torch.cuda.max_memory_allocated()

    # # --- GPU‐side current status (optional) ---
    # gpus = GPUtil.getGPUs()

    # # --- print a neat summary ---
    # print("===== DTU-GE - MoG no cycle Performance =====")
    # print(f"Wall‐clock time  : {t1_wall - t0_wall:.1f} s")
    # print(f"CPU time         : {t1_cpu  - t0_cpu :.1f} s")
    # print(f"RAM change       : {(mem1 - mem0)/1024**3:+.2f} GiB")
    # print(f"Peak GPU memory : {peak_gpu/1024**3:.2f} GiB")
    # for g in gpus:
    #     print(
    #         f" GPU {g.id}: util={g.load*100:.1f}%, mem={g.memoryUsed/1024**3:.1f}/{g.memoryTotal/1024**3:.1f} GiB"
    #     )

    if iter == 0:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Reconstruct data
    # ad_recon_data = abaco_recon(
    #     model=ad_count_mog_no_cycle,
    #     device=device,
    #     data=ad_count_data,
    #     dataloader=ad_count_train_dataloader,
    #     sample_label=ad_count_sample_label,
    #     batch_label=ad_count_batch_label,
    #     bio_label=ad_count_bio_label,
    #     seed=None,
    #     monte_carlo=1,
    # )

    if iter == 0:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ibd_recon_data = abaco_recon(
    #     model=ibd_mog_no_cycle,
    #     device=device,
    #     data=ibd_data,
    #     dataloader=ibd_train_dataloader,
    #     sample_label=ibd_sample_label,
    #     batch_label=ibd_batch_label,
    #     bio_label=ibd_bio_label,
    #     seed=None,
    #     monte_carlo=1,
    # )

    if iter == 0:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # dtu_recon_data = abaco_recon(
    #     model=dtu_mog_no_cycle,
    #     device=device,
    #     data=dtu_data,
    #     dataloader=dtu_train_dataloader,
    #     sample_label=dtu_sample_label,
    #     batch_label=dtu_batch_label,
    #     bio_label=dtu_bio_label,
    #     seed=None,
    #     monte_carlo=1,
    # )

    # ad_recon_data.to_csv(
    #     f"performance_metrics/multi_runs/AD_count/mog_no_recon_{iter}", index=False
    # )
    # ibd_recon_data.to_csv(
    #     f"performance_metrics/multi_runs/IBD/mog_no_recon_contra_1000_2_{iter}",
    #     index=False,
    # )
    # dtu_recon_data.to_csv(
    #     f"performance_metrics/multi_runs/DTU-GE/mog_no_recon_contra_1000_{iter}",
    #     index=False,
    # )

    # # LISI tables
    # clisi_ad = cLISI_full_rank(ad_recon_data, ad_count_bio_label)
    # ilisi_ad = iLISI_full_rank(ad_recon_data, ad_count_batch_label)

    # clisi_ad["iter"] = iter
    # ilisi_ad["iter"] = iter

    # performances_clisi_ad = pd.concat(
    #     [performances_clisi_ad, clisi_ad], axis=0
    # ).reset_index(drop=True)
    # performances_ilisi_ad = pd.concat(
    #     [performances_ilisi_ad, ilisi_ad], axis=0
    # ).reset_index(drop=True)

    # clisi_ibd = cLISI_full_rank(ibd_recon_data, ibd_bio_label)
    # ilisi_ibd = iLISI_full_rank(ibd_recon_data, ibd_batch_label)

    # clisi_ibd["iter"] = iter
    # ilisi_ibd["iter"] = iter

    # performances_clisi_ibd = pd.concat(
    #     [performances_clisi_ibd, clisi_ibd], axis=0
    # ).reset_index(drop=True)
    # performances_ilisi_ibd = pd.concat(
    #     [performances_ilisi_ibd, ilisi_ibd], axis=0
    # ).reset_index(drop=True)

    # clisi_dtu = cLISI_full_rank(dtu_recon_data, dtu_bio_label)
    # ilisi_dtu = iLISI_full_rank(dtu_recon_data, dtu_batch_label)

    # clisi_dtu["iter"] = iter
    # ilisi_dtu["iter"] = iter

    # performances_clisi_dtu = pd.concat(
    #     [performances_clisi_dtu, clisi_dtu], axis=0
    # ).reset_index(drop=True)
    # performances_ilisi_dtu = pd.concat(
    #     [performances_ilisi_dtu, ilisi_dtu], axis=0
    # ).reset_index(drop=True)

    # # PERMANOVA
    # bc, ait = PERMANOVA(
    #     ad_recon_data, ad_count_sample_label, ad_count_batch_label, ad_count_bio_label
    # )

    # bc = pd.DataFrame(
    #     {
    #         "R2": [bc["R2"]],
    #         "p-value": [bc["p-value"]],
    #         "test statistic": [bc["test statistic"]],
    #     }
    # )
    # ait = pd.DataFrame(
    #     {
    #         "R2": [ait["R2"]],
    #         "p-value": [ait["p-value"]],
    #         "test statistic": [ait["test statistic"]],
    #     }
    # )

    # permanova_bc_ad = pd.concat([permanova_bc_ad, bc], axis=0).reset_index(drop=True)

    # permanova_ait_ad = pd.concat([permanova_ait_ad, ait], axis=0).reset_index(drop=True)

    # bc, ait = PERMANOVA(
    #     ibd_recon_data, ibd_sample_label, ibd_batch_label, ibd_bio_label
    # )

    # bc = pd.DataFrame(
    #     {
    #         "R2": [bc["R2"]],
    #         "p-value": [bc["p-value"]],
    #         "test statistic": [bc["test statistic"]],
    #     }
    # )
    # ait = pd.DataFrame(
    #     {
    #         "R2": [ait["R2"]],
    #         "p-value": [ait["p-value"]],
    #         "test statistic": [ait["test statistic"]],
    #     }
    # )

    # permanova_bc_ibd = pd.concat([permanova_bc_ibd, bc], axis=0).reset_index(drop=True)

    # permanova_ait_ibd = pd.concat([permanova_ait_ibd, ait], axis=0).reset_index(
    #     drop=True
    # )

    # bc, ait = PERMANOVA(
    #     dtu_recon_data, dtu_sample_label, dtu_batch_label, dtu_bio_label
    # )

    # bc = pd.DataFrame(
    #     {
    #         "R2": [bc["R2"]],
    #         "p-value": [bc["p-value"]],
    #         "test statistic": [bc["test statistic"]],
    #     }
    # )
    # ait = pd.DataFrame(
    #     {
    #         "R2": [ait["R2"]],
    #         "p-value": [ait["p-value"]],
    #         "test statistic": [ait["test statistic"]],
    #     }
    # )

    # permanova_bc_dtu = pd.concat([permanova_bc_dtu, bc], axis=0).reset_index(drop=True)

    # permanova_ait_dtu = pd.concat([permanova_ait_dtu, ait], axis=0).reset_index(
    #     drop=True
    # )

    # # Performance metrics
    # norm_ad_recon_data = DataTransform(
    #     data=ad_recon_data,
    #     factors=[ad_count_sample_label, ad_count_batch_label, ad_count_bio_label],
    #     count=True,
    # )

    # performance_batch, _ = all_metrics(
    #     data=norm_ad_recon_data,
    #     bio_label=ad_count_bio_label,
    #     batch_label=ad_count_batch_label,
    # )
    # performances_batch_ad.append(performance_batch)

    # norm_ibd_recon_data = DataTransform(
    #     data=ibd_recon_data,
    #     factors=[ibd_sample_label, ibd_batch_label, ibd_bio_label],
    #     count=True,
    # )

    # performance_batch, _ = all_metrics(
    #     data=norm_ibd_recon_data, bio_label=ibd_bio_label, batch_label=ibd_batch_label
    # )
    # performances_batch_ibd.append(performance_batch)

    # norm_dtu_recon_data = DataTransform(
    #     data=dtu_recon_data,
    #     factors=[dtu_sample_label, dtu_batch_label, dtu_bio_label],
    #     count=True,
    # )

    # performance_batch, _ = all_metrics(
    #     data=norm_dtu_recon_data, bio_label=dtu_bio_label, batch_label=dtu_batch_label
    # )
    # performances_batch_dtu.append(performance_batch)

    # MANN-WHITNEY U TEST - AD COUNT
    # 1. Compute relative abundances for raw and corrected
    # count_raw = ad_count_data.select_dtypes(include="number")
    # rel_raw = count_raw.div(count_raw.sum(axis=1), axis=0)

    # count_corr = ad_recon_data.select_dtypes(include="number")
    # rel_corr = count_corr.div(count_corr.sum(axis=1), axis=0)

    # # 2. Identify top 10 OTUs by mean relative abundance in the raw data
    # top10 = rel_raw.mean().sort_values(ascending=False).head(10).index

    # # 3. Subset and reshape both to long form, adding a 'Dataset' column
    # df_raw = (
    #     rel_raw[top10]
    #     .reset_index()
    #     .melt(
    #         id_vars="index", value_vars=top10, var_name="OTU", value_name="RelAbundance"
    #     )
    # )
    # df_raw["Dataset"] = "Raw"

    # df_corr = (
    #     rel_corr[top10]
    #     .reset_index()
    #     .melt(
    #         id_vars="index", value_vars=top10, var_name="OTU", value_name="RelAbundance"
    #     )
    # )
    # df_corr["Dataset"] = "Corrected"

    # # 4. Concatenate
    # df_all = pd.concat([df_raw, df_corr], ignore_index=True)
    # df_all.rename(columns={"index": "Sample"}, inplace=True)

    # # 5. Compute Mann-whitney U test for top 10 taxonomic groups
    # stat_results = []

    # for otu in top10:
    #     raw_values = df_all[(df_all["OTU"] == otu) & (df_all["Dataset"] == "Raw")][
    #         "RelAbundance"
    #     ]
    #     corr_values = df_all[
    #         (df_all["OTU"] == otu) & (df_all["Dataset"] == "Corrected")
    #     ]["RelAbundance"]

    #     # Mann-whitney U test
    #     stat, p = mannwhitneyu(
    #         raw_values, corr_values, alternative="two-sided"
    #     )  # Ha two-sided: the distributions aren't equal / there are significant differences
    #     stat_results.append({"OTU": otu, "U-stat": stat, "p-value": p})

    # df_stats = pd.DataFrame(stat_results)
    # df_stats["adj p-value"] = multipletests(df_stats["p-value"], method="fdr_bh")[1]
    # df_stats["iter"] = iter

    # performances_mannwhit_ad = pd.concat(
    #     [performances_mannwhit_ad, df_stats], axis=0
    # ).reset_index(drop=True)

    # # MANN-WHITNEY U TEST - IBD
    # # 1. Compute relative abundances for raw and corrected
    # count_raw = ibd_data.select_dtypes(include="number")
    # rel_raw = count_raw.div(count_raw.sum(axis=1), axis=0)

    # count_corr = ibd_recon_data.select_dtypes(include="number")
    # rel_corr = count_corr.div(count_corr.sum(axis=1), axis=0)

    # # 2. Identify top 10 OTUs by mean relative abundance in the raw data
    # top10 = rel_raw.mean().sort_values(ascending=False).head(10).index

    # # 3. Subset and reshape both to long form, adding a 'Dataset' column
    # df_raw = (
    #     rel_raw[top10]
    #     .reset_index()
    #     .melt(
    #         id_vars="index", value_vars=top10, var_name="OTU", value_name="RelAbundance"
    #     )
    # )
    # df_raw["Dataset"] = "Raw"

    # df_corr = (
    #     rel_corr[top10]
    #     .reset_index()
    #     .melt(
    #         id_vars="index", value_vars=top10, var_name="OTU", value_name="RelAbundance"
    #     )
    # )
    # df_corr["Dataset"] = "Corrected"

    # # 4. Concatenate
    # df_all = pd.concat([df_raw, df_corr], ignore_index=True)
    # df_all.rename(columns={"index": "Sample"}, inplace=True)

    # # 5. Compute Mann-whitney U test for top 10 taxonomic groups
    # stat_results = []

    # for otu in top10:
    #     raw_values = df_all[(df_all["OTU"] == otu) & (df_all["Dataset"] == "Raw")][
    #         "RelAbundance"
    #     ]
    #     corr_values = df_all[
    #         (df_all["OTU"] == otu) & (df_all["Dataset"] == "Corrected")
    #     ]["RelAbundance"]

    #     # Mann-whitney U test
    #     stat, p = mannwhitneyu(
    #         raw_values, corr_values, alternative="two-sided"
    #     )  # Ha two-sided: the distributions aren't equal / there are significant differences
    #     stat_results.append({"OTU": otu, "U-stat": stat, "p-value": p})

    # df_stats = pd.DataFrame(stat_results)
    # df_stats["adj p-value"] = multipletests(df_stats["p-value"], method="fdr_bh")[1]
    # df_stats["iter"] = iter

    # performances_mannwhit_ibd = pd.concat(
    #     [performances_mannwhit_ibd, df_stats], axis=0
    # ).reset_index(drop=True)

    # # MANN-WHITNEY U TEST - DTU-GE
    # # 1. Compute relative abundances for raw and corrected
    # count_raw = dtu_data.select_dtypes(include="number")
    # rel_raw = count_raw.div(count_raw.sum(axis=1), axis=0)

    # count_corr = dtu_recon_data.select_dtypes(include="number")
    # rel_corr = count_corr.div(count_corr.sum(axis=1), axis=0)

    # # 2. Identify top 10 OTUs by mean relative abundance in the raw data
    # top10 = rel_raw.mean().sort_values(ascending=False).head(10).index

    # # 3. Subset and reshape both to long form, adding a 'Dataset' column
    # df_raw = (
    #     rel_raw[top10]
    #     .reset_index()
    #     .melt(
    #         id_vars="index", value_vars=top10, var_name="OTU", value_name="RelAbundance"
    #     )
    # )
    # df_raw["Dataset"] = "Raw"

    # df_corr = (
    #     rel_corr[top10]
    #     .reset_index()
    #     .melt(
    #         id_vars="index", value_vars=top10, var_name="OTU", value_name="RelAbundance"
    #     )
    # )
    # df_corr["Dataset"] = "Corrected"

    # # 4. Concatenate
    # df_all = pd.concat([df_raw, df_corr], ignore_index=True)
    # df_all.rename(columns={"index": "Sample"}, inplace=True)

    # # 5. Compute Mann-whitney U test for top 10 taxonomic groups
    # stat_results = []

    # for otu in top10:
    #     raw_values = df_all[(df_all["OTU"] == otu) & (df_all["Dataset"] == "Raw")][
    #         "RelAbundance"
    #     ]
    #     corr_values = df_all[
    #         (df_all["OTU"] == otu) & (df_all["Dataset"] == "Corrected")
    #     ]["RelAbundance"]

    #     # Mann-whitney U test
    #     stat, p = mannwhitneyu(
    #         raw_values, corr_values, alternative="two-sided"
    #     )  # Ha two-sided: the distributions aren't equal / there are significant differences
    #     stat_results.append({"OTU": otu, "U-stat": stat, "p-value": p})

    # df_stats = pd.DataFrame(stat_results)
    # df_stats["adj p-value"] = multipletests(df_stats["p-value"], method="fdr_bh")[1]
    # df_stats["iter"] = iter

    # performances_mannwhit_dtu = pd.concat(
    #     [performances_mannwhit_dtu, df_stats], axis=0
    # ).reset_index(drop=True)

    # Save performance to file
    # pd.DataFrame(performances_batch_ad).to_csv(
    #     "performance_metrics/deterministic/AD_count/MoG_no_cycle_batch.csv", index=False
    # )
    # pd.DataFrame(performances_clisi_ad).to_csv(
    #     "performance_metrics/deterministic/AD_count/MoG_no_cycle_clisi.csv", index=False
    # )
    # pd.DataFrame(performances_ilisi_ad).to_csv(
    #     "performance_metrics/deterministic/AD_count/MoG_no_cycle_ilisi.csv", index=False
    # )
    # pd.DataFrame(permanova_ait_ad).to_csv(
    #     "performance_metrics/deterministic/AD_count/MoG_no_cycle_perma_ait.csv", index=False
    # )
    # pd.DataFrame(permanova_bc_ad).to_csv(
    #     "performance_metrics/deterministic/AD_count/MoG_no_cycle_perma_bc.csv", index=False
    # )
    # pd.DataFrame(performances_mannwhit_ad).to_csv(
    #     "performance_metrics/deterministic/AD_count/MoG_no_cycle_mannwhit.csv", index=False
    # )

    # pd.DataFrame(performances_batch_ibd).to_csv(
    #     "performance_metrics/deterministic/IBD/MoG_no_cycle_batch.csv", index=False
    # )
    # pd.DataFrame(performances_clisi_ibd).to_csv(
    #     "performance_metrics/deterministic/IBD/MoG_no_cycle_clisi.csv", index=False
    # )
    # pd.DataFrame(performances_ilisi_ibd).to_csv(
    #     "performance_metrics/deterministic/IBD/MoG_no_cycle_ilisi.csv", index=False
    # )
    # pd.DataFrame(permanova_ait_ibd).to_csv(
    #     "performance_metrics/deterministic/IBD/MoG_no_cycle_perma_ait.csv", index=False
    # )
    # pd.DataFrame(permanova_bc_ibd).to_csv(
    #     "performance_metrics/deterministic/IBD/MoG_no_cycle_perma_bc.csv", index=False
    # )
    # pd.DataFrame(performances_mannwhit_ibd).to_csv(
    #     "performance_metrics/deterministic/IBD/MoG_no_cycle_mannwhit.csv", index=False
    # )

    # pd.DataFrame(performances_batch_dtu).to_csv(
    #     "performance_metrics/deterministic/DTU-GE/MoG_no_cycle_batch.csv", index=False
    # )
    # pd.DataFrame(performances_clisi_dtu).to_csv(
    #     "performance_metrics/deterministic/DTU-GE/MoG_no_cycle_clisi.csv", index=False
    # )
    # pd.DataFrame(performances_ilisi_dtu).to_csv(
    #     "performance_metrics/deterministic/DTU-GE/MoG_no_cycle_ilisi.csv", index=False
    # )
    # pd.DataFrame(permanova_ait_dtu).to_csv(
    #     "performance_metrics/deterministic/DTU-GE/MoG_no_cycle_perma_ait.csv", index=False
    # )
    # pd.DataFrame(permanova_bc_dtu).to_csv(
    #     "performance_metrics/deterministic/DTU-GE/MoG_no_cycle_perma_bc.csv", index=False
    # )
    # pd.DataFrame(performances_mannwhit_dtu).to_csv(
    #     "performance_metrics/deterministic/DTU-GE/MoG_no_cycle_mannwhit.csv", index=False
    # )

    # Set seed for first iteration
    if iter == 0:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    performances_batch_ad = []
    performances_ilisi_ad = pd.DataFrame()
    performances_clisi_ad = pd.DataFrame()
    permanova_ait_ad = pd.DataFrame()
    permanova_bc_ad = pd.DataFrame()

    performances_batch_ibd = []
    performances_ilisi_ibd = pd.DataFrame()
    performances_clisi_ibd = pd.DataFrame()
    permanova_ait_ibd = pd.DataFrame()
    permanova_bc_ibd = pd.DataFrame()

    performances_batch_dtu = []
    performances_ilisi_dtu = pd.DataFrame()
    performances_clisi_dtu = pd.DataFrame()
    permanova_ait_dtu = pd.DataFrame()
    permanova_bc_dtu = pd.DataFrame()

    performances_mannwhit_ad = pd.DataFrame()
    performances_mannwhit_ibd = pd.DataFrame()
    performances_mannwhit_dtu = pd.DataFrame()

    # # for performance metrics
    # proc = psutil.Process()
    # # --- pre‐run snapshots ---
    # t0_wall = time.time()
    # t0_cpu = sum(proc.cpu_times()[:2])  # user + sys CPU s
    # mem0 = proc.memory_info().rss  # resident set size
    # torch.cuda.reset_peak_memory_stats()  # clear PyTorch peak

    # # VMM models
    # ad_count_std_no_cycle = abaco_run(
    #     dataloader=ad_count_train_dataloader,
    #     n_batches=ad_count_batch_size,
    #     n_bios=ad_count_bio_size,
    #     input_size=ad_count_input_size,
    #     device=device,
    #     prior="Normal",
    #     w_contra=100.0,
    #     kl_cycle=False,
    #     seed=None,
    #     smooth_annealing=True,
    #     pre_epochs=2500,
    #     post_epochs=5000,
    #     vae_post_lr=2e-4,
    #     adv_lr=1e-7,
    #     disc_lr=1e-7,
    # )

    # # --- post‐run snapshots ---
    # t1_wall = time.time()
    # t1_cpu = sum(proc.cpu_times()[:2])
    # mem1 = proc.memory_info().rss
    # peak_gpu = torch.cuda.max_memory_allocated()

    # # --- GPU‐side current status (optional) ---
    # gpus = GPUtil.getGPUs()

    # # --- print a neat summary ---
    # print("===== AD count - Std no cycle Performance =====")
    # print(f"Wall‐clock time  : {t1_wall - t0_wall:.1f} s")
    # print(f"CPU time         : {t1_cpu  - t0_cpu :.1f} s")
    # print(f"RAM change       : {(mem1 - mem0)/1024**3:+.2f} GiB")
    # print(f"Peak GPU memory : {peak_gpu/1024**3:.2f} GiB")
    # for g in gpus:
    #     print(
    #         f" GPU {g.id}: util={g.load*100:.1f}%, mem={g.memoryUsed/1024**3:.1f}/{g.memoryTotal/1024**3:.1f} GiB"
    #     )

    if iter == 0:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # # for performance metrics
    # proc = psutil.Process()
    # # --- pre‐run snapshots ---
    # t0_wall = time.time()
    # t0_cpu = sum(proc.cpu_times()[:2])  # user + sys CPU s
    # mem0 = proc.memory_info().rss  # resident set size
    # torch.cuda.reset_peak_memory_stats()  # clear PyTorch peak

    # ibd_std_no_cycle = abaco_run(
    #     dataloader=ibd_train_dataloader,
    #     n_batches=ibd_batch_size,
    #     n_bios=ibd_bio_size,
    #     input_size=ibd_input_size,
    #     device=device,
    #     prior="Normal",
    #     w_contra=1000.0,
    #     kl_cycle=False,
    #     seed=None,
    #     smooth_annealing=True,
    #     pre_epochs=5000,
    #     post_epochs=2500,
    #     vae_post_lr=1e-4,
    #     encoder_net=[2048, 1024, 512, 256],
    #     decoder_net=[256, 512, 1024, 2048],
    #     adv_lr=1e-6,
    #     disc_lr=1e-6,
    # )

    # # --- post‐run snapshots ---
    # t1_wall = time.time()
    # t1_cpu = sum(proc.cpu_times()[:2])
    # mem1 = proc.memory_info().rss
    # peak_gpu = torch.cuda.max_memory_allocated()

    # # --- GPU‐side current status (optional) ---
    # gpus = GPUtil.getGPUs()

    # # --- print a neat summary ---
    # print("===== IBD - Std no cycle Performance =====")
    # print(f"Wall‐clock time  : {t1_wall - t0_wall:.1f} s")
    # print(f"CPU time         : {t1_cpu  - t0_cpu :.1f} s")
    # print(f"RAM change       : {(mem1 - mem0)/1024**3:+.2f} GiB")
    # print(f"Peak GPU memory : {peak_gpu/1024**3:.2f} GiB")
    # for g in gpus:
    #     print(
    #         f" GPU {g.id}: util={g.load*100:.1f}%, mem={g.memoryUsed/1024**3:.1f}/{g.memoryTotal/1024**3:.1f} GiB"
    #     )

    if iter == 0:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # # for performance metrics
    # proc = psutil.Process()
    # # --- pre‐run snapshots ---
    # t0_wall = time.time()
    # t0_cpu = sum(proc.cpu_times()[:2])  # user + sys CPU s
    # mem0 = proc.memory_info().rss  # resident set size
    # torch.cuda.reset_peak_memory_stats()  # clear PyTorch peak

    # dtu_std_no_cycle = abaco_run(
    #     dataloader=dtu_train_dataloader,
    #     n_batches=dtu_batch_size,
    #     n_bios=dtu_bio_size,
    #     input_size=dtu_input_size,
    #     device=device,
    #     prior="Normal",
    #     w_contra=1000.0,
    #     kl_cycle=False,
    #     seed=None,
    #     smooth_annealing=True,
    #     pre_epochs=2500,
    #     post_epochs=5000,
    #     vae_post_lr=2e-4,
    #     adv_lr=1e-5,
    #     disc_lr=1e-5,
    # )

    # # --- post‐run snapshots ---
    # t1_wall = time.time()
    # t1_cpu = sum(proc.cpu_times()[:2])
    # mem1 = proc.memory_info().rss
    # peak_gpu = torch.cuda.max_memory_allocated()

    # # --- GPU‐side current status (optional) ---
    # gpus = GPUtil.getGPUs()

    # # --- print a neat summary ---
    # print("===== DTU-GE - Std no cycle Performance =====")
    # print(f"Wall‐clock time  : {t1_wall - t0_wall:.1f} s")
    # print(f"CPU time         : {t1_cpu  - t0_cpu :.1f} s")
    # print(f"RAM change       : {(mem1 - mem0)/1024**3:+.2f} GiB")
    # print(f"Peak GPU memory : {peak_gpu/1024**3:.2f} GiB")
    # for g in gpus:
    #     print(
    #         f" GPU {g.id}: util={g.load*100:.1f}%, mem={g.memoryUsed/1024**3:.1f}/{g.memoryTotal/1024**3:.1f} GiB"
    #     )

    if iter == 0:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Reconstruct data
    # ad_recon_data = abaco_recon(
    #     model=ad_count_std_no_cycle,
    #     device=device,
    #     data=ad_count_data,
    #     dataloader=ad_count_train_dataloader,
    #     sample_label=ad_count_sample_label,
    #     batch_label=ad_count_batch_label,
    #     bio_label=ad_count_bio_label,
    #     seed=None,
    #     monte_carlo=1,
    # )

    if iter == 0:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ibd_recon_data = abaco_recon(
    #     model=ibd_std_no_cycle,
    #     device=device,
    #     data=ibd_data,
    #     dataloader=ibd_train_dataloader,
    #     sample_label=ibd_sample_label,
    #     batch_label=ibd_batch_label,
    #     bio_label=ibd_bio_label,
    #     seed=None,
    #     monte_carlo=1,
    # )

    if iter == 0:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # dtu_recon_data = abaco_recon(
    #     model=dtu_std_no_cycle,
    #     device=device,
    #     data=dtu_data,
    #     dataloader=dtu_train_dataloader,
    #     sample_label=dtu_sample_label,
    #     batch_label=dtu_batch_label,
    #     bio_label=dtu_bio_label,
    #     seed=None,
    #     monte_carlo=1,
    # )

    # ad_recon_data.to_csv(
    #     f"performance_metrics/multi_runs/AD_count/std_no_recon_{iter}", index=False
    # )
    # ibd_recon_data.to_csv(
    #     f"performance_metrics/multi_runs/IBD/std_no_recon_contra_1000_2_{iter}",
    #     index=False,
    # )
    # dtu_recon_data.to_csv(
    #     f"performance_metrics/multi_runs/DTU-GE/std_no_recon_contra_1000_{iter}",
    #     index=False,
    # )

    # # LISI tables
    # clisi_ad = cLISI_full_rank(ad_recon_data, ad_count_bio_label)
    # ilisi_ad = iLISI_full_rank(ad_recon_data, ad_count_batch_label)

    # clisi_ad["iter"] = iter
    # ilisi_ad["iter"] = iter

    # performances_clisi_ad = pd.concat(
    #     [performances_clisi_ad, clisi_ad], axis=0
    # ).reset_index(drop=True)
    # performances_ilisi_ad = pd.concat(
    #     [performances_ilisi_ad, ilisi_ad], axis=0
    # ).reset_index(drop=True)

    # clisi_ibd = cLISI_full_rank(ibd_recon_data, ibd_bio_label)
    # ilisi_ibd = iLISI_full_rank(ibd_recon_data, ibd_batch_label)

    # clisi_ibd["iter"] = iter
    # ilisi_ibd["iter"] = iter

    # performances_clisi_ibd = pd.concat(
    #     [performances_clisi_ibd, clisi_ibd], axis=0
    # ).reset_index(drop=True)
    # performances_ilisi_ibd = pd.concat(
    #     [performances_ilisi_ibd, ilisi_ibd], axis=0
    # ).reset_index(drop=True)

    # clisi_dtu = cLISI_full_rank(dtu_recon_data, dtu_bio_label)
    # ilisi_dtu = iLISI_full_rank(dtu_recon_data, dtu_batch_label)

    # clisi_dtu["iter"] = iter
    # ilisi_dtu["iter"] = iter

    # performances_clisi_dtu = pd.concat(
    #     [performances_clisi_dtu, clisi_dtu], axis=0
    # ).reset_index(drop=True)
    # performances_ilisi_dtu = pd.concat(
    #     [performances_ilisi_dtu, ilisi_dtu], axis=0
    # ).reset_index(drop=True)

    # # PERMANOVA
    # bc, ait = PERMANOVA(
    #     ad_recon_data, ad_count_sample_label, ad_count_batch_label, ad_count_bio_label
    # )

    # bc = pd.DataFrame(
    #     {
    #         "R2": [bc["R2"]],
    #         "p-value": [bc["p-value"]],
    #         "test statistic": [bc["test statistic"]],
    #     }
    # )
    # ait = pd.DataFrame(
    #     {
    #         "R2": [ait["R2"]],
    #         "p-value": [ait["p-value"]],
    #         "test statistic": [ait["test statistic"]],
    #     }
    # )

    # permanova_bc_ad = pd.concat([permanova_bc_ad, bc], axis=0).reset_index(drop=True)

    # permanova_ait_ad = pd.concat([permanova_ait_ad, ait], axis=0).reset_index(drop=True)

    # bc, ait = PERMANOVA(
    #     ibd_recon_data, ibd_sample_label, ibd_batch_label, ibd_bio_label
    # )

    # bc = pd.DataFrame(
    #     {
    #         "R2": [bc["R2"]],
    #         "p-value": [bc["p-value"]],
    #         "test statistic": [bc["test statistic"]],
    #     }
    # )
    # ait = pd.DataFrame(
    #     {
    #         "R2": [ait["R2"]],
    #         "p-value": [ait["p-value"]],
    #         "test statistic": [ait["test statistic"]],
    #     }
    # )

    # permanova_bc_ibd = pd.concat([permanova_bc_ibd, bc], axis=0).reset_index(drop=True)

    # permanova_ait_ibd = pd.concat([permanova_ait_ibd, ait], axis=0).reset_index(
    #     drop=True
    # )

    # bc, ait = PERMANOVA(
    #     dtu_recon_data, dtu_sample_label, dtu_batch_label, dtu_bio_label
    # )

    # bc = pd.DataFrame(
    #     {
    #         "R2": [bc["R2"]],
    #         "p-value": [bc["p-value"]],
    #         "test statistic": [bc["test statistic"]],
    #     }
    # )
    # ait = pd.DataFrame(
    #     {
    #         "R2": [ait["R2"]],
    #         "p-value": [ait["p-value"]],
    #         "test statistic": [ait["test statistic"]],
    #     }
    # )

    # permanova_bc_dtu = pd.concat([permanova_bc_dtu, bc], axis=0).reset_index(drop=True)

    # permanova_ait_dtu = pd.concat([permanova_ait_dtu, ait], axis=0).reset_index(
    #     drop=True
    # )

    # # Performance metrics
    # norm_ad_recon_data = DataTransform(
    #     data=ad_recon_data,
    #     factors=[ad_count_sample_label, ad_count_batch_label, ad_count_bio_label],
    #     count=True,
    # )

    # performance_batch, _ = all_metrics(
    #     data=norm_ad_recon_data,
    #     bio_label=ad_count_bio_label,
    #     batch_label=ad_count_batch_label,
    # )
    # performances_batch_ad.append(performance_batch)

    # norm_ibd_recon_data = DataTransform(
    #     data=ibd_recon_data,
    #     factors=[ibd_sample_label, ibd_batch_label, ibd_bio_label],
    #     count=True,
    # )

    # performance_batch, _ = all_metrics(
    #     data=norm_ibd_recon_data, bio_label=ibd_bio_label, batch_label=ibd_batch_label
    # )
    # performances_batch_ibd.append(performance_batch)

    # norm_dtu_recon_data = DataTransform(
    #     data=dtu_recon_data,
    #     factors=[dtu_sample_label, dtu_batch_label, dtu_bio_label],
    #     count=True,
    # )

    # performance_batch, _ = all_metrics(
    #     data=norm_dtu_recon_data, bio_label=dtu_bio_label, batch_label=dtu_batch_label
    # )
    # performances_batch_dtu.append(performance_batch)

    # MANN-WHITNEY U TEST - AD COUNT
    # 1. Compute relative abundances for raw and corrected
    # count_raw = ad_count_data.select_dtypes(include="number")
    # rel_raw = count_raw.div(count_raw.sum(axis=1), axis=0)

    # count_corr = ad_recon_data.select_dtypes(include="number")
    # rel_corr = count_corr.div(count_corr.sum(axis=1), axis=0)

    # # 2. Identify top 10 OTUs by mean relative abundance in the raw data
    # top10 = rel_raw.mean().sort_values(ascending=False).head(10).index

    # # 3. Subset and reshape both to long form, adding a 'Dataset' column
    # df_raw = (
    #     rel_raw[top10]
    #     .reset_index()
    #     .melt(
    #         id_vars="index", value_vars=top10, var_name="OTU", value_name="RelAbundance"
    #     )
    # )
    # df_raw["Dataset"] = "Raw"

    # df_corr = (
    #     rel_corr[top10]
    #     .reset_index()
    #     .melt(
    #         id_vars="index", value_vars=top10, var_name="OTU", value_name="RelAbundance"
    #     )
    # )
    # df_corr["Dataset"] = "Corrected"

    # # 4. Concatenate
    # df_all = pd.concat([df_raw, df_corr], ignore_index=True)
    # df_all.rename(columns={"index": "Sample"}, inplace=True)

    # # 5. Compute Mann-whitney U test for top 10 taxonomic groups
    # stat_results = []

    # for otu in top10:
    #     raw_values = df_all[(df_all["OTU"] == otu) & (df_all["Dataset"] == "Raw")][
    #         "RelAbundance"
    #     ]
    #     corr_values = df_all[
    #         (df_all["OTU"] == otu) & (df_all["Dataset"] == "Corrected")
    #     ]["RelAbundance"]

    #     # Mann-whitney U test
    #     stat, p = mannwhitneyu(
    #         raw_values, corr_values, alternative="two-sided"
    #     )  # Ha two-sided: the distributions aren't equal / there are significant differences
    #     stat_results.append({"OTU": otu, "U-stat": stat, "p-value": p})

    # df_stats = pd.DataFrame(stat_results)
    # df_stats["adj p-value"] = multipletests(df_stats["p-value"], method="fdr_bh")[1]
    # df_stats["iter"] = iter

    # performances_mannwhit_ad = pd.concat(
    #     [performances_mannwhit_ad, df_stats], axis=0
    # ).reset_index(drop=True)

    # # MANN-WHITNEY U TEST - IBD
    # # 1. Compute relative abundances for raw and corrected
    # count_raw = ibd_data.select_dtypes(include="number")
    # rel_raw = count_raw.div(count_raw.sum(axis=1), axis=0)

    # count_corr = ibd_recon_data.select_dtypes(include="number")
    # rel_corr = count_corr.div(count_corr.sum(axis=1), axis=0)

    # # 2. Identify top 10 OTUs by mean relative abundance in the raw data
    # top10 = rel_raw.mean().sort_values(ascending=False).head(10).index

    # # 3. Subset and reshape both to long form, adding a 'Dataset' column
    # df_raw = (
    #     rel_raw[top10]
    #     .reset_index()
    #     .melt(
    #         id_vars="index", value_vars=top10, var_name="OTU", value_name="RelAbundance"
    #     )
    # )
    # df_raw["Dataset"] = "Raw"

    # df_corr = (
    #     rel_corr[top10]
    #     .reset_index()
    #     .melt(
    #         id_vars="index", value_vars=top10, var_name="OTU", value_name="RelAbundance"
    #     )
    # )
    # df_corr["Dataset"] = "Corrected"

    # # 4. Concatenate
    # df_all = pd.concat([df_raw, df_corr], ignore_index=True)
    # df_all.rename(columns={"index": "Sample"}, inplace=True)

    # # 5. Compute Mann-whitney U test for top 10 taxonomic groups
    # stat_results = []

    # for otu in top10:
    #     raw_values = df_all[(df_all["OTU"] == otu) & (df_all["Dataset"] == "Raw")][
    #         "RelAbundance"
    #     ]
    #     corr_values = df_all[
    #         (df_all["OTU"] == otu) & (df_all["Dataset"] == "Corrected")
    #     ]["RelAbundance"]

    #     # Mann-whitney U test
    #     stat, p = mannwhitneyu(
    #         raw_values, corr_values, alternative="two-sided"
    #     )  # Ha two-sided: the distributions aren't equal / there are significant differences
    #     stat_results.append({"OTU": otu, "U-stat": stat, "p-value": p})

    # df_stats = pd.DataFrame(stat_results)
    # df_stats["adj p-value"] = multipletests(df_stats["p-value"], method="fdr_bh")[1]
    # df_stats["iter"] = iter

    # performances_mannwhit_ibd = pd.concat(
    #     [performances_mannwhit_ibd, df_stats], axis=0
    # ).reset_index(drop=True)

    # # MANN-WHITNEY U TEST - DTU-GE
    # # 1. Compute relative abundances for raw and corrected
    # count_raw = dtu_data.select_dtypes(include="number")
    # rel_raw = count_raw.div(count_raw.sum(axis=1), axis=0)

    # count_corr = dtu_recon_data.select_dtypes(include="number")
    # rel_corr = count_corr.div(count_corr.sum(axis=1), axis=0)

    # # 2. Identify top 10 OTUs by mean relative abundance in the raw data
    # top10 = rel_raw.mean().sort_values(ascending=False).head(10).index

    # # 3. Subset and reshape both to long form, adding a 'Dataset' column
    # df_raw = (
    #     rel_raw[top10]
    #     .reset_index()
    #     .melt(
    #         id_vars="index", value_vars=top10, var_name="OTU", value_name="RelAbundance"
    #     )
    # )
    # df_raw["Dataset"] = "Raw"

    # df_corr = (
    #     rel_corr[top10]
    #     .reset_index()
    #     .melt(
    #         id_vars="index", value_vars=top10, var_name="OTU", value_name="RelAbundance"
    #     )
    # )
    # df_corr["Dataset"] = "Corrected"

    # # 4. Concatenate
    # df_all = pd.concat([df_raw, df_corr], ignore_index=True)
    # df_all.rename(columns={"index": "Sample"}, inplace=True)

    # # 5. Compute Mann-whitney U test for top 10 taxonomic groups
    # stat_results = []

    # for otu in top10:
    #     raw_values = df_all[(df_all["OTU"] == otu) & (df_all["Dataset"] == "Raw")][
    #         "RelAbundance"
    #     ]
    #     corr_values = df_all[
    #         (df_all["OTU"] == otu) & (df_all["Dataset"] == "Corrected")
    #     ]["RelAbundance"]

    #     # Mann-whitney U test
    #     stat, p = mannwhitneyu(
    #         raw_values, corr_values, alternative="two-sided"
    #     )  # Ha two-sided: the distributions aren't equal / there are significant differences
    #     stat_results.append({"OTU": otu, "U-stat": stat, "p-value": p})

    # df_stats = pd.DataFrame(stat_results)
    # df_stats["adj p-value"] = multipletests(df_stats["p-value"], method="fdr_bh")[1]
    # df_stats["iter"] = iter

    # performances_mannwhit_dtu = pd.concat(
    #     [performances_mannwhit_dtu, df_stats], axis=0
    # ).reset_index(drop=True)

# Save performance to file
# pd.DataFrame(performances_batch_ad).to_csv(
#     "performance_metrics/deterministic/AD_count/Std_no_cycle_batch.csv", index=False
# )
# pd.DataFrame(performances_clisi_ad).to_csv(
#     "performance_metrics/deterministic/AD_count/Std_no_cycle_clisi.csv", index=False
# )
# pd.DataFrame(performances_ilisi_ad).to_csv(
#     "performance_metrics/deterministic/AD_count/Std_no_cycle_ilisi.csv", index=False
# )
# pd.DataFrame(permanova_ait_ad).to_csv(
#     "performance_metrics/deterministic/AD_count/Std_no_cycle_perma_ait.csv", index=False
# )
# pd.DataFrame(permanova_bc_ad).to_csv(
#     "performance_metrics/deterministic/AD_count/Std_no_cycle_perma_bc.csv", index=False
# )
# pd.DataFrame(performances_mannwhit_ad).to_csv(
#     "performance_metrics/deterministic/AD_count/Std_no_cycle_mannwhit.csv", index=False
# )

# pd.DataFrame(performances_batch_ibd).to_csv(
#     "performance_metrics/deterministic/IBD/Std_no_cycle_batch.csv", index=False
# )
# pd.DataFrame(performances_clisi_ibd).to_csv(
#     "performance_metrics/deterministic/IBD/Std_no_cycle_clisi.csv", index=False
# )
# pd.DataFrame(performances_ilisi_ibd).to_csv(
#     "performance_metrics/deterministic/IBD/Std_no_cycle_ilisi.csv", index=False
# )
# pd.DataFrame(permanova_ait_ibd).to_csv(
#     "performance_metrics/deterministic/IBD/Std_no_cycle_perma_ait.csv", index=False
# )
# pd.DataFrame(permanova_bc_ibd).to_csv(
#     "performance_metrics/deterministic/IBD/Std_no_cycle_perma_bc.csv", index=False
# )
# pd.DataFrame(performances_mannwhit_ibd).to_csv(
#     "performance_metrics/deterministic/IBD/Std_no_cycle_mannwhit.csv", index=False
# )

# pd.DataFrame(performances_batch_dtu).to_csv(
#     "performance_metrics/deterministic/DTU-GE/Std_no_cycle_batch.csv", index=False
# )
# pd.DataFrame(performances_clisi_dtu).to_csv(
#     "performance_metrics/deterministic/DTU-GE/Std_no_cycle_clisi.csv", index=False
# )
# pd.DataFrame(performances_ilisi_dtu).to_csv(
#     "performance_metrics/deterministic/DTU-GE/Std_no_cycle_ilisi.csv", index=False
# )
# pd.DataFrame(permanova_ait_dtu).to_csv(
#     "performance_metrics/deterministic/DTU-GE/Std_no_cycle_perma_ait.csv", index=False
# )
# pd.DataFrame(permanova_bc_dtu).to_csv(
#     "performance_metrics/deterministic/DTU-GE/Std_no_cycle_perma_bc.csv", index=False
# )
# pd.DataFrame(performances_mannwhit_dtu).to_csv(
#     "performance_metrics/deterministic/DTU-GE/Std_no_cycle_mannwhit.csv", index=False
# )
