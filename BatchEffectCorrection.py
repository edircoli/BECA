import pandas as pd
import numpy as np
from combat.pycombat import pycombat 
import statsmodels.api as sm
from BatchEffectPlots import DataPreprocess, plotPCA, plotOTUBox, plotRLE, plotClusterHeatMap
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import VarianceThreshold

path = "data/sponge_dataset.csv"
data = DataPreprocess(path)


def correctCombat(data, sample_label = "sample", batch_label = "batch", experiment_label = "tissue"):
    num_data = data.select_dtypes(include = "number")
    batch_data = [batch for batch in data[batch_label]]
    cov_data = [exp for exp in data[experiment_label]]

    corrected_data = pycombat(num_data.T, batch_data, cov_data)
    data_combat = pd.concat([data[[sample_label, batch_label, experiment_label]], corrected_data.T], axis = 1)
    
    return data_combat

def correctLimma_rBE(data, sample_label = "sample", batch_label = "batch", covariates_labels = None):
    
    #Extract numeric variables from data
    num_data = data.select_dtypes(include="number")
    
    # Convert batch labels to a one-hot encoded DataFrame for regression
    batch = pd.get_dummies(data[batch_label], drop_first = True)
    batch.columns = [batch_label]

    # Combine batch and covariates
    if covariates_labels is not None:
        covariates = pd.get_dummies(data[covariates_labels], drop_first = True)
        covariates.columns = [covariates_labels]
        design_matrix = pd.concat([batch, covariates], axis=1)

        # Create a design matrix with only batch effects
        design_matrix_batch = pd.concat([batch, pd.DataFrame(np.zeros_like(covariates), columns=covariates.columns)], axis=1)
        design_matrix_batch = sm.add_constant(design_matrix_batch, has_constant="add")
        design_matrix_batch = design_matrix_batch.astype(int)

    else:
        design_matrix = batch
        design_matrix_batch = design_matrix

    # Ensure an intercept is added to the model
    design_matrix = sm.add_constant(design_matrix, has_constant="add")
    design_matrix = design_matrix.astype(int)

    # Initialize a DataFrame to store batch-corrected values
    corrected_data = pd.DataFrame(index=num_data.index, columns=num_data.columns)

    # Regress out batch effect for each feature
    for feature in num_data.columns:
        model = sm.OLS(num_data[feature], design_matrix).fit()

        # Calculate the effect of batch (ignore the biological effect)
        batch_effect = model.predict(design_matrix_batch)
        
        # Subtract only the batch effect from the data
        corrected_data[feature] = num_data[feature] - batch_effect

    data_limma = pd.concat([data[[sample_label, batch_label]], data[covariates_labels]], axis=1)
    data_limma = pd.concat([data_limma, corrected_data], axis=1)

    return data_limma

def correctSVD(data, sample_label = "sample", batch_label = "batch", experiment_label = "tissue"):

    #Extract OTUs variables and batch labels
    df_otu = data.select_dtypes(include = "number")

    #Calculate standard deviation and mean for each OTU
    otu_sd = df_otu.std(axis=0)
    otu_mu = df_otu.mean(axis=0)

    #Standardize the data
    scaler = StandardScaler(with_mean = True, with_std = True)
    df_scaled = scaler.fit_transform(df_otu)

    #Compute square matrix and perform SVD
    sq_matrix = np.dot(df_scaled.T, df_scaled)
    U, S, Vt = np.linalg.svd(sq_matrix)

    a = U[:, 0]

    t = np.dot(df_scaled, a) / np.sqrt(np.dot(a.T, a))
    c = np.dot(df_scaled.T, t) / np.dot(t.T, t)

    #Deflate the component from data
    svd_deflated_matrix = df_scaled - np.outer(t,c)

    #Add mean and std to return to original state
    corrected_data = np.empty_like(svd_deflated_matrix)
    for i in range(svd_deflated_matrix.shape[1]):
        corrected_data[:, i] = svd_deflated_matrix[:, i] * otu_sd[i] + otu_mu[i]

    #Convert back to DataFrame
    df_corrected = pd.DataFrame(corrected_data, columns = df_otu.columns, index=df_otu.index)

    #Join factor columns
    df_svd = pd.concat([data[[sample_label, experiment_label, batch_label]], df_corrected], axis = 1)

    return df_svd

def correctPLSDA_batch(data,
                       method = "batch",
                       near_zero_var = True,
                       sample_label = "sample",
                       batch_label = "batch", 
                       treatment_label = "tissue"):

    x = data.select_dtypes(include = "number")
    y_trt = data[treatment_label].values
    y_bat = data[batch_label].values

    if method == "batch":

        #One-hot encoding for batch variable
        y_bat = pd.factorize(y_bat)[0]
        y_bat_mat = np.zeros((x.shape[0], len(np.unique(y_bat))), dtype = int)
        y_bat_mat[np.arange(x.shape[0]), y_bat] = 1

        #One-hot encoding for treatment variable
        y_trt = pd.factorize(y_trt)[0]
        y_trt_mat = np.zeros((x.shape[0], len(np.unique(y_trt))), dtype = int)
        y_trt_mat[np.arange(x.shape[0]), y_trt] = 1

        #Handel near-zero variance
        if near_zero_var:
            selector = VarianceThreshold(threshold = 0.01)
            x = selector.fit_transform(x)

        #Scaling OTU data
        scaler = StandardScaler(with_mean=True, with_std=True)
        x_scaled = scaler.fit_transform(x)

        #Fit PLSDA on treatment
        pls_trt = PLSRegression(n_components = len(np.unique(y_trt)))
        pls_trt.fit(x_scaled, y_trt_mat)

        #Compute latent components
        t_trt = pls_trt.x_scores_
        p_trt = pls_trt.x_loadings_

        #Remove treatment effect
        x_no_trt = x_scaled - np.dot(t_trt, p_trt.T)

        #Weight data based on treatment and batch
        class_weights = np.ones(x.shape[0])
        for t in np.unique(y_trt):
            for b in np.unique(y_bat):
                mask = (y_trt == t) & (y_bat == b)
                weight = np.sum(mask) / len(mask)
                class_weights[mask] = weight

        x_scaled *= class_weights[:, None]
        x_no_trt *= class_weights[:, None]

        #Fit PLSDA on batch
        pls_bat = PLSRegression(n_components = len(np.unique(y_bat)))
        pls_bat.fit(x_no_trt, y_bat_mat)

        #Deflate batch effects
        bat_loadings = pls_bat.x_weights_
        x_temp = x_scaled
        for h in range(len(np.unique(y_bat))):
            a_bat = bat_loadings[:, h]
            t_bat = np.dot(x_temp, a_bat)
            x_temp -= np.outer(t_bat, a_bat)

        x_no_bat = x_temp

        #Reverse scaling
        x_no_bat_final = scaler.inverse_transform(x_no_bat)
    
    corrected_data = pd.DataFrame(x_no_bat_final,
             index = data.select_dtypes(include = "number").index, 
             columns = data.select_dtypes(include = "number").columns)

    corrected_data = pd.concat([data[[sample_label, treatment_label, batch_label]], corrected_data], axis = 1)

    return corrected_data
    
    