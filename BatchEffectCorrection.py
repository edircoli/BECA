import pandas as pd
import numpy as np
from combat.pycombat import pycombat 
import statsmodels.api as sm
from BatchEffectPlots import DataPreprocess, plotPCA, plotOTUBox, plotRLE, plotClusterHeatMap

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
