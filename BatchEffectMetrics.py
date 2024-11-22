import pandas as pd
import numpy as np
from scipy.stats import chi2
from sklearn.neighbors import NearestNeighbors

def kBET(data, batch_label = "batch"):

    data_otus = data.select_dtypes(include = "number")
    data_batch = data[batch_label]

    n = len(data_batch)
    k = int(np.sqrt(data_otus.shape[0])) #k-nearest neighbors
    dof = len(data_batch.unique()) - 1 #Degrees of freedom

    knn = NearestNeighbors(n_neighbors = k, metric = "euclidean")
    knn.fit(data_otus)

    _, indx = knn.kneighbors(data_otus)

    p_values = []
    for j, neighbor in enumerate(indx):
        gamma_j = []
        for i in data_batch.unique():
            mu_ij = round(sum(data_batch == i) / n * k + 0.001) #Expected number of samples from batch i in neighbor j
            n_ij = 0 #Actual number of samples from batch i in neighbor j
            
            for s in neighbor:
                if data_batch.iloc[s] == i:
                    n_ij += 1 #Identifies batch label on neighborhood

            gamma_ij = (n_ij - mu_ij)**2 / mu_ij 
            gamma_j.append(gamma_ij)

        sum_gamma_j = sum(gamma_j) #Follows Chi-squared distribution

        p_j = 1 - chi2.cdf(sum_gamma_j, dof) #Pearson Chi-squared test
        p_values.append(p_j)

    return sum(1 for p in p_values if p > 0.05) / n

def iLISI(data, batch_label = "batch"):

    data_otus = data.select_dtypes(include = "number")
    data_batch = data[batch_label]

    n = len(data_batch)
    k = int(np.sqrt(data_otus.shape[0])) #k-nearest neighbors

    knn = NearestNeighbors(n_neighbors = k, metric = "euclidean")
    knn.fit(data_otus)

    _, indx = knn.kneighbors(data_otus)

    ISI_j = []
    for neighborhood in indx:
        p_j = []
        for i in data_batch.unique():
            p_ij = 0
            for s in neighborhood:
                if data_batch.iloc[s] == i:
                    p_ij += 1 #Identifies batch label on neighborhood
            p_ij = p_ij / len(neighborhood) #Proportion of batch label in neighborhood
            p_j.append(p_ij**2)

        ISI_j.append(1 / sum(p_j)) #Inverse Simpson's index for neighborhood

    iLISI = 1/len(indx) * sum(ISI_j)
    
    return iLISI

def cLISI(data, cell_label = "tissue"):

    data_otus = data.select_dtypes(include = "number")
    data_cell = data[cell_label]

    n = len(data_cell)
    k = int(np.sqrt(data_otus.shape[0])) #k-nearest neighbors

    knn = NearestNeighbors(n_neighbors = k, metric = "euclidean")
    knn.fit(data_otus)

    _, indx = knn.kneighbors(data_otus)

    ISI_j = []
    for neighborhood in indx:
        p_j = []
        for i in data_cell.unique():
            p_ij = 0
            for s in neighborhood:
                if data_cell.iloc[s] == i:
                    p_ij += 1 #Identifies batch label on neighborhood
            p_ij = p_ij / len(neighborhood) #Proportion of batch label in neighborhood
            p_j.append(p_ij**2)

        ISI_j.append(1 / sum(p_j)) #Inverse Simpson's index for neighborhood

    cLISI = 1/len(indx) * sum(ISI_j)
    
    return cLISI