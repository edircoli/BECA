import pandas as pd
import numpy as np
from scipy.stats import gmean
from sklearn.preprocessing import StandardScaler


def DataPreprocess(path, factors = ['sample', 'batch', 'tissue']):
    df = pd.read_csv(path)
    df[factors] = df[factors].astype('category')

    return df

def DataTransform(data, factors = ['sample', 'batch', 'tissue'], transformation = "CLR"):

    if transformation == "CLR":
            # Select only OTUs columns and adding a small offset
            df_otu = data.select_dtypes(include='number') + 1e-9

            # Apply CLR transformation to numeric columns
            df_clr = np.log(df_otu.div(gmean(df_otu, axis=1), axis=0))

            # Combine CLR-transformed data with non-numeric columns
            df = pd.concat([data[factors], df_clr], axis=1)
    
    elif transformation == "Sqrt":
            # Select only OTUs columns
            df_otu = data.select_dtypes(include='number')

            #Apply Square-root transformation to numeric columns
            df_sqrt = np.sqrt(df_otu)

            #Standardize the squared-rooted data
            scaler = StandardScaler()
            df_stsqrt = scaler.fit_transform(df_sqrt)

            #Convert back to DataFrame
            df_stsqrt = pd.DataFrame(df_stsqrt, columns=df_sqrt.columns)

            #Combine data with non-numeric columns
            df = pd.concat([data[factors], df_stsqrt], axis = 1)

    elif transformation == "ILR":
        print("Not yet developed")
    
    elif transformation == "ALR":
        print("To be developed")
    
    else:
        raise(ValueError(f"Not a valid transformation: {transformation}"))
    
    return df

def DataReverseTransform(data, original_data, factors = ["sample", "batch", "tissue"], transformation = "CLR"):
    
    if transformation == "CLR":
        df_otu = data.select_dtypes(include = "number")
        df_otu_original = original_data.select_dtypes(include='number') + 1e-9

        df_inv = round(np.exp(df_otu) * gmean(df_otu_original, axis=1)[:, None], 3)

        df = pd.concat([data[factors], df_inv], axis = 1)


    return df