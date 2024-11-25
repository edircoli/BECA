import pandas as pd
import numpy as np
from scipy.stats import gmean
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.validators.scatter.marker import SymbolValidator
from clustergrammer2 import net, Network, CGM2

def DataPreprocess(path, preprocess = True, factors = ['sample', 'batch', 'tissue'], transformation = "CLR"):
    df = pd.read_csv(path)
    df[factors] = df[factors].astype('category')

    if (preprocess):
        if transformation == "CLR":
            # Select only OTUs columns and adding a small offset
            df_otu = df.select_dtypes(include='number') + 0.01

            # Apply CLR transformation to numeric columns
            df_clr = np.log(df_otu.div(gmean(df_otu, axis=1), axis=0))

            # Combine CLR-transformed data with non-numeric columns
            df = pd.concat([df[factors], df_clr], axis=1)
        
        elif transformation == "Sqrt":
            # Select only OTUs columns
            df_otu = df.select_dtypes(include='number')

            #Apply Square-root transformation to numeric columns
            df_sqrt = np.sqrt(df_otu)

            #Standardize the squared-rooted data
            scaler = StandardScaler()
            df_stsqrt = scaler.fit_transform(df_sqrt)

            #Convert back to DataFrame
            df_stsqrt = pd.DataFrame(df_stsqrt, columns=df_sqrt.columns)

            #Combine data with non-numeric columns
            df = pd.concat([df[factors], df_stsqrt], axis = 1)

    return df

def plotPCA(data, sample_label = "sample", batch_label = "batch", experiment_label = "tissue"):
    #Realize the PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data.select_dtypes(include='number'))
    df_pca = pd.DataFrame(data=principal_components, columns = ["PC1", "PC2"])

    #Add the labels from batch and other important information
    df_pca[[sample_label, batch_label, experiment_label]] = data[[sample_label, batch_label, experiment_label]]
    
    #Extracting available symbols to be used per experiment
    raw_symbols = []
    for i in range(2, len(SymbolValidator().values), 12):
        raw_symbols.append(SymbolValidator().values[i])

    #Defining a set of colors to be used for batches
    raw_colors = ["blue","red","green","orange","purple"]

    #Adding symbol to corresponding experiment
    df_pca["marker"] = None
    for n, exp in enumerate(df_pca[experiment_label].unique()):
        df_pca.loc[df_pca[experiment_label] == exp, "marker"] = raw_symbols[n]

    #Adding color to corresponding batch
    df_pca["color"] = None
    for n, batch in enumerate(df_pca[batch_label].unique()):
        df_pca.loc[df_pca[batch_label] == batch, "color"] = raw_colors[n]

    #Creating the plotly figure
    fig = go.Figure()

    #Creating a for loop to alocate PCA data points per batch
    for batch in df_pca[batch_label].unique():
        #Creating a for loop to alocate data points per experiment in the current batch
        for exp in df_pca[experiment_label].unique():
            #Ploting the points corresponding to the current batch and tissue
            fig.add_trace(go.Scatter(x = df_pca[(df_pca[batch_label] == batch) & (df_pca[experiment_label] == exp)]["PC1"],
                                     y = df_pca[(df_pca[batch_label] == batch) & (df_pca[experiment_label] == exp)]["PC2"],
                                     marker = dict(color = df_pca[(df_pca[batch_label] == batch) & (df_pca[experiment_label] == exp)]["color"],
                                                   size = 8),
                                     marker_symbol = df_pca[(df_pca[batch_label] == batch) & (df_pca[experiment_label] == exp)]["marker"],
                                     legendgroup = batch,
                                     legendgrouptitle_text = "Batch {}".format(batch),
                                     name = exp,
                                     mode = "markers"
                                     ))
    
    return fig.show()

def plotOTUBox(data, batch_label = "batch"):
    #Extract OTUs columns names
    otu_cols = [col for col in data.columns if col.startswith("OTU")]
    batch_labels = data[batch_label].unique()
    batch_len = len(batch_labels)

    #Converting DataFrame from wide to long
    df_long = pd.melt(data, id_vars = [batch_label], value_vars = otu_cols, var_name = "OTU", value_name = "value")

    #Defining a set of colors to be used for batches
    raw_colors = ["blue","red","green","orange","purple"]

    #Adding color to corresponding batch
    batch_colors = []
    for i in range(batch_len):
        batch_colors.append(raw_colors[i])
    
    fig = go.Figure()

    # Add traces for each OTU
    for otu in otu_cols:
        for i, batch in enumerate(batch_labels):
            fig.add_trace(go.Box(
                x=df_long[(df_long['OTU'] == otu) & (df_long[batch_label] == batch)][batch_label], 
                y=df_long[(df_long['OTU'] == otu) & (df_long[batch_label] == batch)]['value'],
                marker=dict(color=batch_colors[i]), # Apply color to the batch boxplot
                name=f"Batch {batch}, {otu}",  # Label each trace by the OTU
                visible=False  # Set initially to invisible
            ))

    # First OTU visible by default
    for i in range(batch_len):
        fig.data[i].visible = True

    # Add dropdown to select which OTU to display
    fig.update_layout(
        xaxis_title = "Batch",
        updatemenus=[dict(
            buttons=[
                *[
                    dict(
                        args=[{"visible": [(i >= batch_len*idx) & (i <= batch_len*idx + (batch_len-1)) for i in range(len(fig.data))]}],  # Toggle visibility
                        label=otu,
                        method="update"
                    ) for idx, otu in enumerate(otu_cols)
                ]
            ],
            direction="down",
            showactive=True,
            xanchor="left",
            y=1.15,
            yanchor="top"
        )]
    )

    return fig.show()

def plotRLE(data, sample_label = "sample", batch_label = "batch", experiment_label = "tissue"):

    # Extract OTUs column names
    otu_cols = [col for col in data.columns if col.startswith("OTU")]

    # Converting DataFrame from wide to long
    df_long = pd.melt(data, id_vars = [sample_label, batch_label, experiment_label], value_vars = otu_cols, var_name = "OTU", value_name = "value")

    # Calculating the medians of each OTU within each experiment
    df_long["medians"] = None
    for OTU in df_long["OTU"].unique():
        for exp in df_long[experiment_label].unique():
            med = np.median(df_long[(df_long["OTU"] == OTU) & (df_long[experiment_label] == exp)]["value"])
            df_long.loc[(df_long["OTU"] == OTU) & (df_long[experiment_label] == exp), "medians"] = med

    # Incorporating the difference between OTU value in each sample and the median across all samples from the same tissue
    df_long["RLE"] = df_long["value"] - df_long["medians"]

    #Defining a set of colors to be used for batches
    raw_colors = ["blue","red","green","orange","purple"]

    #Adding color to corresponding batch
    df_long["color"] = None
    for n, batch in enumerate(df_long[batch_label].unique()):
        df_long.loc[df_long[batch_label] == batch, "color"] = raw_colors[n]

    # Generate RLE plots for each experiment
    fig = go.Figure()

    # Add traces for each experiment
    for exp in df_long[experiment_label].unique():
        # Add traces for each batch
        for batch in df_long[batch_label].unique():

            fig.add_trace(go.Box(
                x = df_long[(df_long[experiment_label] == exp) & (df_long[batch_label] == batch)][sample_label], 
                y = df_long[(df_long[experiment_label] == exp) & (df_long[batch_label] == batch)]['RLE'],
                marker_color = df_long[(df_long[experiment_label] == exp) & (df_long[batch_label] == batch)]['color'].iloc[0],
                name="Batch {}".format(batch),  # Label each trace by the batch
                visible=False  # Set initially to invisible
            ))

    # First experiment's traces visible by default
    for i in range(len(df_long[batch_label].unique())):
        fig.data[i].visible = True

    # Add dropdown to select which experiment to display
    fig.update_layout(
        updatemenus=[dict(
            buttons=[
                *[
                    dict(
                        args=[{"visible": [i // len(df_long[batch_label].unique()) == idx for i in range(len(fig.data))]}],  # Toggle visibility
                        label=exp,
                        method="update"
                    ) for idx, exp in enumerate(df_long[experiment_label].unique())
                ]
            ],
            direction="down",
            showactive=True,
            xanchor="left",
            y=1.15,
            yanchor="top"
        )]
    )

    # Add horizontal dashed red line at y = 0 as a reference point
    fig.add_shape(
        type="line",
        x0=0, x1=1,  # Extend the line across the x-axis
        y0=0, y1=0,  # Line positioned at y = 0
        xref="paper", yref="y",  # "paper" allows the line to span the entire plot width
        line=dict(color="red", width=2, dash="dash")  # Dashed red line
    )

    return fig.show()

def plotClusterHeatMap(data, batch_label = "batch", experiment_label = "tissue", sample_label = "sample"):
    #Extracts numerical and categorical data of interest
    data_num = data.select_dtypes(include = "number")
    data_num.index = [str(i) for i in data[sample_label]]
    data_cat = data[[batch_label, experiment_label]]
    data_cat.index = [str(i) for i in data[sample_label]]

    #First scaling process - Ensures every observation is scaled according to OTUs
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_num)
    scaled_data = pd.DataFrame(scaled_data, columns=data_num.columns, index=data_num.index)

    #Second scaling process - Ensures every observation is scaled according to sample
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(scaled_data.T)
    scaled_data = pd.DataFrame(scaled_data, index=[str(i) for i in data_num.columns], columns=[str(i) for i in data_num.index])

    #Create Clustergrammer2 plot
    n2 = Network(CGM2)
    n2.load_df(scaled_data, meta_col = data_cat)
    n2.cluster()
    return n2.widget()