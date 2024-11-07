import pandas as pd
import numpy as np
from scipy.stats import gmean
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.validators.scatter.marker import SymbolValidator

def DataPreprocess(path, preprocess = True, factors = ['sample', 'batch', 'tissue']):
    df = pd.read_csv(path)
    if (preprocess):
        #Selecting numeric variables and applying the Center Log Ratio Transformation
        df[factors] = df[factors].astype('category')

        # Select only OTUs columns and adding a small offset
        df_otu = df.select_dtypes(include='number') + 0.01

        # Apply CLR transformation to numeric columns
        df_clr = np.log(df_otu.div(gmean(df_otu, axis=1), axis=0))

        # Combine CLR-transformed data with non-numeric columns
        df = pd.concat([df[['sample', 'batch', 'tissue']], df_clr], axis=1)

    return df

def PCA_plot(data, sample_label = "sample", batch_label = "batch", experiment_label = "tissue"):
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

def OTUBoxPlot(data, batch_label = "batch"):
    #Extract OTUs columns names
    otu_cols = [col for col in data.columns if col.startswith("OTU")]

    #Converting DataFrame from wide to long
    df_long = pd.melt(data, id_vars = [batch_label], value_vars = otu_cols, var_name = "OTU", value_name = "value")

    fig = go.Figure()

    # Add traces for each OTU
    for otu in otu_cols:
        fig.add_trace(go.Box(
            x=df_long[df_long['OTU'] == otu][batch_label], 
            y=df_long[df_long['OTU'] == otu]['value'],
            name=otu,  # Label each trace by the OTU
            visible=False  # Set initially to invisible
        ))

    # First OTU visible by default
    fig.data[0].visible = True

    # Add dropdown to select which OTU to display
    fig.update_layout(
        updatemenus=[dict(
            buttons=[
                *[
                    dict(
                        args=[{"visible": [i == idx for i in range(len(fig.data))]}],  # Toggle visibility
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

path = "data/sponge_dataset.csv"
data = DataPreprocess(path)
#PCA_plot(data)
OTUBoxPlot(data)

