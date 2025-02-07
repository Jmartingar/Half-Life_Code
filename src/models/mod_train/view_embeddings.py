from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import SpectralEmbedding
from sklearn.decomposition import PCA
import umap.umap_ as umap

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import pandas as pd
import sys

def applyPCA(embedding):
    return PCA(n_components=2, random_state=42).fit_transform(embedding.values)

def applyTSNE(embedding):
    return TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=3, random_state=42).fit_transform(embedding.values)

def applyIsomap(embedding):
    return Isomap().fit_transform(embedding.values)

def applyMDE(embedding):
    return MDS(n_components=2, random_state=42).fit_transform(embedding.values)

def applyLLE(embedding):
    return LocallyLinearEmbedding(n_components=2, random_state=42, eigen_solver='dense').fit_transform(embedding.values)

def applySpectral(embedding):
    return SpectralEmbedding(n_components=2).fit_transform(embedding.values)

def applyUMAP(embedding):
    reducer = umap.UMAP(random_state=42)
    return reducer.fit_transform(embedding.values)

def apply_all_embedding(data_values, responses, name_response):
        
    tsne_embedding = applyTSNE(data_values)
    df_tsne = pd.DataFrame(data=tsne_embedding, columns=["p1", "p2"])
    df_tsne[name_response] = responses

    isomap_embedding = applyIsomap(data_values)
    df_isomap = pd.DataFrame(data=isomap_embedding, columns=["p1", "p2"])
    df_isomap[name_response] = responses
    
    mde_embedding = applyMDE(data_values)
    df_mde = pd.DataFrame(data=mde_embedding, columns=["p1", "p2"])
    df_mde[name_response] = responses
    
    lle_embedding = applyLLE(data_values)
    df_lle = pd.DataFrame(data=lle_embedding, columns=["p1", "p2"])
    df_lle[name_response] = responses
    
    spectral_embedding = applySpectral(data_values)
    df_spectral = pd.DataFrame(data=spectral_embedding, columns=["p1", "p2"])
    df_spectral[name_response] = responses
    
    umap_embedding = applyUMAP(data_values)
    df_umap = pd.DataFrame(data=umap_embedding, columns=["p1", "p2"])
    df_umap[name_response] = responses
    
    return df_tsne, df_isomap, df_mde, df_lle, df_spectral, df_umap

def create_figures(df_tsne, df_isomap, df_mde, df_lle, df_spectral, df_umap, hue, name_export):
    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(2, 3, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    sns.scatterplot(x="p1", y="p2", hue=hue, data=df_tsne, ax=ax1)
    ax1.set_title('T-SNE')

    ax2 = fig.add_subplot(gs[0, 1])
    sns.scatterplot(x="p1", y="p2", hue=hue, data=df_isomap, ax=ax2)
    ax2.set_title('Isomap')

    ax3 = fig.add_subplot(gs[0, 2])
    sns.scatterplot(x="p1", y="p2", hue=hue, data=df_mde, ax=ax3)
    ax3.set_title('MDE')

    ax4 = fig.add_subplot(gs[1, 0])
    sns.scatterplot(x="p1", y="p2", hue=hue, data=df_lle, ax=ax4)
    ax4.set_title('LLE')

    ax5 = fig.add_subplot(gs[1, 1])
    sns.scatterplot(x="p1", y="p2", hue=hue, data=df_spectral, ax=ax5)
    ax5.set_title('Spectral')

    ax6 = fig.add_subplot(gs[1, 2])
    sns.scatterplot(x="p1", y="p2", hue=hue, data=df_umap, ax=ax6)
    ax6.set_title('UMAP')

    plt.tight_layout()
    plt.savefig(name_export, dpi=300)

df_data = pd.read_csv(sys.argv[1])
name_response = sys.argv[2]
name_export = sys.argv[3]

response = df_data[name_response]
df_values = df_data.drop(columns=[name_response])

df_tsne1, df_isomap1, df_mde1, df_lle1, df_spectral1, df_umap1 = apply_all_embedding(
    df_values, 
    response,
    name_response)

create_figures(
    df_tsne1, 
    df_isomap1, 
    df_mde1, 
    df_lle1, 
    df_spectral1, 
    df_umap1, 
    name_response, 
    name_export)

