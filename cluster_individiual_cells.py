from sklearn.cluster import KMeans
import data
from scipy.stats import pearsonr
import sys
import numpy as np
import torch
import autoencoder
import umap

populations, y_avidin, y_cd203c, y_cd63 = data.load_data('./data/25_populations_antiIge.pickle')
model = autoencoder.Autoencoder.load_from_checkpoint("./logs/AE_17_8_4/version_8/checkpoints/epoch=50-step=102000.ckpt")
model.eval()
model.to('cpu')

cluster_ratios = []
for k in range(len(populations)):
    # Calculate clustering with two clusters
    kmeans = KMeans(n_clusters=2)

    #latent_population = model.encode(torch.tensor(populations[k], dtype=torch.float32)).detach().numpy()
    #latent_population = np.array(latent_population)
    
    import matplotlib.pyplot as plt

    # Reduce dimensions using UMAP
    #reducer = umap.UMAP(n_components=2)
    #umap_latent_population = reducer.fit_transform(latent_population)

    clusters = kmeans.fit_predict(populations[k])

    # Count the number of cells in each cluster
    cluster_counts = {}
    for cluster in set(clusters):
        cluster_counts[cluster] = sum(clusters == cluster)

    # Print the counts
    cluster_ratios.append(cluster_counts[0] / (cluster_counts[0] + cluster_counts[1]))

print([*zip(cluster_ratios, y_cd63)])
print(pearsonr(cluster_ratios, y_cd203c)[0])
print(pearsonr(cluster_ratios, y_cd63)[0])


