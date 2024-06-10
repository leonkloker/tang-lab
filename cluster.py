from sklearn.cluster import KMeans
import data
from scipy.stats import pearsonr

populations, y_avidin, y_cd203c, y_cd63 = data.load_data('./data/25_populations_antiIge.pickle')

cluster_ratios = []
for k in range(len(populations)):

    # Calculate clustering with two clusters
    kmeans = KMeans(n_clusters=2)
    clusters = kmeans.fit_predict(populations[0])

    # Count the number of cells in each cluster
    cluster_counts = {}
    for cluster in set(clusters):
        cluster_counts[cluster] = sum(clusters == cluster)

    # Print the counts
    cluster_ratios.append(cluster_counts[0] / (cluster_counts[0] + cluster_counts[1]))

print(pearsonr(cluster_ratios, y_avidin)[0])
print(pearsonr(cluster_ratios, y_cd203c)[0])
print(pearsonr(cluster_ratios, y_cd63)[0])


