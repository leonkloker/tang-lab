import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
import sys
import umap

import data

# Read in the base populations
antigen = "cd63" #cd63 avidin cd203c_dMFI*
file = './data/bat_ifc.csv'
x, y_avidin, y_cd203c, y_cd63, patient_ids = data.get_data(file)

x = data.get_statistical_moment_features(x, ["mean"])
print(patient_ids)

# UMAP
reducer = umap.UMAP(n_components=3, metric='euclidean', random_state=42, n_neighbors=20, min_dist=0.5)
umap_data = reducer.fit_transform(x)

start = 0

# plot 3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(umap_data[start:, 0], umap_data[start:, 1], umap_data[start:, 2], cmap='plasma', s=25, c=y_cd203c[start:], alpha=1.0)
ax.set_title('3D UMAP')
colorbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
colorbar.set_label('CD63 activation', rotation=270, labelpad=20)
plt.show()
#plt.savefig("figures/umap_pdf_{}.png".format(antigen))
