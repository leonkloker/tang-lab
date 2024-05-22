import matplotlib.pyplot as plt
import numpy as np
import pickle

viridis = plt.get_cmap('viridis')
colors = [viridis(i/5) for i in range(0,10,2)]

# Load the data
with open('results/pdf_sampling_points_ablation_cd63.pickle', 'rb') as f:
    data_cd63 = pickle.load(f)

with open('results/pdf_sampling_points_ablation_cd63.pickle', 'rb') as f:
    data_avidin = pickle.load(f)

with open('results/pdf_sampling_points_ablation_cd63.pickle', 'rb') as f:
    data_cd203c = pickle.load(f)

mae_ridge_cd63 = data_cd63["mae_ridge"]
r_ridge_cd63 = data_cd63["r_ridge"]
mae_ridge_avidin = data_avidin["mae_ridge"]
r_ridge_avidin = data_avidin["r_ridge"]
mae_ridge_cd203c = [x / 14. for x in data_cd203c["mae_ridge"]]
r_ridge_cd203c = data_cd203c["r_ridge"]

# Plot the data
x = np.arange(1,31)

# Add some text for labels, title and custom x-axis tick labels, etc.
plt.figure()
plt.plot(x, mae_ridge_cd63, label='CD63', color=colors[0])
plt.plot(x, mae_ridge_avidin, label='Avidin', color=colors[1])
plt.plot(x, mae_ridge_cd203c, label='CD203c', color=colors[2])
plt.ylabel('MAE', fontsize=12)
plt.xlabel('Sampling points', fontsize=12)
plt.legend(prop={'size': 12})
plt.grid(True, linestyle=':', linewidth=0.7, which='both', axis='y')
plt.tight_layout()
plt.show()
#plt.savefig("figures/pdf_two_frequency_ablation_all_markers_r.png", dpi=400)

plt.figure()
plt.plot(x, r_ridge_cd63, label='CD63', color=colors[0])
plt.plot(x, r_ridge_avidin, label='Avidin', color=colors[1])
plt.plot(x, r_ridge_cd203c, label='CD203c', color=colors[2])
plt.ylabel('Pearson coefficient', fontsize=12)
plt.xlabel('Sampling points', fontsize=12)
plt.legend(prop={'size': 12})
plt.grid(True, linestyle=':', linewidth=0.7, which='both', axis='y')
plt.tight_layout()
plt.show()
#plt.savefig("figures/pdf_two_frequency_ablation_all_markers_mae.png", dpi=400)

