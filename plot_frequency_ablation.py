import matplotlib.pyplot as plt
import numpy as np
import pickle

plt.rcParams['font.family'] = 'Arial'

viridis = plt.get_cmap('viridis')
colors = [viridis(i/5) for i in range(0,10,2)]

# Load the data
with open('results/pdf_frequency_ablation_cd63.pickle', 'rb') as f:
    data_cd63 = pickle.load(f)

with open('results/pdf_frequency_ablation_avidin.pickle', 'rb') as f:
    data_avidin = pickle.load(f)

with open('results/pdf_frequency_ablation_cd203c_dMFI*.pickle', 'rb') as f:
    data_cd203c = pickle.load(f)

mae_ridge_cd63 = data_cd63["mae_ridge"][:-1] - data_cd63["mae_ridge"][-1]
r_ridge_cd63 = data_cd63["r_ridge"][:-1] - data_cd63["r_ridge"][-1]
mae_ridge_avidin = data_avidin["mae_ridge"][:-1] - data_avidin["mae_ridge"][-1]
r_ridge_avidin = data_avidin["r_ridge"][:-1] - data_avidin["r_ridge"][-1]
mae_ridge_cd203c = (data_cd203c["mae_ridge"][:-1] - data_cd203c["mae_ridge"][-1])# / 14.07
r_ridge_cd203c = data_cd203c["r_ridge"][:-1] - data_cd203c["r_ridge"][-1]

# Plot the data
width = 0.25  # the width of the bars
x = np.arange(1,16)

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, r_ridge_avidin, width, label='Avidin', color=colors[0])
rects2 = ax.bar(x, r_ridge_cd63, width, label='CD63', color=colors[1])
rects3 = ax.bar(x + width, r_ridge_cd203c, width, label='CD203c', color=colors[2])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Change in r', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(['f{},f{}'.format(f,g) for f in range(1,7) for g in range(f+1,7)])
ax.set_xlabel('Removed frequencies', fontsize=12)
ax.legend()
plt.tight_layout()
plt.legend(prop={'size': 12})
plt.grid(True, linestyle=':', linewidth=0.7, which='both', axis='y')
#plt.show()
plt.savefig("figures/pdf_two_frequency_ablation_all_markers_r.pdf")

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, mae_ridge_avidin, width, label='Avidin', color=colors[0])
rects2 = ax.bar(x, mae_ridge_cd63, width, label='CD63', color=colors[1])
rects3 = ax.bar(x + width, mae_ridge_cd203c, width, label='CD203c', color=colors[2])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('MAE change', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(['f{},f{}'.format(f,g) for f in range(1,7) for g in range(f+1,7)])
ax.set_xlabel('Removed frequencies', fontsize=12)
ax.legend()
plt.tight_layout()
plt.legend()
plt.grid(True, linestyle=':', linewidth=0.7, which='both', axis='y')
#plt.show()
plt.savefig("figures/pdf_two_frequency_ablation_all_markers_mae.pdf")

