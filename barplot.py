import matplotlib.pyplot as plt
import numpy as np
import pickle

viridis = plt.get_cmap('viridis')
colors = [viridis(i/5) for i in range(0,10,2)]

# Load the data
with open('results/pdf_two_frequency_ablation_cd63.pickle', 'rb') as f:
    data_cd63 = pickle.load(f)

with open('results/pdf_two_frequency_ablation_avidin.pickle', 'rb') as f:
    data_avidin = pickle.load(f)

with open('results/pdf_two_frequency_ablation_cd203c_dMFI*.pickle', 'rb') as f:
    data_cd203c = pickle.load(f)

std = np.linspace(0.2,3.0,29)
points = np.arange(1, 30)

mae_ridge_cd63 = data_cd63["mae_ridge"][:-1] - data_cd63["mae_ridge"][-1]
r_ridge_cd63 = data_cd63["r_ridge"][:-1] - data_cd63["r_ridge"][-1]
mae_ridge_avidin = data_avidin["mae_ridge"][:-1] - data_avidin["mae_ridge"][-1]
r_ridge_avidin = data_avidin["r_ridge"][:-1] - data_avidin["r_ridge"][-1]
mae_ridge_cd203c = (data_cd203c["mae_ridge"][:-1] - data_cd203c["mae_ridge"][-1]) / 14.07
r_ridge_cd203c = (data_cd203c["r_ridge"][:-1] - data_cd203c["r_ridge"][-1])

# Plot the data
width = 0.25  # the width of the bars
x = np.arange(1,16)

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, r_ridge_avidin, width, label='Avidin', color=colors[0])
rects2 = ax.bar(x, r_ridge_cd63, width, label='CD63', color=colors[1])
rects3 = ax.bar(x + width, r_ridge_cd203c, width, label='CD203c', color=colors[2])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Change in R')
ax.set_title('Frequency ablation')
ax.set_xticks(x)
ax.set_xticklabels(['{},{}'.format(f,g) for f in range(1,7) for g in range(f+1,7)])
ax.set_xlabel('Removed frequencies')
ax.legend()
plt.tight_layout()
plt.legend()


""" for x in range(0, len(mae_ridge), 20):
    plt.axvline(x, color='red', linestyle='--', linewidth=0.8)
 """
plt.grid(True, linestyle=':', linewidth=0.7, which='both', axis='y')
#plt.show()
plt.savefig("figures/pdf_two_frequency_ablation_all_markers_r.png", dpi=400)

""" plt.figure()
plt.bar(np.arange(len(mae_ridge_avg)), mae_ridge_avg, align='center')
plt.show() """