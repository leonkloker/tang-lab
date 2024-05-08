import matplotlib.pyplot as plt
import numpy as np
import pickle

viridis = plt.get_cmap('viridis')
colors = [viridis(i/10) for i in range(0,10)]

# Load the data
with open('results/pdf_ridge_regularization_ablation_cd63.pickle', 'rb') as f:
    data_cd63 = pickle.load(f)

with open('results/pdf_ridge_regularization_ablation_avidin.pickle', 'rb') as f:
    data_avidin = pickle.load(f)

with open('results/pdf_ridge_regularization_ablation_cd203c_dMFI*.pickle', 'rb') as f:
    data_cd203c = pickle.load(f)

std = np.linspace(0.2,3.0,29)
points = np.arange(1, 30)

mae_ridge_cd63 = data_cd63["mae_ridge"]#[:-1] - data_cd63["mae_ridge"][-1]
r_ridge_cd63 = data_cd63["r_ridge"]#[:-1] - data_cd63["r_ridge"][-1]
mae_ridge_avidin = data_avidin["mae_ridge"]#[:-1] - data_avidin["mae_ridge"][-1]
r_ridge_avidin = data_avidin["r_ridge"]#[:-1] - data_avidin["r_ridge"][-1]
mae_ridge_cd203c = data_cd203c["mae_ridge"]#[:-1] - data_cd203c["mae_ridge"][-1]) / 14.07
r_ridge_cd203c = data_cd203c["r_ridge"]#[:-1] - data_cd203c["r_ridge"][-1]) / 14.07

mae_ridge_cd203c = [mae / 14.07 for mae in mae_ridge_cd203c]
mean = [(r1 + r2 + r3) / 3 for (r1,r2),r3 in zip(zip(r_ridge_avidin, r_ridge_cd63),r_ridge_cd203c)]
i = np.argmax(np.array(mean))
xmax = np.logspace(-3,2,60)[i]

plt.figure()
""" plt.plot(points[1:], mae_ridge_avidin, label='Avidin', color=colors[0])
plt.plot(points, mae_ridge_cd63, label='CD63', color=colors[1])
plt.plot(points[1:], mae_ridge_cd203c, label='CD203c', color=colors[2]) """
plt.plot(np.logspace(-3,2,60),r_ridge_avidin, label='Avidin', color=colors[0])
plt.plot(np.logspace(-3,2,60),r_ridge_cd63, label='CD63', color=colors[4])
plt.plot(np.logspace(-3,2,60),r_ridge_cd203c, label='CD203c', color=colors[6])
plt.plot(np.logspace(-3,2,60), mean, color=colors[8], label='Mean')
ax = plt.gca()
ylim = ax.get_ylim()
plt.plot([xmax, xmax], [0, 1], '--', color=colors[8])
ax.set_ylim(ylim)
plt.grid(True, linestyle=':', linewidth=0.7)
plt.legend()
plt.xscale('log')
plt.xlabel('Ridge regularization parameter')
plt.ylabel('Pearson coefficient')
#plt.title('Regularization tuning')
plt.tight_layout()
#plt.show()
plt.savefig("figures/pdf_regularization_ridge_tuning_all_markers_r.png", dpi=400)

""" plt.figure()
plt.bar(np.arange(len(mae_ridge_avg)), mae_ridge_avg, align='center')
plt.show() """