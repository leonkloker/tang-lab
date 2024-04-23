import matplotlib.pyplot as plt
import numpy as np
import pickle

# Load the data
with open('results/pdf_sampling_points_ablation_cd63.pickle', 'rb') as f:
    data = pickle.load(f)

std = np.linspace(0.2,3.0,29)
points = np.arange(1, 30)

mae_ridge = data["mae_ridge"]#[:-1] - data["mae_ridge"][-1]
r_ridge = data["r_ridge"]#[:-1] - data["r_ridge"][-1]
r_lasso = data["r_lasso"][:-1] - data["r_lasso"][-1]
f1 = data["f1_svc"][:-1] - data["f1_svc"][-1]

# Plot the data
plt.figure()
plt.plot(points, r_ridge, label='MAE Ridge')
#plt.bar(np.arange(len(mae_ridge)), mae_ridge, align='center')

#for x in range(0, len(mae_ridge), 20):
#    plt.axvline(x, color='red', linestyle='--', linewidth=0.8)

plt.grid(True, linestyle=':', linewidth=0.5, which='both', axis='y')
plt.show()