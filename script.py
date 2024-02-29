import numpy as np
import matplotlib.pyplot as plt
import pickle

file = open("./results/ablation.pickle", "rb")
mae_linear = pickle.load(file)
mae_lasso = pickle.load(file)
mae_ridge = pickle.load(file)
mae_svr = pickle.load(file)

classes = ['linear', 'lasso', 'ridge', 'svr']
data = np.array([mae_linear, mae_lasso, mae_ridge, mae_svr]).T
n_data_points = data.shape[0]
x = np.arange(n_data_points)
bar_width = 0.2
fig, ax = plt.subplots(figsize=(10, 6))

# Plotting each class
for i in range(4):
    ax.bar(x + i*bar_width, data[:, i], width=bar_width, label=classes[i])

# Adding some customization
ax.set_xlabel('Feature')
ax.set_ylabel('MAE')
ax.set_title('Mean absolute errors for leaving out different features')
ax.set_xticks(x + bar_width + bar_width/2)
ax.set_xticklabels([f'{i+1}' for i in range(n_data_points)])
ax.legend()

# Show plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('./figures/ablation.png')
plt.show()