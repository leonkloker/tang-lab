import numpy as np
import matplotlib.pyplot as plt
import pickle

file = open("./results/ablation.pickle", "rb")
mae_linear_baseline = 0.03656097339116495
mae_lasso_baseline = 0.038576994164388145
mae_ridge_baseline = 0.03644015886901764
mae_svr_baseline = 0.04064968702876791
mae_linear = [mae - mae_linear_baseline for mae in pickle.load(file)]
mae_lasso = [mae - mae_lasso_baseline for mae in pickle.load(file)]
mae_ridge = [mae - mae_ridge_baseline for mae in pickle.load(file)]
mae_svr = [mae - mae_svr_baseline for mae in pickle.load(file)]

classes = ['Linear', 'Lasso', 'Ridge', 'SVR']
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