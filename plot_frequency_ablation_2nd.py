import matplotlib.pyplot as plt
import numpy as np
import pickle
import re
import sys

def extract_numbers(line):
    return [float(num) for num in re.findall(r'\d+\.\d+', line)]

plt.rcParams['font.family'] = 'Arial'

viridis = plt.get_cmap('viridis')
colors = [viridis(i/5) for i in range(0,10,2)]

# data
r_avidin_ref = 0.6633261794034846
r_avidin = []
r_cd63_ref = 0.9104391620112589
r_cd63 = []
r_cd203c_ref = 0.8529383103999604
r_cd203c = []

mae_avidin_ref = 0.08465914056947567
mae_avidin = []
mae_cd63_ref = 0.10057486255179175
mae_cd63 = []
mae_cd203c_ref = 0.9546141336414323
mae_cd203c = []

f1_avidin_ref = 0.9083333333333333
f1_avidin = []
f1_cd63_ref = 0.8636363636363636
f1_cd63 = []
f1_cd203c_ref = 0.9542583732057416
f1_cd203c = []

for f1 in range(5,6):
    for f2 in [1,2,3,4,6]:
        if f1 == f2:
            continue
        linear_regression_numbers = []
        random_forest_numbers = []
        linear_regression_found = False
        random_forest_found = False
        with open('results/15_fold_mean_unnormalized_freqs{}{}.txt'.format(*sorted([f1, f2])), 'r') as file:
            for line in file:
                if 'Linear Regression' in line:
                    linear_regression_found = True
                    continue
                
                if linear_regression_found and len(linear_regression_numbers) < 6:
                    linear_regression_numbers.extend(extract_numbers(line))
                    if len(linear_regression_numbers) >= 6:
                        linear_regression_numbers = linear_regression_numbers[:6]  # Ensure we only get 6 numbers
                        linear_regression_found = False
                
                if 'Random Forest' in line:
                    random_forest_found = True
                    continue
                
                if random_forest_found and len(random_forest_numbers) < 3:
                    random_forest_numbers.extend(extract_numbers(line))
                    if len(random_forest_numbers) >= 3:
                        random_forest_numbers = random_forest_numbers[:3]  # Ensure we only get 3 numbers
                        random_forest_found = False
                
                if not linear_regression_found and not random_forest_found and len(linear_regression_numbers) == 6 and len(random_forest_numbers) == 3:
                    break

        r_avidin.append(linear_regression_numbers[1])
        r_cd63.append(linear_regression_numbers[5])
        r_cd203c.append(linear_regression_numbers[3])
        mae_avidin.append(linear_regression_numbers[0])
        mae_cd63.append(linear_regression_numbers[4])
        mae_cd203c.append(linear_regression_numbers[2])
        f1_avidin.append(random_forest_numbers[0])
        f1_cd63.append(random_forest_numbers[2])
        f1_cd203c.append(random_forest_numbers[1])

r_avidin = [r - r_avidin_ref for r in r_avidin]
r_cd63 = [r - r_cd63_ref for r in r_cd63]
r_cd203c = [r - r_cd203c_ref for r in r_cd203c]
mae_avidin = [r - mae_avidin_ref for r in mae_avidin]
mae_cd63 = [r - mae_cd63_ref for r in mae_cd63]
mae_cd203c = [r - mae_cd203c_ref for r in mae_cd203c]
f1_avidin = [r - f1_avidin_ref for r in f1_avidin]
f1_cd63 = [r - f1_cd63_ref for r in f1_cd63]
f1_cd203c = [r - f1_cd203c_ref for r in f1_cd203c]

# Plot the data
width = 0.25  # the width of the bars
x = np.array([1,2,3,4,6])
xv = np.array([1,2,3,4,5,6])

fig, ax = plt.subplots()
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
rects1 = ax.bar(x - width, r_avidin, width, label='Avidin', color=colors[0])
rects2 = ax.bar(x, r_cd63, width, label='CD63', color=colors[1])
rects3 = ax.bar(x + width, r_cd203c, width, label='CD203c', color=colors[2])

# Add vertical lines
for i in range(len(xv)-1):
    ax.axvline((xv[i] + xv[i+1])/2, color='black', linestyle='--', linewidth=0.7)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Change in Pearson', fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels(['f{},f{}'.format(5, f) for f in [1,2,3,4,6]])
ax.set_xlabel('Removed frequencies', fontsize=15)
ax.legend()
plt.tight_layout()
plt.legend(prop={'size': 12})
plt.grid(True, linestyle=':', linewidth=0.7, which='both', axis='y')
plt.xticks(rotation=90)
#plt.show()
plt.savefig("results/frequency_ablation_r_2nd.pdf")

fig, ax = plt.subplots()
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
rects1 = ax.bar(x - width, mae_avidin, width, label='Avidin', color=colors[0])
rects2 = ax.bar(x, mae_cd63, width, label='CD63', color=colors[1])
rects3 = ax.bar(x + width, mae_cd203c, width, label='CD203c', color=colors[2])

# Add vertical lines
for i in range(len(xv)-1):
    ax.axvline((xv[i] + xv[i+1])/2, color='black', linestyle='--', linewidth=0.7)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('MAE change', fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels(['f{},f{}'.format(5, f) for f in [1,2,3,4,6]])
ax.set_xlabel('Removed frequencies', fontsize=15)
ax.legend()
plt.tight_layout()
plt.legend()
plt.grid(True, linestyle=':', linewidth=0.7, which='both', axis='y')
plt.xticks(rotation=90)
#plt.show()
plt.savefig("results/frequency_ablation_mae_2nd.pdf")

fig, ax = plt.subplots()
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
rects1 = ax.bar(x - width, f1_avidin, width, label='Avidin', color=colors[0])
rects2 = ax.bar(x, f1_cd63, width, label='CD63', color=colors[1])
rects3 = ax.bar(x + width, f1_cd203c, width, label='CD203c', color=colors[2])

# Add vertical lines
for i in range(len(xv)-1):
    ax.axvline((xv[i] + xv[i+1])/2, color='black', linestyle='--', linewidth=0.7)
    
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('F1 change', fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels(['f{},f{}'.format(5, f) for f in [1,2,3,4,6]])
ax.set_xlabel('Removed frequencies', fontsize=15)
ax.legend()
plt.tight_layout()
plt.legend()
plt.grid(True, linestyle=':', linewidth=0.7, which='both', axis='y')
plt.xticks(rotation=90)
#plt.show()
plt.savefig("results/frequency_ablation_f1_2nd.pdf")
