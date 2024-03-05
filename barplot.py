import matplotlib.pyplot as plt
import numpy as np

# Usage of features
values = [266, 858, 474, 999, 1000, 999, 57, 0, 249, 47, 0, 198, 826, 1, 0, 978, 43]
values = [1 - val/1000 for val in values]

# Creating a bar plot
plt.figure(figsize=(10, 6))
plt.bar(range(len(values)), values, color='skyblue')
plt.xlabel('Feature index')
plt.ylabel('Feature usage')
plt.title('Rate of non-zero feature coefficient over 1000 runs')
plt.xticks(np.arange(len(values)), rotation=45)
plt.tight_layout()

# Show plot
plt.savefig("feature_usage.png")