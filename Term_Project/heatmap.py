import numpy as np
import matplotlib.pyplot as plt

# Create the data
data = np.array([
    [311, 343],
    [1205, 163]
])

# Create the heatmap
plt.imshow(data, cmap='Blues')
plt.colorbar()

# Add labels and title
plt.xlabel('True Class')
plt.ylabel('Predected Class')
plt.title('Testing Results Heatmap')

# Add numbers to the heatmap
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        plt.text(j, i, data[i, j], ha='center', va='center')

# Show the plot
plt.show()