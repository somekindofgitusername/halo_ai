import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the data
data = pd.read_csv('attributes.csv')  # Replace 'your_dataset.csv' with your actual file path

# Step 2: Summary Statistics
print("Summary Statistics:")
print(data.describe())

# Step 3: Visualization of Distributions
# Plotting histograms for all features
data.hist(bins=15, figsize=(15, 10))
plt.suptitle('Histograms of All Features')
plt.show()

# Step 4: Correlation Matrix
corr = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Step 5: Pair Plot
# Selecting a subset if too many features, for better visualization
columns_to_plot = ['eulerx', 'eulery', 'eulerz', 'eyedir_x', 'eyedir_y', 'eyedir_z']
sns.pairplot(data[columns_to_plot])
plt.suptitle('Pair Plot of Selected Features')
plt.show()
