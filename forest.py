import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Load your data
data = pd.read_csv('attributes.csv')  # Replace with your actual data file

# Preparing the input features and target variable
X = data[['eulerx', 'eulery', 'eulerz', 'eyedir_x', 'eyedir_y', 'eyedir_z']]
y = data[['color_red', 'color_green', 'color_blue']]  # Assuming these are your target variables

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a Random Forest model
# Random Forest can capture non-linearities and complex interactions between features
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict on the test set
y_pred = rf.predict(X_test)

# Calculate the Mean Squared Error of the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Feature Importance from the Random Forest model
feature_importances = rf.feature_importances_
indices = np.argsort(feature_importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(X.shape[1]):
    print(f"{f + 1}. feature {X.columns[indices[f]]} ({feature_importances[indices[f]]})")

# Scatter plot of actual vs predicted colors for a specific color channel (e.g., red)
plt.scatter(y_test['color_red'], y_pred[:, 0], alpha=0.5)
plt.title('Actual vs Predicted Red Color Intensity')
plt.xlabel('Actual Red Intensity')
plt.ylabel('Predicted Red Intensity')
plt.show()

# Further visualizations
# Here we'll plot the actual vs predicted color intensities
sns.pairplot(data, vars=['color_red', 'color_green', 'color_blue', 'eulerx', 'eulery', 'eulerz'],
             palette='viridis', diag_kind='kde')
plt.suptitle('Pair Plot of Color Intensities and Euler Angles')
plt.show()
