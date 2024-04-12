import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, make_scorer

# Custom scorer function that penalizes underpredictions of high values more
def custom_scorer(y_true, y_pred):
    error = y_true - y_pred
    underestimation_penalty = np.where(error > 0, 10, 1)  # More penalty if underprediction
    return np.mean((error**2) * underestimation_penalty)

# Load your data
data = pd.read_csv('attributes.csv')  # Replace with your actual data file

# Preparing the input features and target variable
X = data[['eulerx', 'eulery', 'eulerz', 'eyedir_x', 'eyedir_y', 'eyedir_z']]
y = data['color_red']  # Focus on red color intensity

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Assign higher weights to higher intensity values during model training
weights = np.where(y_train > 0.8, 10, 1)  # Increase the weight for high color intensities

# Fit a Gradient Boosting model
gb = GradientBoostingRegressor(n_estimators=200, random_state=42)
gb.fit(X_train, y_train, sample_weight=weights)

# Predict on the test set
y_pred = gb.predict(X_test)

# Calculate the Mean Squared Error of the model using the custom scorer
mse = mean_squared_error(y_test, y_pred)
custom_mse = custom_scorer(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'Custom Mean Squared Error: {custom_mse}')

# Plot actual vs predicted values for red color intensity
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)  # Ideal prediction line
plt.title('Actual vs Predicted Red Color Intensity')
plt.xlabel('Actual Red Intensity')
plt.ylabel('Predicted Red Intensity')
plt.show()

# Feature Importance from the model
feature_importances = gb.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]

# Plot the feature importance
plt.barh(range(X.shape[1]), feature_importances[sorted_indices], align='center')
plt.yticks(range(X.shape[1]), X.columns[sorted_indices])
plt.title('Feature Importance')
plt.show()
