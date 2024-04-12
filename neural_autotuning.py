import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from kerastuner.tuners import RandomSearch
import sys
import subprocess

# Load the dataset
data = pd.read_csv('preprocessed_attributes.csv')

# Prepare the features and target variable
features = data[['MyOrient_w', 'MyOrient_x', 'MyOrient_y', 'MyOrient_z', 'MyLightDir_x', 'MyLightDir_y', 'MyLightDir_z']]
target = data['Brightness']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define a model-building function
def build_model(hp):
    model = Sequential()
    model.add(Dense(hp.Int('input_units', min_value=32, max_value=512, step=32), activation='relu', input_shape=(X_train_scaled.shape[1],)))
    
    for i in range(hp.Int('n_layers', 1, 3)):
        model.add(Dense(hp.Int(f'layer_{i}_units', min_value=32, max_value=512, step=32), activation='relu'))
    
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                  loss='mse')
    return model

# Initialize the tuner
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=20,
    executions_per_trial=1,
    directory='keras_tuner_dir',
    project_name='brightness_prediction'
)

# Perform hyperparameter tuning
tuner.search(X_train_scaled, y_train, epochs=20, validation_split=0.2, verbose=1)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"Best hyperparameters:\n{best_hps.values}")

# Build the model with the best hyperparameters and train it on the data
best_model = tuner.hypermodel.build(best_hps)
best_model.fit(X_train_scaled, y_train, epochs=20, validation_split=0.2, verbose=1)

# Save the best model
best_model.save('best_model.h5')
# Save the model in SavedModel format
model_save_path = "best_model_savedmodel"
best_model.save(model_save_path, save_format="tf")


# Convert the TensorFlow SavedModel to ONNX
output_onnx_model = 'best_model.onnx'
subprocess.run([
    sys.executable, "-m", "tf2onnx.convert",
    "--saved-model", model_save_path,
    "--output", output_onnx_model
], check=True)

# Predictions
predictions = best_model.predict(X_test_scaled).flatten()

# Evaluate the model
print(f"Neural Network MAE: {mean_absolute_error(y_test, predictions)}")
print(f"Neural Network MSE: {mean_squared_error(y_test, predictions)}")
print(f"Neural Network R²: {r2_score(y_test, predictions)}")
