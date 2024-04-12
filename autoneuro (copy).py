# Modified Script with Terminal Inputs for File Selection
# ---------------------------------------------------------------------------------

# Import Section
import pandas as pd
import ast
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras_tuner import RandomSearch, HyperParameters
import sys
import subprocess

# Request file paths from the user via the terminal. Use default values if no input is provided.
file_path = input("Enter the path to the CSV file you want to process (default 'attributes.csv'): ") or "attributes.csv"
output_file_path = input("Enter the path where you want to save the processed CSV file (default 'attributes_preprocessed.csv'): ") or "attributes_preprocessed.csv"


# ---------------------------------------------------------------------------------
# Part 1: Preprocessing Data

def preprocess_and_feature_engineer(file_path, output_file_path):
    # Load the data
    data = pd.read_csv(file_path)
    
    # Convert string representations of tuples to actual tuples
    data['MyOrient'] = data['MyOrient'].apply(ast.literal_eval)
    data['Cd'] = data['Cd'].apply(ast.literal_eval)
    data['MyLightDir'] = data['MyLightDir'].apply(ast.literal_eval)
    
    # Calculate the brightness of 'Cd'
    def calculate_brightness(rgb):
        return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    
    data['Brightness'] = data['Cd'].apply(calculate_brightness)
    
    # Extract individual components from 'MyOrient' and 'MyLightDir'
    data[['MyOrient_w', 'MyOrient_x', 'MyOrient_y', 'MyOrient_z']] = pd.DataFrame(data['MyOrient'].tolist(), index=data.index)
    data[['MyLightDir_x', 'MyLightDir_y', 'MyLightDir_z']] = pd.DataFrame(data['MyLightDir'].tolist(), index=data.index)
    
    # Drop the original columns to simplify the dataframe
    data = data.drop(columns=['MyOrient', 'Cd', 'MyLightDir'])
    
    # Save the preprocessed and feature-engineered data to a new CSV file
    data.to_csv(output_file_path, index=False)
    print(f"Preprocessed data has been saved to {output_file_path}")

# Run the preprocessing function with provided file paths
preprocess_and_feature_engineer(file_path, output_file_path)

# ---------------------------------------------------------------------------------
# Part 2: Neural Network Autotuning

# Load the dataset
data = pd.read_csv(output_file_path)

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
    model.add(Dense(units=hp.Int('units', min_value=32, max_value=512, step=32),
                    activation='relu', input_shape=(X_train_scaled.shape[1],)))
    for i in range(hp.Int('n_layers', 1, 5)):
        model.add(Dense(units=hp.Int(f'layer_{i}_units', min_value=32, max_value=256, step=32),
                        activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error'])
    return model

# Initialize the tuner
#tuner = RandomSearch(
#    build_model,
#    objective='val_loss',
#    max_trials=10,
#    executions_per_trial=1,
#    directory='tuner_results',
#    project_name='brightness_prediction'
#)

from keras_tuner import Hyperband

tuner = Hyperband(
    build_model,
    objective='val_loss',
    max_epochs=40,
    directory='tuner_results',
    project_name='brightness_prediction',
    overwrite=True
)



# Perform hyperparameter search
tuner.search(X_train_scaled, y_train, epochs=20, validation_data=(X_test_scaled, y_test), verbose=2)

# Fetch the best model
best_model = tuner.get_best_models(num_models=1)[0]

# ---------------------------------------------------------------------------------
# Continue with the evaluation, saving, and conversion as before

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
