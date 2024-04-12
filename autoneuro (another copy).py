# Modified Script with Terminal Inputs for File Selection and Tuner Selection
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
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner import RandomSearch, Hyperband, BayesianOptimization, HyperParameters
import sys
import subprocess

# Request file paths from the user via the terminal. Use default values if no input is provided.
file_path = input("Enter the path to the CSV file you want to process (default 'attributes.csv'): ") or "attributes.csv"
output_file_path = input("Enter the path where you want to save the processed CSV file (default 'attributes_preprocessed.csv'): ") or "attributes_preprocessed.csv"

# Ask the user to select a tuner
print("Select a tuner: 1. RandomSearch, 2. Hyperband, 3. BayesianOptimization")
tuner_selection = input("Enter the number of your choice (default '1'): ") or "1"

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

'''
#model 1
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
'''

'''
# Define a model-building function with regularization and dropout
# model 2
def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units', min_value=32, max_value=512, step=32),
                    activation='relu', input_shape=(X_train_scaled.shape[1],),
                    kernel_regularizer=l2(hp.Float('l2', 1e-5, 1e-2, sampling='log'))))
    for i in range(hp.Int('n_layers', 1, 5)):
        model.add(Dropout(hp.Float('dropout_'+str(i), 0, 0.5, step=0.1)))
        model.add(Dense(units=hp.Int(f'layer_{i}_units', min_value=32, max_value=256, step=32),
                        activation=hp.Choice('act_'+str(i), ['relu', 'tanh']),
                        kernel_regularizer=l2(hp.Float('l2_'+str(i), 1e-5, 1e-2, sampling='log'))))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error'])
    return model
'''
'''
# model 3
def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units', min_value=32, max_value=512, step=32),
                    activation='relu', input_shape=(X_train_scaled.shape[1],),
                    kernel_regularizer=l2(hp.Float('l2', min_value=1e-5, max_value=1e-2, sampling='log'))))
    for i in range(hp.Int('n_layers', min_value=1, max_value=5)):
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float(f'dropout_{i}', min_value=0, max_value=0.5, step=0.1)))
        model.add(Dense(units=hp.Int(f'layer_{i}_units', min_value=32, max_value=256, step=32),
                        activation='relu',
                        kernel_regularizer=l2(hp.Float(f'l2_{i}', min_value=1e-5, max_value=1e-2, sampling='log'))))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error'])
    return model
    '''

# model 4
def build_model(hp):
    model = Sequential()
    # Input layer
    model.add(Dense(
        units=hp.Int('input_units', min_value=32, max_value=512, step=32),
        activation='relu',
        input_shape=(X_train_scaled.shape[1],)
    ))
    
    # Hidden layers
    for i in range(hp.Int('n_layers', 1, 5)):
        model.add(Dense(
            units=hp.Int(f'layer_{i}_units', min_value=32, max_value=256, step=32),
            activation=hp.Choice(f'layer_{i}_activation', ['relu', 'tanh', 'elu'])
        ))
        model.add(Dropout(hp.Float(f'layer_{i}_dropout', min_value=0, max_value=0.5, step=0.1)))

    # Output layer
    model.add(Dense(1, activation='linear'))
    
    # Compile model
    model.compile(
        optimizer=Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )
    
    return model




# Function to initialize the tuner
def initialize_tuner():
    if tuner_selection == "1":
        return RandomSearch(
            build_model,
            objective='val_loss',
            max_trials=10,
            executions_per_trial=1,
            directory='tuner_results',
            project_name='brightness_prediction_random_search',
            overwrite=True
        )
    elif tuner_selection == "2":
        return Hyperband(
            build_model,
            objective='val_loss',
            factor = 3,
            max_epochs=100,
            directory='tuner_results',
            project_name='brightness_prediction_hyperband',
            overwrite=True
        )
    elif tuner_selection == "3":
        return BayesianOptimization(
            build_model,
            objective='val_loss',
            max_trials=20,
            directory='tuner_results',
            project_name='brightness_prediction_bayesian',
            overwrite=True
        )

# Initialize the selected tuner
tuner = initialize_tuner()

# EarlyStopping Callback
#early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Modify the search call to include EarlyStopping
#tuner.search(X_train_scaled, y_train, epochs=20, validation_data=(X_test_scaled, y_test), verbose=2, callbacks=[early_stopping])


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
