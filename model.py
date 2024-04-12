# Machine Learning and Neural Networks libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Layer, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras import backend as K

from constants import FEATURE_COLUMNS, TARGET_COLUMNS
from quaternion_eye_dir_layer import QuaternionEyeDirLayer
from helper_functions import tf_r2_score

# Model-building function with hyperparameters

# Modify the build_model function to change the output layer
# Initialize the Huber loss with a specific delta value
huber_loss = tf.keras.losses.Huber(delta=1)  # Adjust delta as needed
num_features = len(FEATURE_COLUMNS)  # Automatically updates based on defined feature columns

'''def custom_loss():
    threshold = 0.8  # Define the threshold for strong colors
    high_weight = 10.0  # Higher penalty for errors above the threshold
    low_weight = 1.0  # Standard penalty for other cases
    
    def loss(y_true, y_pred):
        # Apply weights based on the condition
        weights = tf.where(y_true > threshold, high_weight, low_weight)
        # Calculate weighted squared difference
        squared_difference = tf.square(y_true - y_pred) * weights
        return tf.reduce_mean(squared_difference, axis=-1)
    
    return loss
'''

def custom_loss(y_true, y_pred):
    threshold = 0.8
    high_weight = 10.0
    low_weight = 1.0

    # Ensure these are tensors with the correct data type
    high_weight = tf.constant(high_weight, dtype=tf.float32)
    low_weight = tf.constant(low_weight, dtype=tf.float32)
    
    error = y_true - y_pred
    is_high_value = y_true > threshold
    is_underestimation = error > 0

    # Apply higher weight for underpredictions of high values
    # Directly use tf.where without '== True'
    weights = tf.where(is_high_value & is_underestimation, high_weight, low_weight)
    weighted_squared_error = tf.square(error) * weights
    
    return tf.reduce_mean(weighted_squared_error)


'''
def build_model(hp, input_shape=(3,), num_features=9):
    model = Sequential()
    model.add(QuaternionEyeDirLayer(
        units=hp.Int('combined_units', min_value=16, max_value=512, step=32),
        #activation='relu',
        activation='tanh',
        input_shape=(num_features,)  # Input shape depends on the total number of features
    ))
    # Add hidden layers as before
    for i in range(hp.Int('n_layers', 4, 4)):
        model.add(Dense(
            units=hp.Int(f'layer_{i}_units', min_value=16, max_value=512, step=32),
            #activation=hp.Choice(f'layer_{i}_activation', ['relu', 'tanh', 'elu'])
            activation=hp.Choice(f'layer_{i}_activation', [
            'relu',  # Rectified Linear Unit
            'tanh',  # Hyperbolic Tangent
            'elu',   # Exponential Linear Unit
            #'selu',  # Scaled Exponential Linear Unit
            #'sigmoid',  # Sigmoid
            #'softplus',  # SoftPlus
            #'softsign',  # SoftSign
            #'hard_sigmoid',  # Hard Sigmoid
            #'linear'  # Linear Activation
        ])
        ))
        model.add(Dropout(hp.Float(f'layer_{i}_dropout', min_value=0, max_value=0.5, step=0.1)))

    # Adjust the output layer to have 3 units for the RGB components
    model.add(Dense(3, activation='linear'))  # Consider the activation based on your color vector's range

    model.compile(optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
              #loss=custom_loss(threshold=0.5, high_weight=10.0, low_weight=1.0),  # Assuming you are using the custom loss as described previously
              loss=custom_loss(),  # No need to pass parameters here
              metrics=['mean_absolute_error',  tf_r2_score])

    
    return model
'''
def build_model_alt(hp):
    model = Sequential()
    # Initial Dense layer to process the input features
    model.add(Dense(
        units=hp.Int('initial_units', min_value=16, max_value=512, step=32),
        activation=hp.Choice('initial_activation', ['relu', 'tanh', 'elu']),
        input_shape=(num_features,),
    ))

    # Add hidden layers as before
    for i in range(hp.Int('n_layers', 1, 5)):
        model.add(Dense(
            units=hp.Int(f'layer_{i}_units', min_value=16, max_value=512, step=32),
            activation=hp.Choice(f'layer_{i}_activation', ['relu', 'tanh', 'elu']),
        ))
        model.add(Dropout(hp.Float(f'layer_{i}_dropout', min_value=0, max_value=0.5, step=0.1)))

    # Output layer for RGB components
    model.add(Dense(3, activation='linear'))  # Consider activation based on your color vector's range

    # Compile model
    model.compile(optimizer=Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                  loss=custom_loss,  # Custom loss function
                  metrics=['mean_absolute_error', tf_r2_score])

    return model
