# Machine Learning and Neural Networks libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Layer, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import HeNormal

from constants import FEATURE_COLUMNS, TARGET_COLUMNS
#from quaternion_eye_dir_layer import QuaternionEyeDirLayer
#from helper_functions import tf_r2_score

# Model-building function with hyperparameters

# Modify the build_model function to change the output layer
# Initialize the Huber loss with a specific delta value
huber_loss = tf.keras.losses.Huber(delta=1)  # Adjust delta as needed
num_features = len(FEATURE_COLUMNS)  # Automatically updates based on defined feature columns
num_targets = len(TARGET_COLUMNS)

def build_model_alt(hp):
    model = Sequential()
    # Initial Dense layer to process the input features
    model.add(Dense(
        units=hp.Int('initial_units', min_value=32, max_value=512, step=32),
        activation=hp.Choice('initial_activation', ['relu', 'tanh', 'elu']),
        #activation = 'relu',  # Use ReLU activation for the initial layer
        kernel_initializer=HeNormal(),
        kernel_regularizer=l2(0.01),
        input_shape=(num_features,),
    ))

    # Add hidden layers as before
    for i in range(hp.Int('n_layers', 2, 5)): #base 2,5 layers
        model.add(Dense(
            units=hp.Int(f'layer_{i}_units', min_value=32, max_value=512, step=32),
            activation=hp.Choice(f'layer_{i}_activation', ['relu', 'tanh', 'elu']),
            kernel_initializer=HeNormal(),
            kernel_regularizer=l2(0.01),
        ))
        model.add(Dropout(hp.Float(f'layer_{i}_dropout', min_value=0.01, max_value=0.5, step=0.1)))

    # Output layer for RGB components
    #model.add(Dense(3, activation='linear'))  # Consider activation based on your color vector's range
    model.add(Dense(num_targets, activation='linear', kernel_initializer=HeNormal()))  # Consider activation based on your color vector's range
    
    # Hyperparameter tuning for Huber loss delta
    huber_delta = hp.Float('huber_delta', min_value=0.5, max_value=2.0, step=0.1)
    huber_loss = tf.keras.losses.Huber(delta=huber_delta)
    
    # Compile model
    model.compile(optimizer=Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                  #loss=custom_loss,  # Custom loss function
                  #loss='huber_loss',  # Huber loss with delta=1
                  #loss = tf.keras.losses.Huber(delta=1),
                  loss = huber_loss,
                  #loss='mean_squared_error',
                  #loss='mean_absolute_error',
                  #loss='binary_crossentropy',# nicely blue
                  #loss='categorical_crossentropy',
                  #loss='sparse_categorical_crossentropy',# bad
                  #loss='cosine_similarity',
                  #loss='poisson',
                  #metrics=['mean_absolute_error', tf_r2_score]
                  metrics=['mean_absolute_error', 'mean_squared_error'] 
                  )

    return model
