import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from constants import FEATURE_COLUMNS, TARGET_COLUMNS

num_features = len(FEATURE_COLUMNS)
num_targets = len(TARGET_COLUMNS)

def build_best_model():
    num_features = len(FEATURE_COLUMNS)
    num_targets = len(TARGET_COLUMNS)

    model = Sequential()
    model.add(Dense(
        units=16,
        activation='tanh',
        kernel_initializer=HeNormal(),
        kernel_regularizer=l2(0.01),
        input_shape=(num_features,)
    ))
    model.add(BatchNormalization())

    # Define each layer as per best trial
    layers = [
        (32, 'relu', 0.15),
        (48, 'relu', 0.1),
        (144, 'elu', 0.4),
        (64, 'elu', 0.25)
    ]
    for units, activation, dropout in layers:
        model.add(Dense(
            units=units,
            activation=activation,
            kernel_initializer=HeNormal(),
            kernel_regularizer=l2(0.01)
        ))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

    model.add(Dense(num_targets, activation='linear', kernel_initializer=HeNormal()))

    # Compile model with the adaptive learning rate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.0004946049376824358,
        decay_steps=4000,
        decay_rate=0.72
    )
    optimizer = Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.Huber(delta=1.6),
                  metrics=['mean_absolute_error', 'mean_squared_error'])

    return model

def build_best_model2():
    num_features = len(FEATURE_COLUMNS)
    num_targets = len(TARGET_COLUMNS)

    model = Sequential()
    # Initial Dense layer to process the input features
    model.add(Dense(
        units=176,  # From hyperparameters
        activation='elu',  # From hyperparameters
        kernel_initializer=HeNormal(),
        kernel_regularizer=l2(0.01),
        input_shape=(num_features,)
    ))
    model.add(BatchNormalization())

    # Define each layer as per best trial
    layers = [
        (112, 'elu', 0.4),   # Layer 0
        (64, 'tanh', 0.05),  # Layer 1
        (80, 'relu', 0.4),   # Layer 2
        (208, 'tanh', 0.4)   # Layer 3
    ]
    for units, activation, dropout in layers:
        model.add(Dense(
            units=units,
            activation=activation,
            kernel_initializer=HeNormal(),
            kernel_regularizer=l2(0.01)
        ))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

    # Output layer
    model.add(Dense(num_targets, activation='linear', kernel_initializer=HeNormal()))

    # Compile model with adaptive learning rate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.008253677322333449,  # From hyperparameters
        decay_steps=8000,  # From hyperparameters
        decay_rate=0.8  # From hyperparameters
    )
    optimizer = Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.Huber(delta=1.6),
                  metrics=['mean_absolute_error', 'mean_squared_error'])

    return model
