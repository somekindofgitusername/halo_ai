import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import HeNormal, GlorotUniform, RandomNormal, LecunNormal, Orthogonal
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from constants import FEATURE_COLUMNS, TARGET_COLUMNS

num_features = len(FEATURE_COLUMNS)
num_targets = len(TARGET_COLUMNS)

def build_model(hp):
    model = Sequential()
    # Allows choosing the kernel initializer
    kernel_initializer = hp.Choice('kernel_initializer', ['he_normal', 'glorot_uniform', 'random_normal', 'lecun_normal', 'orthogonal'])
    initializer_dict = {
    'he_normal': HeNormal(),
    'glorot_uniform': GlorotUniform(),
    'random_normal': RandomNormal(),
    'lecun_normal': LecunNormal(),
    'orthogonal': Orthogonal()
    }
    # Initial Dense layer to process the input features
    model.add(Dense(
        units=hp.Int('initial_units', min_value=16, max_value=256, step=8),
        activation=hp.Choice('initial_activation', ['relu', 'tanh', 'elu', 'sigmoid']),
        #kernel_initializer=LecunNormal(),
        kernel_initializer=initializer_dict[kernel_initializer],
        #kernel_regularizer=l2(0.0001),
        #kernel_regularizer=l2( hp.Float('l2', min_value=0.001, max_value=0.1, step=0.001)),
        input_shape=(num_features,)
    ))
    model.add(BatchNormalization())  # Add batch normalization layer after the input layer

    # Add hidden layers
    for i in range(hp.Int('n_layers', 6, 6)): # 4
        model.add(Dense(
            units=hp.Int(f'layer_{i}_units', min_value=16, max_value=256, step=8),
            activation=hp.Choice('initial_activation', ['relu', 'tanh', 'elu', 'sigmoid']),
            #kernel_initializer=HeNormal(),
            #kernel_initializer=LecunNormal(),
            kernel_initializer=initializer_dict[kernel_initializer],
            #kernel_regularizer=l2( hp.Float('l2', min_value=0.001, max_value=0.1, step=0.001)),
            #kernel_regularizer=l2(0.0001),
        ))
        #model.add(BatchNormalization())
        #model.add(Dropout(hp.Float(f'layer_{i}_dropout', min_value=0.05, max_value=0.5, step=0.05)))

    # Output layer
    model.add(Dense(num_targets, activation='linear', kernel_initializer=initializer_dict[kernel_initializer]))

    # Compile model with adaptive learning rate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=hp.Float('initial_learning_rate', min_value=1e-4, max_value=1e-2, sampling='log'),
        decay_steps=hp.Int('decay_steps', min_value=1000, max_value=10000, step=1000),
        decay_rate=hp.Float('decay_rate', min_value=0.7, max_value=0.99, step=0.01)
    )
    optimizer = Adam(learning_rate=lr_schedule)
    
    delta = hp.Float('delta', min_value=0.1, max_value=5.0, step=0.1)
    
    model.compile(optimizer=optimizer,
                  #loss=tf.keras.losses.Huber(delta=0.5),
                  loss=tf.keras.losses.Huber(delta=delta),
                  metrics=['mean_absolute_error', 'mean_squared_error']
                 )

    return model

'''# Include EarlyStopping and ModelCheckpoint in your training script
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(
    filepath='model.h5',  # Save your model to a file
    monitor='val_loss',
    save_best_only=True)'''

