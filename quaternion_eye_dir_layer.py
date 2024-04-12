import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import L2

# Custom layer for quaternion and eye direction vector
class QuaternionEyeDirLayer(Layer):
    def __init__(self, units=512, activation='tanh', **kwargs):
        super(QuaternionEyeDirLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
 
        self.quaternion_kernel = self.add_weight(
            shape=(4, self.units), 
            initializer='glorot_uniform', 
            name='quaternion_kernel', 
            trainable=True,
            regularizer=L2(0.00121)
        )
        self.eye_dir_kernel = self.add_weight(
            shape=(3, self.units), 
            initializer='glorot_uniform', 
            name='eye_dir_kernel', 
            trainable=True,
            regularizer=L2(0.00121)
        )
      
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros', 
            name='bias', 
            trainable=True,
            regularizer=L2(0.00121)
        )

    def call(self, inputs):
        quaternion = inputs[:, :4]
        eye_dir = inputs[:, 4:7]
        other_features = inputs[:, 7:]
        
        quaternion_transformed = tf.matmul(quaternion, self.quaternion_kernel)
        eye_dir_transformed = tf.matmul(eye_dir, self.eye_dir_kernel)
        
        features_transformed = quaternion_transformed + eye_dir_transformed + self.bias
        activated_features = self.activation(features_transformed)
        
        return tf.concat([activated_features, other_features], axis=1)

    def get_config(self):
        config = super(QuaternionEyeDirLayer, self).get_config()
        config.update({'units': self.units, 'activation': self.activation})
        return config
