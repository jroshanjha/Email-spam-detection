import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

@tf.keras.utils.register_keras_serializable()
class AttentionLayer(Layer):
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        super().build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(K.dot(x, self.W))
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)
