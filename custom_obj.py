# ✅ Alternative (Cleaner) Way
# Register the layer when saving and loading:

from keras.models import load_model
from keras.utils import custom_object_scope

# Re-define your AttentionLayer (must match saved model definition)
from keras.layers import Layer
import tensorflow as tf
    
from keras.utils import register_keras_serializable

@register_keras_serializable()
class AttentionLayer(Layer):
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], 1),
                                 initializer='random_normal', trainable=True)
        super().build(input_shape)
    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W))
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)
