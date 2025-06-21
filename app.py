import streamlit as st
import joblib
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import custom_obj

from keras.models import load_model
from keras.utils import custom_object_scope

# Re-define your AttentionLayer (must match saved model definition)
from keras.layers import Layer
import tensorflow as tf

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

# Load model and tokenizer
# model = load_model("spam_lstm_model.h5")
tokenizer = joblib.load("tokenizer.pkl")

# 👇 Correct way to load a model with custom layers
model = load_model("spam_lstm_model.h5", custom_objects={"AttentionLayer": AttentionLayer})

st.title("📧 Email Spam Detection")
text = st.text_area("Enter email text:")

if st.button("Predict"):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=100)
    prediction = model.predict(padded)[0][0]
    label = "Spam 🚫" if prediction > 0.5 else "Not Spam ✅"
    st.success(f"Prediction: {label} (Confidence: {prediction:.2f})")

    if prediction > 0.5:
        st.warning("This email is likely to be spam. Please check and do not send it.")
    else:
        st.success("This email is not likely to be spam. You can send it safely.")

    st.write("Confidence: {:.2f}".format(prediction))
    