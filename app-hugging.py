import gradio as gr
import joblib
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import get_custom_objects

# Redefine the AttentionLayer
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

class AttentionLayer(Layer):
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], 1),
                                 initializer='random_normal', trainable=True)
        super().build(input_shape)
        
    def call(self, x):
        e = K.tanh(K.dot(x, self.W))
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

# Register the custom layer
get_custom_objects().update({'AttentionLayer': AttentionLayer})

# Load model and tokenizer
tokenizer = joblib.load("tokenizer.pkl")
model = load_model("spam_lstm_model.h5")

# Define prediction function
def predict_spam(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100)
    prob = model.predict(padded)[0][0]
    return {"Spam 🚫": float(round(prob, 3)), "Not Spam ✅": float(round(1 - prob, 3))}

# Launch Gradio UI
iface = gr.Interface(
    fn=predict_spam,
    inputs="text",
    outputs="label",
    title="📧 Email Spam Detection",
    description="Powered by LSTM + AttentionLayer with Gradio"
)

iface.launch()
