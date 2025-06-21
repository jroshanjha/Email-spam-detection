from flask import Flask, request, jsonify,render_template,render_template_string
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

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


app = Flask(__name__)
# model = load_model("spam_lstm_model.h5")
tokenizer = joblib.load("tokenizer.pkl")

# 👇 Correct way to load a model with custom layers
model = load_model("spam_lstm_model.h5", custom_objects={"AttentionLayer": AttentionLayer})


HTML = """
<!DOCTYPE html>
<html>
<head><title>Email Spam Detector</title></head>
<body style="font-family:sans-serif; text-align:center;">
    <h1>📧 Email Spam Detection</h1>
    <form method="post">
        <textarea name="email" rows="10" cols="60" placeholder="Paste your email here..."></textarea><br><br>
        <input type="submit" value="Predict">
    </form>
    {% if prediction %}
        <h3>Prediction: {{ prediction }}</h3>
        <p>Confidence: {{ confidence }}</p>
    {% endif %}
</body>
</html>
"""
@app.route("/", methods=["GET", "POST"])
def home():
    prediction, confidence = None, None
    if request.method == "POST":
        text = request.form["email"]
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=100)
        prob = model.predict(padded)[0][0]
        prediction = "Spam 🚫" if prob > 0.5 else "Not Spam ✅"
        confidence = f"{prob:.2f}"
    return render_template_string(HTML, prediction=prediction, confidence=confidence)


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=100)
    probability = model.predict(padded)[0][0]
    return jsonify({
        "prediction": "Spam" if probability > 0.5 else "Not Spam",
        "confidence": float(round(probability, 3))
    })

if __name__ == "__main__":
    app.run(debug=True)
