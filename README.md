# 📧 Spam Detection App (ML + LSTM + Streamlit + Flask + Gradio)
A simple ML/DL-powered spam detector using TensorFlow + Gradio, deployed on Hugging Face Spaces.

Built a hybrid ML + DL system to classify email text as Spam or Not Spam using TF-IDF + StackingClassifier and a BiLSTM model with a custom Attention Layer.

Developed a RESTful Flask API and interactive Streamlit and Gradio UIs for real-time predictions.

Implemented overfitting control techniques: Dropout, EarlyStopping, and ReduceLROnPlateau.

Evaluated models using Accuracy, Precision, and ROC-AUC scores for reliability in production use.

Packaged and deployed the application using Docker, and automated testing/deployment with GitHub Actions CI/CD.

Hosted on Hugging Face Spaces for public access and testing with Gradio frontend.

This project predicts whether an email is **Spam** or **Not Spam** using:
- ✅ ML (StackingClassifier: RandomForest + GradientBoost + LogisticRegression)
- ✅ DL (BiLSTM + AttentionLayer)
- ✅ Streamlit Frontend
- ✅ Flask API
- ✅ Docker Deployment
- ✅ CI/CD with GitHub Actions

---

## 📦 Project Structure

A complete ML + DL-based spam detection system with:

- ✅ Streamlit Web App
- ✅ Flask REST API with HTML frontend
- ✅ LSTM + Attention Model
- ✅ Docker Deployment
- ✅ GitHub CI/CD Pipeline

---

## 🔧 Features

| Component        | Tech Used               |
|------------------|-------------------------|
| UI               | Streamlit / HTML (Flask)|
| Model            | LSTM with Attention     |
| Deployment       | Docker, CI/CD           |
| Input            | Free text (email body)  |
| Output           | Spam or Not Spam        |

---

- TF-IDF Vectorizer + Ensemble Model
- LSTM with Custom Attention Layer
- Overfitting Techniques: Dropout, EarlyStopping, ReduceLROnPlateau
- Evaluation: Accuracy, Precision, ROC-AUC
- Flask API + Streamlit UI
- Docker-ready
- GitHub CI/CD starter

## 🚀 Getting Started

### 1. Clone this repository

```bash
git clone https://github.com/jroshanjha/Email-spam-detection.git
cd Email-spam-detection 

## Created Virtual Environment 
python -m venv venv

## Activated Virtual Environment
venv/Scripts/activate

## Install Dependience Libraries
pip install -r requirements.txt


#🐳 Docker Build & Run
docker build -t email-spam-detection .
docker run -p 8501:8501 email-spam-detection

📊 UI Access
Streamlit: http://localhost:8501

Flask API: http://localhost:5000

🚀 4. Deployment to Hugging Face Spaces

🧪 Local Test (before deployment)
pip install gradio tensorflow joblib
python app.py

Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://e941041fbc01fde488.gradio.live
# API 
http://localhost:5000/api/predicit
Input:  { "text": "Congratulations, you've won!" }
Output: { "prediction": "Spam", "confidence": 0.97 }


---

## 🌐 Deploy to Hugging Face Spaces (FastAPI + Gradio)

Create `app.py` for HF Spaces with `Gradio` UI:

```python
import gradio as gr
import joblib
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Register AttentionLayer
from tensorflow.keras.utils import get_custom_objects
get_custom_objects().update({'AttentionLayer': AttentionLayer})

tokenizer = joblib.load("tokenizer.pkl")
model = load_model("spam_lstm_model.h5")

def predict(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100)
    prob = model.predict(padded)[0][0]
    return {"Spam 🚫": float(prob), "Not Spam ✅": float(1 - prob)}

iface = gr.Interface(fn=predict, inputs="text", outputs="label", title="Email Spam Detection")
iface.launch()
