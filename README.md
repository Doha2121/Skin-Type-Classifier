# 🧴 Skin Type Classifier

A deep learning–based web app that classifies human **skin type (Dry, Oily, Normal, or Sensitive)** from an uploaded or live-captured image.  
Built using **ResNet-50**, **FastAPI**, and **Streamlit**, and deployed on [Hugging Face Spaces](https://huggingface.co/spaces/Doha000/Skin_Type_Classifier).

---

## 🚀 Live Demo

👉 **Try it here:** [Hugging Face Space – Skin Type Classifier](https://huggingface.co/spaces/Doha000/Skin_Type_Classifier)

Upload a skin image or take one using your webcam, and the app will instantly predict your skin type.

---

## 🧠 Project Overview

This project leverages a **ResNet-50** model fine-tuned on a custom dataset of facial skin patches to classify:
- 🧼 **Dry Skin**
- 💧 **Oily Skin**
- 🌿 **Normal Skin**
  
The model was trained using **PyTorch** and integrated into a **FastAPI** backend, then wrapped in a **Streamlit** frontend for easy, interactive use.

Dataset Link : https://universe.roboflow.com/skins-aup8m/skin-type-l6qra/dataset/10
---

## ⚙️ Tech Stack

| Category | Tools & Libraries |
|-----------|------------------|
| **Frontend** | Streamlit |
| **Backend** | FastAPI |
| **Deep Learning** | PyTorch, torchvision |
| **Model** | ResNet-50 (fine-tuned) |
| **Deployment** | Hugging Face Spaces |
| **Other** | Pillow, Requests, NumPy |

---
## 🧩 How It Works

1. **Upload or Capture an Image**
   - The user uploads a facial skin image or captures one using the webcam.
2. **Preprocessing**
   - The image is resized, normalized, and transformed into a tensor.
3. **Model Prediction**
   - The ResNet-50 model outputs probabilities for each skin type.
4. **Result Display**
   - The predicted skin type is shown with a confidence score.
## 🧰 Installation & Local Setup

To run locally:

```bash
# 1️⃣ Clone the repository
git clone https://github.com/Doha000/Skin_Type_Classifier.git
cd Skin_Type_Classifier

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Run the FastAPI backend
cd app
uvicorn main:app --reload

# 4️⃣ Run the Streamlit frontend (in another terminal)
streamlit run skin_app.py
Your local app will be available at http://localhost:8501

