# Skin Type Classifier

**AI web app that instantly classifies human skin type (Dry, Oily, Normal, Sensitive) from a photo or webcam.**

Built with **PyTorch (ResNet-50)**, **FastAPI**, and **Streamlit**. Fully deployed and ready for real clients.

👉 **[Live Demo – Try it now!](https://huggingface.co/spaces/Doha000/Skin_Type_Classifier)**

---

## 🎯 Why This Project Matters (Business Value)
- Perfect for **beauty shops, cosmetics stores, dermatology apps, and skincare e-commerce** in Egypt.
- Clients can use it to recommend products based on skin type automatically.
- High demand on Nafezly & Kafiil: “أداة AI تحدد نوع البشرة وتقترح منتجات”.

---

## ✨ Features
- Upload photo or use webcam
- Real-time prediction with confidence score
- Clean, modern Streamlit UI
- FastAPI backend (ready to scale)
- Deployed on Hugging Face Spaces

---

## 🛠 Tech Stack

| Category       | Tools                              |
|----------------|------------------------------------|
| Deep Learning  | PyTorch, torchvision, ResNet-50    |
| Backend        | FastAPI                            |
| Frontend       | Streamlit                          |
| Deployment     | Hugging Face Spaces                |
| Others         | Pillow, NumPy, Requests            |

---

## 📊 Dataset
Custom dataset from Roboflow: [Skin Type Dataset](https://universe.roboflow.com/skins-aup8m/skin-type-l6qra/dataset/10)

---

## 🚀 Live Demo
[Click here to test the app](https://huggingface.co/spaces/Doha000/Skin_Type_Classifier)

---


## How to Run Locally
```bash
git clone https://github.com/Doha2121/Skin-Type-Classifier.git
cd Skin-Type-Classifier
pip install -r requirements.txt

# Terminal 1: Run backend
cd app
uvicorn main:app --reload

# Terminal 2: Run frontend
streamlit run skin_app.py
