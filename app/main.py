# app/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

# ----------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------
# Classes (make sure these match training order!)
SKIN_CLASSES = ['dry', 'normal', 'oily', 'sensitive']

# Path to your trained model
MODEL_FILE = r"D:\PREGrad reseach\skin_classifier_api\model\resnet50984_skin_model.pth"

# Normalization and resize must match training/validation
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ----------------------------------------------------
# APP INITIALIZATION
# ----------------------------------------------------
app = FastAPI()
MODEL = None
DEVICE = torch.device("cpu")

# ----------------------------------------------------
# MODEL LOADING FUNCTION
# ----------------------------------------------------
def get_model():
    """Load ResNet50 model and trained weights."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(
        nn.Dropout(0.6),
        nn.Linear(model.fc.in_features, len(SKIN_CLASSES))
    )

    state_dict = torch.load(MODEL_FILE, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ----------------------------------------------------
# IMAGE PREPROCESSING
# ----------------------------------------------------
def get_inference_transform():
    """Same preprocessing as used during validation in training."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])


# ----------------------------------------------------
# FASTAPI EVENTS
# ----------------------------------------------------
@app.on_event("startup")
async def startup_event():
    global MODEL
    MODEL = get_model()
    print("✅ Model loaded successfully!")


# ----------------------------------------------------
# ENDPOINTS
# ----------------------------------------------------
@app.get("/")
def read_root():
    return {"message": "✅ PyTorch ResNet50 Skin Classification API is running!"}


@app.post("/classify_skin")
async def classify_image(file: UploadFile = File(...), debug: bool = False):
    """Classify an uploaded skin image."""
    if MODEL is None:
        return JSONResponse(status_code=500, content={"error": "Model not available"})

    try:
        # 1️⃣ Read and convert image to RGB
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 2️⃣ Apply preprocessing
        transform = get_inference_transform()
        img_tensor = transform(image)
        input_batch = img_tensor.unsqueeze(0).to(DEVICE)

        # 3️⃣ Model inference
        with torch.no_grad():
            output = MODEL(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # 4️⃣ Get predictions
        confidence, predicted_index = torch.max(probabilities, 0)
        predicted_class = SKIN_CLASSES[predicted_index.item()]

        # 5️⃣ Build response
        response = {
            "filename": file.filename,
            "prediction": predicted_class
        }

        # Optional debug info
        if debug:
            topk = torch.topk(probabilities, k=min(3, len(SKIN_CLASSES)))
            response["top3"] = [
                {"class": SKIN_CLASSES[i.item()], "prob": round(p.item(), 4)}
                for p, i in zip(topk.values, topk.indices)
            ]
            response["tensor_stats"] = {
                "shape": list(img_tensor.shape),
                "min": round(float(img_tensor.min()), 4),
                "max": round(float(img_tensor.max()), 4),
                "mean": round(float(img_tensor.mean()), 4),
            }

        return JSONResponse(content=response)

    except Exception as e:
        print("❌ Prediction Error:", e)
        return JSONResponse(status_code=500, content={"error": f"Processing failed: {str(e)}"})
