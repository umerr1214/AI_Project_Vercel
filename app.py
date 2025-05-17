from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import os

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Check if model files exist
MODEL_PATH = 'fruit_classifier.pth'
MAPPING_PATH = 'class_mapping.pth'

if not os.path.exists(MODEL_PATH) or not os.path.exists(MAPPING_PATH):
    print("Model files not found. Please run train.py first to train the model.")

try:
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)  # Initialize without pre-trained weights
    num_classes = 10  # Number of fruit classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    
    # Load class mapping
    if os.path.exists(MAPPING_PATH):
        idx_to_class = torch.load(MAPPING_PATH)
    else:
        idx_to_class = {}
except Exception as e:
    print(f"Error loading model: {str(e)}")

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open('static/index.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(MAPPING_PATH):
        raise HTTPException(
            status_code=500,
            detail="Model not found. Please train the model first by running train.py"
        )
    
    try:
        # Read and transform image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class = idx_to_class[predicted.item()]
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence = probabilities[predicted].item()
        
        return {
            "predicted_class": predicted_class,
            "confidence": float(confidence)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        ) 