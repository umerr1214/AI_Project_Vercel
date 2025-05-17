from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import os

# Define the same model architecture as in train.py
class LightweightFruitClassifier(nn.Module):
    def __init__(self, num_classes):
        super(LightweightFruitClassifier, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

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
    model = LightweightFruitClassifier(num_classes=10)  # Number of fruit classes
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    # Load class mapping
    idx_to_class = torch.load(MAPPING_PATH)
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
            detail="Model not found. Please run train.py first to train the model"
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