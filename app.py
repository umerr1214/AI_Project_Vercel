from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import torch
import torch.nn as nn
import torch.quantization
from torchvision import transforms
from PIL import Image
import io
import os

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class MicroFruitClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MicroFruitClassifier, self).__init__()
        
        self.quant = torch.quantization.QuantStub()
        self.features = nn.Sequential(
            DepthwiseSeparableConv(3, 8),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),
            
            DepthwiseSeparableConv(8, 16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            DepthwiseSeparableConv(16, 32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(32 * 4 * 4, num_classes)
        )
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.dequant(x)
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
    # Initialize the model with quantization
    model = MicroFruitClassifier(num_classes=10)
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model = torch.quantization.prepare(model)
    model = torch.quantization.convert(model)
    
    # Load the state dict
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    
    # Load class mapping
    idx_to_class = torch.load(MAPPING_PATH)
except Exception as e:
    print(f"Error loading model: {str(e)}")

# Image transforms with smaller size
transform = transforms.Compose([
    transforms.Resize((96, 96)),
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
        image_tensor = transform(image).unsqueeze(0)
        
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