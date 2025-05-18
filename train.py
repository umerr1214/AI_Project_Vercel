import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os
import torch.quantization

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
            nn.AdaptiveAvgPool2d((4, 4))  # Adaptive pooling for fixed output size
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(32 * 4 * 4, num_classes)  # Simplified classifier
        )
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.dequant(x)
        return x

def train_model():
    try:
        print("Starting model training...")
        
        # Data transforms with even smaller image size
        data_transforms = transforms.Compose([
            transforms.Resize((96, 96)),  # Further reduced from 128x128
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load datasets
        try:
            train_dataset = ImageFolder(os.path.join('fruit_dataset', 'train'), transform=data_transforms)
            test_dataset = ImageFolder(os.path.join('fruit_dataset', 'test'), transform=data_transforms)
            print(f"Found {len(train_dataset.classes)} classes: {train_dataset.classes}")
        except Exception as e:
            print(f"Error loading datasets: {str(e)}")
            print("Please make sure the fruit_dataset directory exists with train and test subdirectories.")
            return

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Initialize the model
        num_classes = len(train_dataset.classes)
        model = MicroFruitClassifier(num_classes)

        # Prepare for static quantization
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)

        # Save class mapping
        class_to_idx = train_dataset.class_to_idx
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        torch.save(idx_to_class, 'class_mapping.pth', _use_new_zipfile_serialization=False)  # Use old format
        print("Saved class mapping")

        # Training parameters
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 5

        # Training loop
        best_accuracy = 0.0
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            epoch_accuracy = 100 * correct / total
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Loss: {running_loss/len(train_loader):.4f}')
            print(f'Training Accuracy: {epoch_accuracy:.2f}%')

            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            print(f'Accuracy on test set: {accuracy:.2f}%')
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                # Convert to static quantized model
                quantized_model = torch.quantization.convert(model.eval(), inplace=False)
                torch.save(quantized_model.state_dict(), 'fruit_classifier.pth', _use_new_zipfile_serialization=False)  # Use old format
                print(f"Saved quantized model with accuracy: {accuracy:.2f}%")

        print("Training completed!")
        print(f"Best accuracy achieved: {best_accuracy:.2f}%")
        
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")

if __name__ == "__main__":
    train_model() 