import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os

# Define a lightweight custom model
class LightweightFruitClassifier(nn.Module):
    def __init__(self, num_classes):
        super(LightweightFruitClassifier, self).__init__()
        
        # Smaller number of filters and layers
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

def train_model():
    try:
        print("Starting model training...")
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Data transforms
        data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
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

        # Initialize the lightweight model
        num_classes = len(train_dataset.classes)
        model = LightweightFruitClassifier(num_classes).to(device)

        # Save class mapping
        class_to_idx = train_dataset.class_to_idx
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        
        # Save with optimization for size
        torch.save(idx_to_class, 'class_mapping.pth', _use_new_zipfile_serialization=True)
        print("Saved class mapping")

        # Training parameters
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.00001)
        num_epochs = 5  # Increased epochs for better accuracy with smaller model

        # Training loop
        best_accuracy = 0.0
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
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
            if (epoch + 1) % 2 == 0:
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                
                accuracy = 100 * correct / total
                print(f'Accuracy on test set: {accuracy:.2f}%')
                
                # Save best model
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    # Save with optimization for size
                    torch.save(model.state_dict(), 'fruit_classifier.pth', _use_new_zipfile_serialization=True)
                    print(f"Saved better model with accuracy: {accuracy:.2f}%")

        print("Training completed!")
        print(f"Best accuracy achieved: {best_accuracy:.2f}%")
        
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")

if __name__ == "__main__":
    train_model() 