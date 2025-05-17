import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os

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

        # Load pre-trained ResNet model
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_classes = len(train_dataset.classes)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(device)

        # Save class mapping
        class_to_idx = train_dataset.class_to_idx
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        torch.save(idx_to_class, 'class_mapping.pth')
        print("Saved class mapping")

        # Training parameters
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 10

        # Training loop
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
                print(f'Accuracy on test set: {100 * correct / total:.2f}%')

        # Save the model
        torch.save(model.state_dict(), 'fruit_classifier.pth')
        print("Training completed and model saved!")
        
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")

if __name__ == "__main__":
    train_model() 