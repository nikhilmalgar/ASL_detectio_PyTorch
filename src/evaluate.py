# src/evaluate.py

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import os

# ✅ Correct import of your model
from model import ASLClassifier

# ✅ Path to test data
TEST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'asl_alphabet_test'))
print("✅ Test data path:", TEST_DIR)
print("📁 Directory exists?", os.path.exists(TEST_DIR))

# ✅ Data transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ✅ Load test dataset
test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
CLASS_NAMES = test_dataset.classes
print("✅ Classes:", CLASS_NAMES)

# ✅ Load model
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'asl_model.pth'))
model = ASLClassifier(num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# ✅ Evaluate
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"✅ Test Accuracy: {accuracy:.2f}%")
