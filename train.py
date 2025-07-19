# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from src.dataset import get_data_loaders
from src.model import ASLClassifier

# 🔧 Config
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
DATA_DIR = 'data/asl_alphabet_train'
NUM_CLASSES = 29
MODEL_PATH = "asl_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📦 Load Data
train_loader, val_loader, class_names, num_classes = get_data_loaders(DATA_DIR, batch_size=BATCH_SIZE)


# 🧠 Initialize Model
model = ASLClassifier(num_classes=NUM_CLASSES).to(DEVICE)

# 🧮 Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 🚂 Training Loop
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    acc = 100 * total_correct / total_samples
    print(f"✅ Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {running_loss:.4f}, Accuracy: {acc:.2f}%")

# 💾 Save Model
torch.save(model.state_dict(), MODEL_PATH)
print(f"\n🎉 Model saved to {MODEL_PATH}")
