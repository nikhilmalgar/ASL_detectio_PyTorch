# src/dataset.py

import os
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

def get_data_loaders(data_dir, img_size=64, batch_size=64, val_split=0.2):
    # Define image transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    
    # Load the dataset
    full_dataset = ImageFolder(root=data_dir, transform=transform)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    
    # Train/Val Split
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, class_names, num_classes
