# test_dataloader.py

from src.dataset import get_data_loaders

# Call the function from dataset.py
train_loader, val_loader, class_names, num_classes = get_data_loaders(
    data_dir="data/asl_alphabet_train",  # ✅ Your correct path
    img_size=64,
    batch_size=32
)

# Print outputs
print("✅ Classes:", class_names)
print("Total Classes:", num_classes)

# Preview one batch
for images, labels in train_loader:
    print("🔍 Image batch shape:", images.shape)
    print("🔍 Label batch shape:", labels.shape)
    break
