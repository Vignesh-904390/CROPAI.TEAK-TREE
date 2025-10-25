import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader

# ===== Paths =====
data_dir = "dataset/teak_train"                 # <<--- CHANGED
model_path = "model/teak_effnet.pth"         # <<--- CHANGED
os.makedirs("model", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📦 Using device: {device}")

# ===== Augmentation =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

print("📁 Loading TEAK dataset...")   # <<--- CHANGED
dataset = datasets.ImageFolder(data_dir, transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True)
num_classes = len(dataset.classes)
print(f"✅ {len(dataset)} TEAK images | Classes: {dataset.classes}")   # <<--- CHANGED

# ===== EfficientNet Model =====
print("🧠 Loading EfficientNet-B0...")
model = models.efficientnet_b0(pretrained=True)

# Freeze feature extractor
for param in model.features.parameters():
    param.requires_grad = False

model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ===== Training =====
print("🚀 Training on TEAK leaves...")   # <<--- CHANGED
epochs = 200
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss:.4f}")

# ===== Save Model =====
torch.save({'model_state_dict': model.state_dict(),
            'class_names': dataset.classes}, model_path)

print("✅ TEAK model training complete & saved at:", model_path)   # <<--- CHANGED
