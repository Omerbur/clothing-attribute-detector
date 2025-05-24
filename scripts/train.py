import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from utils.dataloader import ClothingDataset
import os

# 🔧 Configuration
ATTRIBUTES = ['Red', 'Blue', 'Striped', 'Solid', 'LongSleeve', 'ShortSleeve']
CSV_PATH = "/content/drive/MyDrive/clothing_project/labels.csv"
IMAGE_DIR = "/content/drive/MyDrive/clothing_project/images"
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4

# 📦 Data loading
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = ClothingDataset(CSV_PATH, IMAGE_DIR, ATTRIBUTES, transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 🧠 Model setup
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Sequential(
    nn.Linear(model.classifier[1].in_features, len(ATTRIBUTES)),
    nn.Sigmoid()
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 🧪 Training setup
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# 🏋️ Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs)
        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

# 💾 Save model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/best_model.pth")
