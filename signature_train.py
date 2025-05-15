import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

real_dir = "data/real_aug"
fake_dir = "data/fake_aug"
combined_dir = "data/combined_dataset"
train_dir = os.path.join(combined_dir, "train")
val_dir = os.path.join(combined_dir, "val")

def prepare_dataset():
    if os.path.exists(combined_dir):
        shutil.rmtree(combined_dir)

    os.makedirs(os.path.join(train_dir, "real"))
    os.makedirs(os.path.join(train_dir, "fake"))
    os.makedirs(os.path.join(val_dir, "real"))
    os.makedirs(os.path.join(val_dir, "fake"))

    real_images = [os.path.join(real_dir, img) for img in os.listdir(real_dir)]
    fake_images = [os.path.join(fake_dir, img) for img in os.listdir(fake_dir)]

    real_train, real_val = train_test_split(real_images, test_size=0.2, random_state=42)
    fake_train, fake_val = train_test_split(fake_images, test_size=0.2, random_state=42)

    for img_path in real_train:
        shutil.copy(img_path, os.path.join(train_dir, "real"))
    for img_path in real_val:
        shutil.copy(img_path, os.path.join(val_dir, "real"))
    for img_path in fake_train:
        shutil.copy(img_path, os.path.join(train_dir, "fake"))
    for img_path in fake_val:
        shutil.copy(img_path, os.path.join(val_dir, "fake"))

prepare_dataset()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

class SignatureModel(nn.Module):
    def __init__(self):
        super(SignatureModel, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.model(x)

model = SignatureModel().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def train_model(model, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.long()  # Convert labels to long for comparison
                outputs = model(inputs)
                preds = (torch.sigmoid(outputs) > 0.5).squeeze().long()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = 100 * correct / total
        print(f"Validation Accuracy: {acc:.2f}%")

    os.makedirs("saved_models", exist_ok=True)
    model_path = "saved_models/signature_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

train_model(model, train_loader, val_loader, epochs=10)
