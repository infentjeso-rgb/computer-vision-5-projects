import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.utils.data import DataLoader
from pathlib import Path

DATASET = Path("dataset_split")
MODEL_PATH = "models/fruit_quality_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

train_data = datasets.ImageFolder(DATASET / "train", transform=transform)
val_data = datasets.ImageFolder(DATASET / "val", transform=val_transform)

train_loader = DataLoader(
    train_data,
    batch_size=32,
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_data,
    batch_size=32,
    shuffle=False,
    num_workers=2
)

print("Classes:", train_data.classes)
print("Training images:", len(train_data))
print("Validation images:", len(val_data))

model = models.mobilenet_v3_small(
    weights=models.MobileNet_V3_Small_Weights.DEFAULT
)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    2
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=0.0005
)

EPOCHS = 8

best_accuracy = 0.0

for epoch in range(EPOCHS):

    model.train()

    train_correct = 0
    train_total = 0
    train_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)

    train_accuracy = 100 * train_correct / train_total

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_accuracy = 100 * val_correct / val_total

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {train_loss/len(train_loader):.4f} | "
        f"Train: {train_accuracy:.2f}% | "
        f"Val: {val_accuracy:.2f}%"
    )

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "classes": train_data.classes
            },
            MODEL_PATH
        )

        print("  Saved best model.")

print()
print("Training complete.")
print(f"Best validation accuracy: {best_accuracy:.2f}%")
print(f"Model saved to: {MODEL_PATH}")
