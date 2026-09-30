import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

DATASET = Path("dataset_split")
MODEL_PATH = "models/fruit_quality_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

test_data = datasets.ImageFolder(
    DATASET / "test",
    transform=transform
)

test_loader = DataLoader(
    test_data,
    batch_size=32,
    shuffle=False,
    num_workers=2
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model = models.mobilenet_v3_small(weights=None)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    2
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(predictions.cpu().numpy())

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print()
print("================================")
print("FRUIT QUALITY MODEL EVALUATION")
print("================================")

print(f"Test images: {len(test_data)}")
print(f"Test accuracy: {accuracy * 100:.2f}%")

print()
print("Classification Report:")
print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=test_data.classes,
        digits=4
    )
)

print("Confusion Matrix:")
print(confusion_matrix(
    all_labels,
    all_predictions
))
