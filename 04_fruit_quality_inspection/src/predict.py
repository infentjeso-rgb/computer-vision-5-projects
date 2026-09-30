import cv2
import torch
from torchvision import transforms, models
from torch import nn
from pathlib import Path

# ==============================
# PATHS
# ==============================

MODEL_PATH = Path("models/fruit_quality_model.pth")
IMAGE_PATH = Path("dataset_split/test/good")

RESULT_PATH = Path("results/fruit_quality_result.jpg")
SCREENSHOT_PATH = Path("screenshots/fruit_good_prediction.jpg")

# Create folders if needed
RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
SCREENSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ==============================
# DEVICE
# ==============================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# ==============================
# LOAD MODEL
# ==============================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

classes = checkpoint["classes"]

model = models.mobilenet_v3_small(
    weights=None
)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    2
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()

print("Classes:", classes)

# ==============================
# IMAGE TRANSFORMATION
# ==============================

transform = transforms.Compose([
    transforms.ToPILImage(),

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ==============================
# FIND DEFECTIVE TEST IMAGE
# ==============================

test_images = []

for ext in ["*.jpg", "*.jpeg", "*.png"]:
    test_images.extend(IMAGE_PATH.glob(ext))

if not test_images:
    print("No defective test images found.")
    exit()

# Use the first defective test image
image_path = test_images[0]

print("Image:", image_path)

# ==============================
# READ IMAGE
# ==============================

image = cv2.imread(str(image_path))

if image is None:
    print("Could not read image.")
    exit()

rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

# ==============================
# PREDICTION
# ==============================

input_tensor = transform(
    rgb
).unsqueeze(0).to(device)

with torch.no_grad():

    output = model(input_tensor)

    probabilities = torch.softmax(
        output,
        dim=1
    )

confidence, prediction = torch.max(
    probabilities,
    dim=1
)

label = classes[prediction.item()]

confidence = confidence.item() * 100

# ==============================
# DISPLAY TEXT
# ==============================

display_label = label.upper()

if label == "good":
    text_color = (0, 255, 0)
else:
    text_color = (0, 0, 255)

cv2.rectangle(
    image,
    (15, 15),
    (650, 115),
    (0, 0, 0),
    -1
)

cv2.putText(
    image,
    f"FRUIT QUALITY: {display_label}",
    (30, 55),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.9,
    text_color,
    2
)

cv2.putText(
    image,
    f"Confidence: {confidence:.2f}%",
    (30, 95),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.75,
    (255, 255, 255),
    2
)

# ==============================
# SAVE RESULT
# ==============================

cv2.imwrite(
    str(RESULT_PATH),
    image
)

cv2.imwrite(
    str(SCREENSHOT_PATH),
    image
)

# ==============================
# PRINT RESULT
# ==============================

print()
print("================================")
print("FRUIT QUALITY PREDICTION")
print("================================")
print("Image:", image_path)
print("Prediction:", display_label)
print(f"Confidence: {confidence:.2f}%")
print("Result saved:", RESULT_PATH)
print("Screenshot saved:", SCREENSHOT_PATH)
