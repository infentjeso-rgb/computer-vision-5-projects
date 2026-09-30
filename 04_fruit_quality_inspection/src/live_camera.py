import cv2
import torch
from torchvision import transforms, models
from torch import nn
from pathlib import Path
from collections import deque

# ==========================================
# MODEL
# ==========================================

MODEL_PATH = Path("models/fruit_quality_model.pth")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

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

# ==========================================
# IMAGE TRANSFORMATION
# ==========================================

transform = transforms.Compose([
    transforms.ToPILImage(),

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

# ==========================================
# PREDICTION SMOOTHING
# ==========================================

prediction_history = deque(maxlen=10)

# ==========================================
# CAMERA LOOP
# ==========================================

print()
print("========================================")
print("FRUIT QUALITY INSPECTION SYSTEM")
print("========================================")
print("Place the fruit inside the green box.")
print("Press G to save GOOD screenshot.")
print("Press D to save DEFECTIVE screenshot.")
print("Press G to save GOOD screenshot.")
print("Press D to save DEFECTIVE screenshot.")
print("Press Q to quit.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera.")
        break

    height, width = frame.shape[:2]

    # ======================================
    # CENTRAL INSPECTION REGION
    # ======================================

    box_width = int(width * 0.55)
    box_height = int(height * 0.65)

    x1 = (width - box_width) // 2
    y1 = (height - box_height) // 2

    x2 = x1 + box_width
    y2 = y1 + box_height

    # Extract inspection region
    roi = frame[y1:y2, x1:x2]

    # ======================================
    # MODEL PREDICTION
    # ======================================

    rgb = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2RGB
    )

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

    current_label = classes[prediction.item()]
    current_confidence = confidence.item()

    # Store prediction
    prediction_history.append(
        (
            current_label,
            current_confidence
        )
    )

    # ======================================
    # SMOOTH PREDICTIONS
    # ======================================

    good_score = 0.0
    defective_score = 0.0

    for label, conf in prediction_history:

        if label == "good":
            good_score += conf
        else:
            defective_score += conf

    if good_score >= defective_score:
        final_label = "good"
        total_score = good_score
    else:
        final_label = "defective"
        total_score = defective_score

    average_confidence = (
        total_score / len(prediction_history)
    ) * 100

    # ======================================
    # CONFIDENCE CHECK
    # ======================================

    if average_confidence < 60:

        display_label = "CHECK FRUIT POSITION"
        text_color = (0, 165, 255)

    elif final_label == "good":

        display_label = "GOOD / FRESH"
        text_color = (0, 255, 0)

    else:

        display_label = "DEFECTIVE / ROTTEN"
        text_color = (0, 0, 255)

    # ======================================
    # DRAW INSPECTION BOX
    # ======================================

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3
    )

    # Corner markers
    marker = 25

    # Top-left
    cv2.line(
        frame,
        (x1, y1),
        (x1 + marker, y1),
        (0, 255, 0),
        5
    )

    cv2.line(
        frame,
        (x1, y1),
        (x1, y1 + marker),
        (0, 255, 0),
        5
    )

    # Top-right
    cv2.line(
        frame,
        (x2, y1),
        (x2 - marker, y1),
        (0, 255, 0),
        5
    )

    cv2.line(
        frame,
        (x2, y1),
        (x2, y1 + marker),
        (0, 255, 0),
        5
    )

    # Bottom-left
    cv2.line(
        frame,
        (x1, y2),
        (x1 + marker, y2),
        (0, 255, 0),
        5
    )

    cv2.line(
        frame,
        (x1, y2),
        (x1, y2 - marker),
        (0, 255, 0),
        5
    )

    # Bottom-right
    cv2.line(
        frame,
        (x2, y2),
        (x2 - marker, y2),
        (0, 255, 0),
        5
    )

    cv2.line(
        frame,
        (x2, y2),
        (x2, y2 - marker),
        (0, 255, 0),
        5
    )

    # ======================================
    # TOP INFORMATION PANEL
    # ======================================

    cv2.rectangle(
        frame,
        (10, 10),
        (width - 10, 125),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        "FRUIT QUALITY INSPECTION",
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Prediction: {display_label}",
        (25, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        text_color,
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {average_confidence:.2f}%",
        (25, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # ======================================
    # INSPECTION INSTRUCTION
    # ======================================

    cv2.putText(
        frame,
        "Place fruit inside inspection box",
        (x1 + 10, y2 + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2
    )

    # ======================================
    # DISPLAY
    # ======================================

    cv2.imshow(
        "Fruit Quality Inspection",
        frame
    )

    # ======================================
    # KEYBOARD CONTROLS
    # ======================================

    key = cv2.waitKey(1) & 0xFF

    # Save GOOD screenshot
    if key == ord("g"):
        path = "screenshots/live_good_prediction.jpg"
        cv2.imwrite(path, frame)
        print(f"GOOD screenshot saved: {path}")

    # Save DEFECTIVE screenshot
    elif key == ord("d"):
        path = "screenshots/live_defective_prediction.jpg"
        cv2.imwrite(path, frame)
        print(f"DEFECTIVE screenshot saved: {path}")

    # Quit
    elif key == ord("q"):
        break

# ==========================================
# CLEANUP
# ==========================================

cap.release()
cv2.destroyAllWindows()

print("Camera closed.")
