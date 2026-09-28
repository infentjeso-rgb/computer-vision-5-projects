from ultralytics import YOLO
import cv2
import time

# Load the pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Open the laptop webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

# Objects treated as obstacles by our simulated robot
obstacle_classes = {
    "person",
    "car",
    "bus",
    "bicycle",
    "motorcycle"
}

previous_time = time.time()

print("Starting Real-Time Object Detection...")
print("Press 'q' to quit.")

while True:

    # Capture one frame from the webcam
    success, frame = cap.read()

    if not success:
        print("ERROR: Could not read webcam frame.")
        break

    # Run YOLO object detection
    results = model(frame, verbose=False)

    # Default robot command
    robot_command = "MOVE"

    detected_obstacle = "None"
    best_confidence = 0.0

    # Process every detected object
    for result in results:

        for box in result.boxes:

            confidence = float(box.conf[0])

            # Ignore weak detections
            if confidence < 0.50:
                continue

            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            # Get bounding-box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Show object name and confidence
            label = f"{class_name} {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # Robotics decision logic
            if class_name in obstacle_classes:

                # Find the centre of the detected object
                object_center_x = (x1 + x2) // 2

                # Get camera width
                frame_width = frame.shape[1]

                # Use the strongest obstacle detection
                if confidence > best_confidence:

                    best_confidence = confidence
                    detected_obstacle = class_name

                    # Object on left side
                    if object_center_x < frame_width / 3:
                        robot_command = "AVOID RIGHT"

                    # Object on right side
                    elif object_center_x > (2 * frame_width / 3):
                        robot_command = "AVOID LEFT"

                    # Object in centre
                    else:
                        robot_command = "STOP"

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    # Draw LEFT/CENTRE/RIGHT zone lines
    frame_width = frame.shape[1]

    cv2.line(
        frame,
        (frame_width // 3, 0),
        (frame_width // 3, frame.shape[0]),
        (255, 255, 0),
        2
    )

    cv2.line(
        frame,
        (2 * frame_width // 3, 0),
        (2 * frame_width // 3, frame.shape[0]),
        (255, 255, 0),
        2
    )

    # Display robot command
    cv2.putText(
        frame,
        f"ROBOT: {robot_command}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3
    )

    # Display detected obstacle
    cv2.putText(
        frame,
        f"Obstacle: {detected_obstacle}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Display FPS
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Show the camera output
    cv2.imshow("Robotic Object Detection - YOLO11", frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release webcam
cap.release()

# Close OpenCV windows
cv2.destroyAllWindows()
