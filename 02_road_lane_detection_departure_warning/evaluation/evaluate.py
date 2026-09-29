import cv2
import json
import glob
import os
import numpy as np

IMAGE_DIR = "dataset/images"
RESULT_DIR = "results"
JSON_FILE = "dataset/label_data_0313.json"

TARGET_Y = 650
ERROR_LIMIT = 50


def get_gt_lane_center(annotation):

    h_samples = annotation["h_samples"]
    lanes = annotation["lanes"]

    # Find ground-truth sample closest to y = 650
    distances = [
        abs(y - TARGET_Y)
        for y in h_samples
    ]

    index = int(np.argmin(distances))

    valid_x = []

    for lane in lanes:

        x = lane[index]

        if x >= 0:
            valid_x.append(float(x))

    if len(valid_x) < 2:
        return None

    # TuSimple images are 1280 pixels wide
    image_center = 640

    left = [
        x for x in valid_x
        if x < image_center
    ]

    right = [
        x for x in valid_x
        if x > image_center
    ]

    if not left or not right:
        return None

    left_x = max(left)
    right_x = min(right)

    return (left_x + right_x) / 2.0


def get_predicted_lane_center(image):

    height, width = image.shape[:2]

    # Blue line drawn by our detector
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    blue = cv2.inRange(
        hsv,
        np.array([100, 100, 100]),
        np.array([140, 255, 255])
    )

    # Look ONLY near the same y-coordinate
    # used for ground truth.
    y_start = max(
        0,
        TARGET_Y - 5
    )

    y_end = min(
        height,
        TARGET_Y + 6
    )

    region = blue[
        y_start:y_end,
        :
    ]

    ys, xs = np.where(
        region > 0
    )

    if len(xs) < 5:
        return None

    # At y = 650, the blue centre line
    # should have only a small number of x values.
    return float(
        np.median(xs)
    )


def main():

    # ==========================================
    # Load TuSimple annotations
    # ==========================================

    annotations = {}

    with open(
        JSON_FILE,
        "r"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            item = json.loads(line)

            raw_file = item["raw_file"]

            parts = raw_file.split("/")

            # Example:
            # clips/0313-1/6040/20.jpg
            #
            # Converted filename:
            # 6040_20.jpg

            filename = (
                parts[-2]
                + "_"
                + parts[-1]
            )

            annotations[filename] = item

    print(
        "Loaded annotations:",
        len(annotations)
    )

    # ==========================================
    # Result images
    # ==========================================

    result_files = sorted(
        glob.glob(
            os.path.join(
                RESULT_DIR,
                "*.jpg"
            )
        )
    )

    total = len(result_files)

    evaluated = 0
    successful = 0
    incomplete = 0

    errors = []

    safe = 0
    departure_left = 0
    departure_right = 0

    # ==========================================
    # Evaluate each image
    # ==========================================

    for result_path in result_files:

        filename = os.path.basename(
            result_path
        )

        if filename not in annotations:
            continue

        result = cv2.imread(
            result_path
        )

        if result is None:
            continue

        gt_center = get_gt_lane_center(
            annotations[filename]
        )

        if gt_center is None:
            continue

        evaluated += 1

        pred_center = get_predicted_lane_center(
            result
        )

        # Detector did not produce
        # a centre line
        if pred_center is None:

            incomplete += 1
            continue

        successful += 1

        # ======================================
        # Lane centre error
        # ======================================

        error = abs(
            pred_center -
            gt_center
        )

        errors.append(
            error
        )

        # ======================================
        # Warning classification
        # ======================================

        offset = (
            pred_center - 640
        )

        if offset < -80:

            departure_left += 1

        elif offset > 80:

            departure_right += 1

        else:

            safe += 1

    # ==========================================
    # Calculate metrics
    # ==========================================

    if errors:

        mae = float(
            np.mean(errors)
        )

        median_error = float(
            np.median(errors)
        )

        accuracy_50px = (
            sum(
                e <= ERROR_LIMIT
                for e in errors
            )
            / len(errors)
            * 100
        )

    else:

        mae = 0.0
        median_error = 0.0
        accuracy_50px = 0.0

    if evaluated:

        completeness = (
            successful /
            evaluated *
            100
        )

    else:

        completeness = 0.0

    # ==========================================
    # Display results
    # ==========================================

    print()
    print("=" * 55)
    print("TU-SIMPLE LANE DETECTION EVALUATION")
    print("=" * 55)

    print(
        f"Result images          : {total}"
    )

    print(
        f"GT-matched images      : {evaluated}"
    )

    print(
        f"Successful detections  : {successful}"
    )

    print(
        f"Incomplete detections  : {incomplete}"
    )

    print(
        f"Detection completeness : "
        f"{completeness:.2f}%"
    )

    print(
        f"Mean Absolute Error    : "
        f"{mae:.2f} pixels"
    )

    print(
        f"Median Error           : "
        f"{median_error:.2f} pixels"
    )

    print(
        f"Accuracy <= 50 pixels  : "
        f"{accuracy_50px:.2f}%"
    )

    print()
    print("Predicted status:")

    print(
        f"LANE SAFE              : {safe}"
    )

    print(
        f"DEPARTURE LEFT         : "
        f"{departure_left}"
    )

    print(
        f"DEPARTURE RIGHT        : "
        f"{departure_right}"
    )

    print("=" * 55)

    # ==========================================
    # Save metrics
    # ==========================================

    with open(
        "evaluation/metrics.txt",
        "w"
    ) as f:

        f.write(
            "TuSimple Lane Detection Evaluation\n"
        )

        f.write(
            "===================================\n"
        )

        f.write(
            f"Result images: {total}\n"
        )

        f.write(
            f"GT-matched images: {evaluated}\n"
        )

        f.write(
            f"Successful detections: {successful}\n"
        )

        f.write(
            f"Incomplete detections: {incomplete}\n"
        )

        f.write(
            f"Detection completeness: "
            f"{completeness:.2f}%\n"
        )

        f.write(
            f"Mean Absolute Error: "
            f"{mae:.2f} pixels\n"
        )

        f.write(
            f"Median Error: "
            f"{median_error:.2f} pixels\n"
        )

        f.write(
            f"Accuracy <= 50 pixels: "
            f"{accuracy_50px:.2f}%\n"
        )

        f.write("\nPredicted status:\n")

        f.write(
            f"LANE SAFE: {safe}\n"
        )

        f.write(
            f"DEPARTURE LEFT: "
            f"{departure_left}\n"
        )

        f.write(
            f"DEPARTURE RIGHT: "
            f"{departure_right}\n"
        )

    print()
    print(
        "Metrics saved to "
        "evaluation/metrics.txt"
    )


if __name__ == "__main__":
    main()
