import cv2
import numpy as np
import os
import glob


INPUT_DIR = "dataset/images"
OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def region_of_interest(image):

    height, width = image.shape[:2]

    mask = np.zeros_like(image)

    polygon = np.array([[
        (0, height),
        (int(width * 0.35), int(height * 0.52)),
        (int(width * 0.65), int(height * 0.52)),
        (width, height)
    ]], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)

    return cv2.bitwise_and(image, mask)


def make_line_points(image, slope, intercept):

    height, width = image.shape[:2]

    if abs(slope) < 0.35:
        return None

    y1 = height
    y2 = int(height * 0.52)

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    if not (-width <= x1 <= 2 * width):
        return None

    if not (-width <= x2 <= 2 * width):
        return None

    return (x1, y1, x2, y2)


def detect_lanes(image):

    height, width = image.shape[:2]

    center_x = width // 2

    # ------------------------------------------------
    # GRAYSCALE
    # ------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # ------------------------------------------------
    # GAUSSIAN BLUR
    # ------------------------------------------------

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # ------------------------------------------------
    # CANNY
    # ------------------------------------------------

    edges = cv2.Canny(
        blur,
        40,
        120
    )

    # ------------------------------------------------
    # WHITE ROAD MARKINGS
    # ------------------------------------------------

    white = cv2.inRange(
        gray,
        150,
        255
    )

    # ------------------------------------------------
    # YELLOW ROAD MARKINGS
    # ------------------------------------------------

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    yellow = cv2.inRange(
        hsv,
        np.array([15, 60, 80]),
        np.array([40, 255, 255])
    )

    # Combine lane information
    combined = cv2.bitwise_or(
        edges,
        white
    )

    combined = cv2.bitwise_or(
        combined,
        yellow
    )

    # ------------------------------------------------
    # REGION OF INTEREST
    # ------------------------------------------------

    roi = region_of_interest(
        combined
    )

    # ------------------------------------------------
    # HOUGH LINE DETECTION
    # ------------------------------------------------

    lines = cv2.HoughLinesP(
        roi,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=25,
        maxLineGap=200
    )

    left_candidates = []
    right_candidates = []

    if lines is not None:

        for line in lines:

            coords = np.asarray(
                line
            ).reshape(-1)

            if len(coords) != 4:
                continue

            x1, y1, x2, y2 = coords.astype(int)

            dx = x2 - x1
            dy = y2 - y1

            if dx == 0:
                continue

            slope = dy / float(dx)

            # Ignore nearly horizontal lines
            if abs(slope) < 0.35:
                continue

            intercept = y1 - slope * x1

            # Position where line reaches bottom
            bottom_x = (
                height - intercept
            ) / slope

            if not np.isfinite(bottom_x):
                continue

            bottom_x = int(bottom_x)

            # ----------------------------------------
            # LEFT SIDE
            # ----------------------------------------

            if slope < -0.35:

                if (
                    bottom_x > -100
                    and bottom_x < center_x - 30
                ):

                    left_candidates.append(
                        (
                            abs(
                                center_x - bottom_x
                            ),
                            slope,
                            intercept,
                            bottom_x
                        )
                    )

            # ----------------------------------------
            # RIGHT SIDE
            # ----------------------------------------

            elif slope > 0.35:

                if (
                    bottom_x > center_x + 30
                    and bottom_x < width + 100
                ):

                    right_candidates.append(
                        (
                            abs(
                                bottom_x - center_x
                            ),
                            slope,
                            intercept,
                            bottom_x
                        )
                    )

    # ------------------------------------------------
    # SELECT BEST LEFT LINE
    # ------------------------------------------------

    left_lane = None

    if left_candidates:

        # Select candidate closest to vehicle centre
        left_candidates.sort(
            key=lambda x: x[0]
        )

        selected = left_candidates[:8]

        slopes = [
            x[1] for x in selected
        ]

        intercepts = [
            x[2] for x in selected
        ]

        slope = np.median(
            slopes
        )

        intercept = np.median(
            intercepts
        )

        left_lane = make_line_points(
            image,
            slope,
            intercept
        )

    # ------------------------------------------------
    # SELECT BEST RIGHT LINE
    # ------------------------------------------------

    right_lane = None

    if right_candidates:

        right_candidates.sort(
            key=lambda x: x[0]
        )

        selected = right_candidates[:8]

        slopes = [
            x[1] for x in selected
        ]

        intercepts = [
            x[2] for x in selected
        ]

        slope = np.median(
            slopes
        )

        intercept = np.median(
            intercepts
        )

        right_lane = make_line_points(
            image,
            slope,
            intercept
        )

    # ------------------------------------------------
    # DRAW RESULT
    # ------------------------------------------------

    result = image.copy()

    # Left lane
    if left_lane is not None:

        cv2.line(
            result,
            (left_lane[0], left_lane[1]),
            (left_lane[2], left_lane[3]),
            (0, 255, 0),
            7
        )

    # Right lane
    if right_lane is not None:

        cv2.line(
            result,
            (right_lane[0], right_lane[1]),
            (right_lane[2], right_lane[3]),
            (0, 255, 0),
            7
        )

    # ------------------------------------------------
    # LANE STATUS
    # ------------------------------------------------

    warning = "LANE DETECTION INCOMPLETE"

    if (
        left_lane is not None
        and right_lane is not None
    ):

        left_x = left_lane[0]
        right_x = right_lane[0]

        lane_center = (
            left_x + right_x
        ) // 2

        vehicle_center = center_x

        offset = (
            lane_center -
            vehicle_center
        )

        # Departure left
        if offset < -80:

            warning = (
                "LANE DEPARTURE - LEFT"
            )

        # Departure right
        elif offset > 80:

            warning = (
                "LANE DEPARTURE - RIGHT"
            )

        else:

            warning = "LANE SAFE"

        # Lane centre
        cv2.line(
            result,
            (lane_center, height),
            (
                lane_center,
                int(height * 0.52)
            ),
            (255, 0, 0),
            4
        )

        # Vehicle centre
        cv2.line(
            result,
            (vehicle_center, height),
            (
                vehicle_center,
                int(height * 0.52)
            ),
            (0, 255, 255),
            3
        )

    # ------------------------------------------------
    # STATUS BOX
    # ------------------------------------------------

    cv2.rectangle(
        result,
        (20, 20),
        (700, 90),
        (0, 0, 0),
        -1
    )

    if "DEPARTURE" in warning:

        color = (0, 0, 255)

    elif "SAFE" in warning:

        color = (0, 255, 0)

    else:

        color = (0, 255, 255)

    cv2.putText(
        result,
        warning,
        (35, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2
    )

    return result


def main():

    images = sorted(
        glob.glob(
            os.path.join(
                INPUT_DIR,
                "*.jpg"
            )
        )
    )

    print(
        f"Found {len(images)} images"
    )

    for i, image_path in enumerate(images):

        image = cv2.imread(
            image_path
        )

        if image is None:

            print(
                f"Could not read: {image_path}"
            )

            continue

        result = detect_lanes(
            image
        )

        filename = os.path.basename(
            image_path
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            filename
        )

        cv2.imwrite(
            output_path,
            result
        )

        print(
            f"[{i + 1}/{len(images)}] "
            f"{filename}"
        )

    print()
    print(
        "Lane detection completed."
    )


if __name__ == "__main__":

    main()
