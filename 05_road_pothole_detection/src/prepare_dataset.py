import os
import random
import shutil
import xml.etree.ElementTree as ET

SOURCE = "dataset/raw/annotated-images"
OUTPUT = "dataset/yolo"

random.seed(42)

# Create YOLO directory structure
for split in ["train", "val", "test"]:
    os.makedirs(f"{OUTPUT}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUTPUT}/labels/{split}", exist_ok=True)

# Find all images
images = [
    f for f in os.listdir(SOURCE)
    if f.lower().endswith(".jpg")
]

images.sort()
random.shuffle(images)

# 70 / 15 / 15 split
n = len(images)
train_end = int(0.70 * n)
val_end = train_end + int(0.15 * n)

splits = {
    "train": images[:train_end],
    "val": images[train_end:val_end],
    "test": images[val_end:]
}

def convert_annotation(xml_path, image_width, image_height):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    labels = []

    for obj in root.findall("object"):
        # Single class: pothole = 0
        class_id = 0

        bbox = obj.find("bndbox")

        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # Convert VOC coordinates to YOLO format
        x_center = ((xmin + xmax) / 2) / image_width
        y_center = ((ymin + ymax) / 2) / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height

        labels.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} "
            f"{width:.6f} {height:.6f}"
        )

    return labels


total_annotations = 0

for split, split_images in splits.items():

    for image_name in split_images:

        image_path = os.path.join(SOURCE, image_name)
        xml_path = os.path.join(
            SOURCE,
            os.path.splitext(image_name)[0] + ".xml"
        )

        # Read image dimensions from XML
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find("size")
        image_width = float(size.find("width").text)
        image_height = float(size.find("height").text)

        labels = convert_annotation(
            xml_path,
            image_width,
            image_height
        )

        # Copy image
        shutil.copy2(
            image_path,
            f"{OUTPUT}/images/{split}/{image_name}"
        )

        # Write YOLO label
        label_name = os.path.splitext(image_name)[0] + ".txt"

        with open(
            f"{OUTPUT}/labels/{split}/{label_name}",
            "w"
        ) as f:
            f.write("\n".join(labels))

        total_annotations += len(labels)

# Create data.yaml
yaml_content = f"""path: {os.path.abspath(OUTPUT)}
train: images/train
val: images/val
test: images/test

names:
  0: pothole
"""

with open(f"{OUTPUT}/data.yaml", "w") as f:
    f.write(yaml_content)

print("=" * 50)
print("DATASET PREPARATION COMPLETE")
print("=" * 50)

for split, split_images in splits.items():
    print(f"{split}: {len(split_images)} images")

print(f"Total images: {len(images)}")
print(f"Total pothole annotations: {total_annotations}")
print(f"YOLO dataset: {OUTPUT}")
