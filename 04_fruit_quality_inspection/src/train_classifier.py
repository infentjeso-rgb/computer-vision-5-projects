import os
import shutil
import random
from pathlib import Path

from sklearn.model_selection import train_test_split

SOURCE = Path("dataset")
OUT = Path("dataset_split")

random.seed(42)

classes = {
    "good": 0,
    "defective": 1
}

# Clean previous split
if OUT.exists():
    shutil.rmtree(OUT)

for split in ["train", "val", "test"]:
    for cls in classes:
        (OUT / split / cls).mkdir(parents=True, exist_ok=True)

for cls in classes:
    images = []

    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        images.extend((SOURCE / cls).glob(ext))

    train, temp = train_test_split(
        images,
        test_size=0.30,
        random_state=42
    )

    val, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=42
    )

    for split, files in [
        ("train", train),
        ("val", val),
        ("test", test)
    ]:
        for src in files:
            shutil.copy2(
                src,
                OUT / split / cls / src.name
            )

    print(
        f"{cls}: "
        f"train={len(train)}, "
        f"val={len(val)}, "
        f"test={len(test)}"
    )

print("\nDataset split created.")
