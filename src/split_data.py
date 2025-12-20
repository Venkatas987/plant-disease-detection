import os
import shutil
import random

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

random.seed(42)

def create_dirs(classes):
    for split in ["train", "val", "test"]:
        for cls in classes:
            os.makedirs(os.path.join(PROCESSED_DIR, split, cls), exist_ok=True)

def split_dataset():
    classes = os.listdir(RAW_DIR)
    create_dirs(classes)

    for cls in classes:
        images = os.listdir(os.path.join(RAW_DIR, cls))
        random.shuffle(images)

        total = len(images)
        train_end = int(total * TRAIN_SPLIT)
        val_end = int(total * (TRAIN_SPLIT + VAL_SPLIT))

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:]
        }

        for split, imgs in splits.items():
            for img in imgs:
                src = os.path.join(RAW_DIR, cls, img)
                dst = os.path.join(PROCESSED_DIR, split, cls, img)
                shutil.copy2(src, dst)

        print(f"{cls}: {len(splits['train'])} train | {len(splits['val'])} val | {len(splits['test'])} test")

if __name__ == "__main__":
    split_dataset()
