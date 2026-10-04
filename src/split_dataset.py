import os
import shutil
import csv
from sklearn.model_selection import train_test_split

from config import (
    RAW_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    SEED,
    RESULTS_DIR
)

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
STATS_PATH = os.path.join(RESULTS_DIR, "dataset_split_stats.csv")


def create_folder(path):
    os.makedirs(path, exist_ok=True)


def clear_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)


def get_image_files(folder):
    return [
        f for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTS)
    ]


def copy_files(files, src_dir, dst_dir):
    create_folder(dst_dir)
    for f in files:
        shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))


def split_dataset():
    print("===== SPLITTING DATASET =====")

    # tạo folder results
    create_folder(RESULTS_DIR)

    # xóa dữ liệu cũ
    clear_folder(TRAIN_DIR)
    clear_folder(VAL_DIR)
    clear_folder(TEST_DIR)

    # tạo lại
    create_folder(TRAIN_DIR)
    create_folder(VAL_DIR)
    create_folder(TEST_DIR)

    classes = [
        c for c in os.listdir(RAW_DIR)
        if os.path.isdir(os.path.join(RAW_DIR, c))
    ]

    stats = []

    for class_name in classes:
        class_path = os.path.join(RAW_DIR, class_name)
        images = get_image_files(class_path)

        if len(images) == 0:
            print(f"Skip: {class_name} (no images)")
            continue

        print(f"{class_name}: total={len(images)}")

        # chia dữ liệu
        if len(images) < 3:
            train_files = images
            val_files = []
            test_files = []
        else:
            train_files, temp_files = train_test_split(
                images,
                test_size=(1 - TRAIN_RATIO),
                random_state=SEED,
                shuffle=True
            )

            val_ratio_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)

            if len(temp_files) < 2:
                val_files = temp_files
                test_files = []
            else:
                val_files, test_files = train_test_split(
                    temp_files,
                    test_size=(1 - val_ratio_adjusted),
                    random_state=SEED,
                    shuffle=True
                )

        # copy file
        copy_files(train_files, class_path, os.path.join(TRAIN_DIR, class_name))
        copy_files(val_files, class_path, os.path.join(VAL_DIR, class_name))
        copy_files(test_files, class_path, os.path.join(TEST_DIR, class_name))

        stats.append({
            "class": class_name,
            "total": len(images),
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files)
        })

        print(f"Done: {class_name} | train={len(train_files)} val={len(val_files)} test={len(test_files)}")

    # lưu thống kê
    with open(STATS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "total", "train", "val", "test"])
        writer.writeheader()
        writer.writerows(stats)

    print("\nSaved stats to:", STATS_PATH)
    print("===== DONE =====")


if __name__ == "__main__":
    split_dataset()