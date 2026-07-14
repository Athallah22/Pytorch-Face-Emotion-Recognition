import os
import sys
import shutil
import random
import importlib.util

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.py')
spec = importlib.util.spec_from_file_location("config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

DATASET_DIR = config.DATASET_DIR
SPLIT_DIR = config.SPLIT_DIR
TRAIN_RATIO = config.TRAIN_RATIO
TRAIN_DIR = config.TRAIN_DIR
VAL_DIR = config.VAL_DIR

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def split_dataset():
    """
    Membagi dataset gambar dari DATASET_DIR ke dalam folder train dan val
    di dalam SPLIT_DIR sesuai dengan TRAIN_RATIO.
    """
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Direktori dataset sumber '{DATASET_DIR}' tidak ditemukan.")
        return

    if os.path.exists(SPLIT_DIR):
        print(f"Menghapus direktori split yang sudah ada: '{SPLIT_DIR}'")
        shutil.rmtree(SPLIT_DIR)

    print("Memulai proses pemisahan dataset...")
    print(f"   - Sumber: {DATASET_DIR}")
    print(f"   - Tujuan: {SPLIT_DIR}")
    print(f"   - Rasio: {TRAIN_RATIO*100:.0f}% Train / {(1-TRAIN_RATIO)*100:.0f}% Val")

    class_names = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    for split_name in ["train", "val"]:
        for class_name in class_names:
            os.makedirs(os.path.join(SPLIT_DIR, split_name, class_name), exist_ok=True)

    for class_name in class_names:
        class_path = os.path.join(DATASET_DIR, class_name)
        images = os.listdir(class_path)
        random.shuffle(images)

        split_idx = int(len(images) * TRAIN_RATIO)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        for img_name in train_imgs:
            src_path = os.path.join(class_path, img_name)
            dst_path = os.path.join(TRAIN_DIR, class_name, img_name)
            shutil.copy2(src_path, dst_path)

        for img_name in val_imgs:
            src_path = os.path.join(class_path, img_name)
            dst_path = os.path.join(VAL_DIR, class_name, img_name)
            shutil.copy2(src_path, dst_path)
            
    print("\nDataset berhasil dibagi ke dalam folder:")
    print(f"   - Train: {TRAIN_DIR}")
    print(f"   - Val  : {VAL_DIR}")

if __name__ == '__main__':
    split_dataset()

