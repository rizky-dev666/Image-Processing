import os
import cv2
import numpy as np
from tqdm import tqdm

def preprocess_image(image_path, output_path, size=(48, 48)):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"  Gagal membaca gambar: {image_path}")
            return False

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, size)
        cv2.imwrite(output_path, resized)
        return True

    except Exception as e:
        print(f" Error memproses {image_path}: {e}")
        return False


def preprocess_folder(input_folder, output_folder, size=(48, 48)):
    print(f"\n Memproses folder: {input_folder}")

    if not os.path.exists(input_folder):
        raise FileNotFoundError(f" Folder input tidak ditemukan: {input_folder}")

    for emotion in sorted(os.listdir(input_folder)):
        emotion_path = os.path.join(input_folder, emotion)
        if not os.path.isdir(emotion_path):
            continue

        output_emotion_path = os.path.join(output_folder, emotion)
        os.makedirs(output_emotion_path, exist_ok=True)

        image_files = [
            f for f in os.listdir(emotion_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        if not image_files:
            print(f" Kelas '{emotion}' kosong. Lewati.")
            continue

        print(f" Kelas: {emotion} ({len(image_files)} gambar)")
        for img_name in tqdm(image_files, desc=f"    {emotion}", unit="img"):
            input_img_path = os.path.join(emotion_path, img_name)
            output_img_path = os.path.join(output_emotion_path, img_name)
            preprocess_image(input_img_path, output_img_path, size)


def load_images_from_folder(folder, image_size=(48, 48)):
    if not os.path.exists(folder):
        raise FileNotFoundError(f" Folder tidak ditemukan: {folder}")

    images, labels = [], []
    for label in sorted(os.listdir(folder)):
        label_folder = os.path.join(folder, label)
        if not os.path.isdir(label_folder):
            continue

        image_files = [
            f for f in os.listdir(label_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        if not image_files:
            continue

        print(f" Loading kelas: {label} ({len(image_files)} gambar)")
        for filename in tqdm(image_files, desc=f"    {label}", unit="img"):
            img_path = os.path.join(label_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, image_size)
                images.append(img)
                labels.append(label)
            else:
                print(f" Gagal baca: {filename}")

    return np.array(images), np.array(labels)


def load_train_test_data(train_dir="dataset/archive/train", test_dir="dataset/archive/test", image_size=(48, 48)):
    print("\n Memuat data train...")
    X_train, y_train = load_images_from_folder(train_dir, image_size)

    print("\n Memuat data test...")
    X_test, y_test = load_images_from_folder(test_dir, image_size)

    print(f"\n Data loaded successfully!")
    print(f" Train: {len(X_train)} gambar | Test: {len(X_test)} gambar")
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    # === Konfigurasi Path ===
    INPUT_TRAIN_DIR = "dataset/archive/train"
    INPUT_TEST_DIR = "dataset/archive/test"

    OUTPUT_TRAIN_DIR = "preprocessed/archive/train"
    OUTPUT_TEST_DIR = "preprocessed/archive/test"

    # === Validasi input ===
    if not os.path.exists(INPUT_TRAIN_DIR):
        raise FileNotFoundError(f" Dataset train tidak ditemukan di: {INPUT_TRAIN_DIR}")
    if not os.path.exists(INPUT_TEST_DIR):
        raise FileNotFoundError(f" Dataset test tidak ditemukan di: {INPUT_TEST_DIR}")

    # === Buat folder output ===
    os.makedirs(OUTPUT_TRAIN_DIR, exist_ok=True)
    os.makedirs(OUTPUT_TEST_DIR, exist_ok=True)

    # === Jalankan preprocessing ===
    print(" Memulai preprocessing dataset...")

    preprocess_folder(INPUT_TRAIN_DIR, OUTPUT_TRAIN_DIR, size=(48, 48))
    preprocess_folder(INPUT_TEST_DIR, OUTPUT_TEST_DIR, size=(48, 48))

    print("\n Preprocessing selesai!")
    print(f" Hasil disimpan di:")
    print(f"   - {OUTPUT_TRAIN_DIR}")
    print(f"   - {OUTPUT_TEST_DIR}")