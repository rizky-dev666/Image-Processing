import os
import cv2
import time
import numpy as np
from tqdm import tqdm
from utils import extract_combined_features


def extract_features_from_folder(base_dir, output_file, sample_rate=1.0):
    X = []
    y = []

    print(f"\n Ekstraksi fitur dari folder: {base_dir}")
    print(f"Menggunakan 100% data (tanpa sampling)\n")

    # === Timer total ===
    total_start_time = time.time()

    for label_name in sorted(os.listdir(base_dir)):
        label_dir = os.path.join(base_dir, label_name)
        if not os.path.isdir(label_dir):
            continue

        image_files = [
            f for f in os.listdir(label_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        print(f"Kelas: {label_name} | Diproses: {len(image_files)} gambar")

        # === Timer per kelas ===
        class_start_time = time.time()

        for img_name in tqdm(image_files, desc=f"    {label_name}", unit="img"):
            img_path = os.path.join(label_dir, img_name)
            try:
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise ValueError(f"Tidak bisa membaca gambar: {img_path}")
                image = cv2.resize(image, (48, 48))
                features = extract_combined_features(image)
                
                X.append(features)
                y.append(label_name)
            except Exception as e:
                print(f"Gagal: {img_path} → {e}")

        # === Hitung waktu per kelas ===
        class_time = time.time() - class_start_time
        print(f"Selesai dalam {class_time:.2f} detik")

    if not X:
        raise ValueError(f"Tidak ada gambar valid ditemukan di {base_dir}")

    X = np.array(X)
    y = np.array(y)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savez_compressed(output_file, X=X, y=y)

    # === Hitung waktu total ===
    total_time = time.time() - total_start_time
    print(f"\n Ekstraksi selesai!")
    print(f"Waktu total: {total_time:.2f} detik")
    print(f"Fitur disimpan ke: {output_file}")
    print(f"Jumlah  {len(X)} | Dimensi fitur: {X.shape[1]}")
    return X, y


if __name__ == "__main__":
    # Konfigurasi
    TRAIN_DIR = "preprocessed/archive/train"
    TEST_DIR = "preprocessed/archive/test"
    OUTPUT_DIR = "features"
    SAMPLE_RATE = 1.0 

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("========== EKSTRAKSI FITUR HOG + LBP ==========")

    extract_features_from_folder(
        TRAIN_DIR,
        os.path.join(OUTPUT_DIR, "train_features.npz"),
        sample_rate=SAMPLE_RATE
    )
    extract_features_from_folder(
        TEST_DIR,
        os.path.join(OUTPUT_DIR, "test_features.npz"),
        sample_rate=SAMPLE_RATE
    )

    print("\n Ekstraksi fitur selesai!")