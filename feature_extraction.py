import os
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from tqdm import tqdm
import time

# === Parameter LBP ===
LBP_RADIUS = 1
LBP_N_POINTS = 8 * LBP_RADIUS  # = 8
LBP_METHOD = 'uniform'
LBP_BINS = LBP_N_POINTS + 3    # = 11 bins


def extract_hog_features(image):
    return hog(
        image,
        orientations=9,
        pixels_per_cell=(4, 4), 
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False,
        feature_vector=True
    )


def extract_lbp_features(image):
    lbp = local_binary_pattern(image, LBP_N_POINTS, LBP_RADIUS, LBP_METHOD)
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(LBP_BINS),    
        range=(0, LBP_BINS - 1)   
    )
    hist = hist.astype("float32")
    hist /= (hist.sum() + 1e-7)
    return hist


def extract_hog_lbp_features(image):
    hog_feat = extract_hog_features(image)
    lbp_feat = extract_lbp_features(image)
    return np.hstack([hog_feat, lbp_feat])


def extract_combined_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Tidak bisa membaca gambar: {image_path}")
    image = cv2.resize(image, (48, 48))
    return extract_hog_lbp_features(image)


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
                features = extract_combined_features(img_path)
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