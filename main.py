import os
import numpy as np
import joblib
from preprocess import load_train_test_data
from utils import encode_labels, create_svm_model, extract_combined_features
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# === PATH KONFIGURASI ===
DATASET_PATH = "preprocessed/archive"
MODEL_DIR = "saved_model"
FEATURES_DIR = "features"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "svm_model.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
TRAIN_FEATURES_PATH = os.path.join(FEATURES_DIR, "train_features.npz")
TEST_FEATURES_PATH = os.path.join(FEATURES_DIR, "test_features.npz")


def preprocess_data():
    print("\n=== [1] PREPROCESS DATASET ===")
    X_train, y_train, X_test, y_test = load_train_test_data(
        train_dir=os.path.join(DATASET_PATH, "train"),
        test_dir=os.path.join(DATASET_PATH, "test"),
        image_size=(48, 48)
    )
    print(f"Data train: {len(X_train)} gambar | Data test: {len(X_test)} gambar")
    return X_train, X_test, y_train, y_test


def extract_features(X_train, X_test):
    print("\n=== [2] EKSTRAKSI FITUR HOG + LBP ===")
    print("Ekstraksi fitur train...")
    X_train_features = np.array([extract_combined_features(img) for img in X_train])
    print("Ekstraksi fitur test...")
    X_test_features = np.array([extract_combined_features(img) for img in X_test])
    print(f"Ekstraksi selesai. Dimensi fitur: {X_train_features.shape[1]}")
    return X_train_features, X_test_features


def train_model(X_train_features, X_test_features, y_train, y_test):
    print("\n=== [3] TRAINING MODEL SVM ===")
    y_train_encoded, le = encode_labels(y_train)
    y_test_encoded = le.transform(y_test)

    svm = create_svm_model()
    
    print("Melatih model...")
    svm.fit(X_train_features, y_train_encoded)

    y_pred = svm.predict(X_test_features)
    acc = accuracy_score(y_test_encoded, y_pred)
    print(f"Akurasi: {acc * 100:.2f}%")

    print("\n Laporan klasifikasi:")
    print(classification_report(y_test_encoded, y_pred, target_names=le.classes_))

    # Simpan model & encoder
    joblib.dump(svm, MODEL_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)
    print(f"\n Model disimpan di: '{MODEL_PATH}'")
    print(f"Label encoder disimpan di: '{LABEL_ENCODER_PATH}'")


def test_model():
    print("\n=== [4] TESTING MODEL (dari file fitur) ===")
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_ENCODER_PATH):
        print("Model belum ada. Jalankan training dulu.")
        return

    svm = joblib.load(MODEL_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)

    data_test = np.load(TEST_FEATURES_PATH, allow_pickle=True)
    X_test, y_test = data_test["X"], data_test["y"]

    y_test_encoded = le.transform(y_test)
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test_encoded, y_pred)
    print(f"\n Akurasi model (dari file): {acc * 100:.2f}%")

    print("\n Laporan klasifikasi:")
    print(classification_report(y_test_encoded, y_pred, target_names=le.classes_))

    # Confusion Matrix
    cm = confusion_matrix(y_test_encoded, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=le.classes_, yticklabels=le.classes_
    )
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.title("Confusion Matrix - Deteksi Emosi Pemain Game")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix_main.png"), dpi=150)
    plt.close()


def save_features_to_disk(X_train, X_test, y_train, y_test):
    np.savez_compressed(TRAIN_FEATURES_PATH, X=X_train, y=y_train)
    np.savez_compressed(TEST_FEATURES_PATH, X=X_test, y=y_test)
    print(f"\n Fitur disimpan ke:\n   - {TRAIN_FEATURES_PATH}\n   - {TEST_FEATURES_PATH}")


if __name__ == "__main__":
    print("========== UAS ==========")

    try:
        # Tahap 1: Load data
        X_train, X_test, y_train, y_test = preprocess_data()

        # Tahap 2: Ekstraksi fitur
        X_train_features, X_test_features = extract_features(X_train, X_test)

        # Tahap 3: Simpan fitur
        save_features_to_disk(X_train_features, X_test_features, y_train, y_test)

        # Tahap 4: Training
        train_model(X_train_features, X_test_features, y_train, y_test)

        # Tahap 5: Testing
        test_model()

        print("\n SEMUA TAHAP SELESAI! SIAP UNTUK DEMO & PAPER!")

    except Exception as e:
        print(f"\n Terjadi error: {e}")
        import traceback
        traceback.print_exc()