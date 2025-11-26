import os
import time
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from tqdm import tqdm

def train_model(train_features_path, save_model_dir):
    print("\n Memulai pelatihan model SVM...\n")

    # === Validasi file input ===
    if not os.path.exists(train_features_path):
        raise FileNotFoundError(f"File fitur tidak ditemukan: {train_features_path}")

    # === Load data ===
    try:
        data = np.load(train_features_path, allow_pickle=True)
        X = data["X"]
        y = data["y"]
    except Exception as e:
        raise ValueError(f"Gagal memuat fitur: {e}")

    print(f"Jumlah data latih: {len(y)} sampel")
    print(f"Jumlah fitur per gambar: {X.shape[1]}")
    print(f"Label unik: {np.unique(y)}\n")

    # === Encode label ===
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # === Buat pipeline ===
    model = make_pipeline(
        StandardScaler(),
        PCA(n_components=200),  
        SVC(
            kernel='rbf',
            C=0.1,              
            gamma=0.001,
            class_weight='balanced',
            probability=True,
            random_state=42
        )
    )

    # === Training dengan "progress" ===
    print("Melatih model...")
    start_time = time.time()
    
    with tqdm(total=100, desc="  Training", unit="%") as pbar:
        model.fit(X, y_encoded)
        pbar.update(100)

    end_time = time.time()
    train_time = end_time - start_time

    # === Evaluasi ===
    y_pred_train = model.predict(X)
    train_acc = accuracy_score(y_encoded, y_pred_train)

    print(f"\n Training selesai!")
    print(f"Waktu training: {train_time:.2f} detik")
    print(f"Akurasi di data latih: {train_acc * 100:.2f}%")

    # === Simpan ===
    os.makedirs(save_model_dir, exist_ok=True)
    model_path = os.path.join(save_model_dir, "svm_model.pkl")
    encoder_path = os.path.join(save_model_dir, "label_encoder.pkl")

    joblib.dump(model, model_path)
    joblib.dump(label_encoder, encoder_path)

    # === Simpan info untuk paper ===
    info_path = os.path.join(save_model_dir, "training_info.txt")
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write(f"Jumlah data latih: {len(y)}\n")
        f.write(f"Jumlah fitur asli: {X.shape[1]}\n")
        f.write(f"Jumlah fitur setelah PCA: 200\n")
        f.write(f"Akurasi latih: {train_acc:.4f}\n")
        f.write(f"Kernel SVM: rbf\n")
        f.write(f"C: 0.1\n")        
        f.write(f"Gamma: 0.001\n")
        f.write(f"Class weight: balanced\n")
        f.write(f"Waktu training: {train_time:.2f} detik\n")

    print(f"\n Model disimpan di: {model_path}")
    print(f"Label encoder disimpan di: {encoder_path}")
    print(f"Info training disimpan di: {info_path}")

    return model, label_encoder


if __name__ == "__main__":
    TRAIN_FEATURES_PATH = "features/train_features.npz"
    SAVE_MODEL_DIR = "saved_model"

    print("========== TRAINING MODEL SVM ==========")
    try:
        model, encoder = train_model(TRAIN_FEATURES_PATH, SAVE_MODEL_DIR)
        print("\n Pelatihan selesai dengan sukses!")
    except Exception as e:
        print(f"\n Terjadi error: {e}")
        import traceback
        traceback.print_exc()