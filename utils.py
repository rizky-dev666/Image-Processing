import os
import cv2
import joblib
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA

# ==========================
# Konstanta Global
# ==========================
IMAGE_SIZE = (48, 48)
LBP_RADIUS = 1
LBP_N_POINTS = 8 * LBP_RADIUS
LBP_METHOD = 'uniform'
LBP_BINS = LBP_N_POINTS + 3   


# ==========================
# Fungsi Umum
# ==========================

def resize_image(image, width=48, height=48):
    return cv2.resize(image, (width, height))


# ==========================
# Model Handling
# ==========================

def create_svm_model():
    """
    Membuat pipeline model SVM dengan StandardScaler dan PCA.
    """
    return make_pipeline(
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


def save_model(model, label_encoder, model_path="saved_model/svm_model.pkl", encoder_path="saved_model/label_encoder.pkl"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(label_encoder, encoder_path)
    print(f"[INFO] Model disimpan di {model_path}")
    print(f"[INFO] Label encoder disimpan di {encoder_path}")


def load_model(model_path="saved_model/svm_model.pkl", encoder_path="saved_model/label_encoder.pkl"):
    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        raise FileNotFoundError("Model atau encoder tidak ditemukan.")
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    print("Model dan encoder berhasil dimuat!")
    return model, label_encoder


# ==========================
# Dataset Handling
# ==========================

def save_features(X_train, X_test, y_train, y_test, prefix="features"):
    os.makedirs(prefix, exist_ok=True)
    np.savez_compressed(os.path.join(prefix, "train_features.npz"), X=X_train, y=y_train)
    np.savez_compressed(os.path.join(prefix, "test_features.npz"), X=X_test, y=y_test)
    print(f"[INFO] Fitur disimpan ke: {prefix}/")


def encode_labels(y):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return y_encoded, encoder


# ==========================
# Ekstraksi Fitur (48x48)
# ==========================

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


def extract_combined_features(image):
    hog_feat = extract_hog_features(image)
    lbp_feat = extract_lbp_features(image)
    return np.hstack([hog_feat, lbp_feat])


# =================
# Prediksi Gambar
# =================

def predict_emotion(image_path, model, label_encoder):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Gagal membaca gambar: {image_path}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, IMAGE_SIZE)
    
    features = extract_combined_features(gray).reshape(1, -1)
    prediction = model.predict(features)
    return label_encoder.inverse_transform(prediction)[0]


# ==========================
# Visualisasi (Opsional)
# ==========================

def show_image_with_label(image_path, label):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Gagal menampilkan {image_path}")
        return
    cv2.putText(image, str(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Emotion Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==========================
# Demo Langsung
# ==========================
if __name__ == "__main__":
    try:
        model, encoder = load_model()
        # Cari gambar uji otomatis
        test_dirs = [
            "preprocessed/archive/test",
            "dataset/archive/test",
            "test"
        ]
        img_path = None
        for d in test_dirs:
            if os.path.exists(d):
                for emotion in os.listdir(d):
                    e_dir = os.path.join(d, emotion)
                    if os.path.isdir(e_dir):
                        files = [f for f in os.listdir(e_dir) if f.lower().endswith(('.jpg', '.png'))]
                        if files:
                            img_path = os.path.join(e_dir, files[0])
                            break
                if img_path:
                    break
        
        if img_path:
            label = predict_emotion(img_path, model, encoder)
            print(f"[RESULT] Prediksi: {label}")
            show_image_with_label(img_path, label)
        else:
            print("Tidak ada gambar uji ditemukan.")
    except Exception as e:
        print(f"[ERROR] {e}")