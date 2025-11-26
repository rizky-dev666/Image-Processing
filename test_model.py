import os
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def test_model(model_path, label_encoder_path, test_features_path, save_cm=True):
    print("\n Menguji model...\n")

    # === Cek keberadaan file ===
    for path, name in [(model_path, "Model"), (label_encoder_path, "Label Encoder"), (test_features_path, "Fitur Test")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} tidak ditemukan: {path}")

    # === Load model dan encoder ===
    model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)

    # === Load fitur test ===
    data = np.load(test_features_path, allow_pickle=True)
    X_test = data["X"]
    y_test = data["y"]  # ini adalah label string

    print(f"Jumlah data test: {len(y_test)}")
    print(f"Label unik di test: {np.unique(y_test)}")
    print(f"Label yang dikenal oleh model: {label_encoder.classes_}")

    # === Validasi: pastikan semua label di test ada di encoder ===
    unknown_labels = set(y_test) - set(label_encoder.classes_)
    if unknown_labels:
        raise ValueError(f" Label tidak dikenal ditemukan di data test: {unknown_labels}. "
                         f"Pastikan tidak ada label baru di test set.")

    # === Prediksi ===
    y_pred_encoded = model.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    # === Hitung akurasi ===
    acc = accuracy_score(y_test, y_pred)
    print(f"Akurasi Model: {acc * 100:.2f}%\n")

    # === Confusion Matrix ===
    cm = confusion_matrix(y_test, y_pred, labels=label_encoder.classes_)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        cbar_kws={'shrink': 0.8}
    )
    plt.xlabel('Predicted Emotion')
    plt.ylabel('True Emotion')
    plt.title('Confusion Matrix - Deteksi Emosi Pemain Game')
    plt.tight_layout()

    if save_cm:
        cm_path = os.path.join(os.path.dirname(model_path), "confusion_matrix_test.png")
        plt.savefig(cm_path, dpi=150)
        print(f" Confusion matrix disimpan di: {cm_path}")

    plt.show()

    # === Laporan Klasifikasi ===
    print("\n Laporan Klasifikasi:\n")
    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
    print(report)

    return acc


if __name__ == "__main__":
    MODEL_PATH = "saved_model/svm_model.pkl"
    LABEL_ENCODER_PATH = "saved_model/label_encoder.pkl"
    TEST_FEATURES_PATH = "features/test_features.npz"

    try:
        accuracy = test_model(
            model_path=MODEL_PATH,
            label_encoder_path=LABEL_ENCODER_PATH,
            test_features_path=TEST_FEATURES_PATH,
            save_cm=True
        )
        print(f"\n Pengujian selesai! Akurasi akhir: {accuracy * 100:.2f}%")
    except Exception as e:
        print(f"\n Terjadi error: {e}")
        import traceback
        traceback.print_exc()