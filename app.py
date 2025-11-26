import os
import tempfile
import time
from collections import Counter

import cv2
import numpy as np
import streamlit as st

from utils import (
    extract_combined_features,
    load_model,
    resize_image,
)

# ==========================
# KONFIGURASI MODEL
# ==========================
MODEL_PATH = "saved_model/svm_model.pkl"
ENCODER_PATH = "saved_model/label_encoder.pkl"

st.set_page_config(
    page_title="Deteksi Ekspresi Manusia | HOG + LBP + SVM",
    layout="wide",
)

EMOTION_COLORS = {
    "happy": (46, 204, 113),
    "sad": (241, 196, 15),
    "angry": (235, 87, 87),
    "neutral": (52, 152, 219),
    "surprise": (155, 89, 182),
}

CUSTOM_STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');
:root {
    --bg: #050712;
    --panel: rgba(255, 255, 255, 0.05);
    --surface: rgba(255, 255, 255, 0.05);
    --surface-strong: rgba(255, 255, 255, 0.09);
    --border: rgba(255, 255, 255, 0.12);
    --text: #f6f8ff;
    --muted: #c5cee0;
    --accent: #6df2c3;
    --accent-2: #6ab5ff;
    --amber: #f5c361;
    --shadow: 0 24px 70px rgba(0, 0, 0, 0.4);
}

/* ---------- Base layout ---------- */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 20% 18%, rgba(109, 242, 195, 0.12), transparent 26%),
        radial-gradient(circle at 82% 14%, rgba(106, 181, 255, 0.14), transparent 24%),
        radial-gradient(circle at 32% 78%, rgba(123, 111, 255, 0.12), transparent 22%),
        linear-gradient(155deg, #05060f 0%, #0a1020 45%, #050712 100%);
    background-attachment: fixed;
}
.main {
    font-family: "Space Grotesk", "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
    color: var(--text);
}
.block-container { padding-top: 2.3rem; padding-bottom: 2.6rem; max-width: 1260px; }
section[data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
.page-shell {
    position: relative;
    max-width: 1240px;
    margin: 1.2rem auto 2.4rem;
    padding: 0 0.5rem;
}
.page-shell::before {
    content: "";
    position: absolute;
    inset: -6rem 18% auto;
    height: 38px;
    background: linear-gradient(120deg, rgba(109, 242, 195, 0.22), rgba(106, 181, 255, 0.18));
    filter: blur(16px);
    opacity: 0.86;
}
.spacer-sm { height: 0.6rem; }
.spacer-md { height: 1.1rem; }

/* ---------- Hero & spotlight ---------- */
.hero-wrap {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 1rem;
    align-items: stretch;
    position: relative;
}
.hero-card {
    position: relative;
    background: linear-gradient(150deg, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0.02));
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 1.45rem 1.55rem;
    box-shadow: var(--shadow);
    overflow: hidden;
}
.hero-card::after {
    content: "";
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 20% 20%, rgba(109, 242, 195, 0.08), transparent 32%);
    opacity: 0.9;
    pointer-events: none;
}
.hero-card h1 { margin: 0.45rem 0 0.35rem; font-size: 2.05rem; letter-spacing: -0.4px; }
.hero-card p { margin: 0; color: var(--muted); }
.hero-side { background: linear-gradient(160deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.03)); }
.text-glow { color: var(--accent); text-shadow: 0 0 22px rgba(109, 242, 195, 0.65); }

/* ---------- Chips & pills ---------- */
.chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.38rem 0.85rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.87rem;
    letter-spacing: 0.1px;
}
.chip-soft { background: rgba(255, 255, 255, 0.06); border: 1px solid rgba(255, 255, 255, 0.16); color: var(--text); }
.chip-strong { background: linear-gradient(120deg, rgba(109, 242, 195, 0.4), rgba(106, 181, 255, 0.42)); border: 1px solid rgba(255, 255, 255, 0.22); color: #051022; }
.chip-line { border: 1px solid rgba(255, 255, 255, 0.28); background: rgba(255, 255, 255, 0.03); color: var(--text); }
.hero-actions { display: flex; flex-wrap: wrap; gap: 0.55rem; }
.pill-row { display: flex; flex-wrap: wrap; gap: 0.55rem; margin-top: 0.9rem; }
.pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.42rem 1rem;
    border-radius: 999px;
    font-weight: 700;
    border: 1px solid rgba(255, 255, 255, 0.2);
    background: rgba(255, 255, 255, 0.03);
    color: var(--text);
}
.pill.accent { border-color: rgba(109, 242, 195, 0.6); background: linear-gradient(130deg, rgba(109, 242, 195, 0.32), rgba(106, 181, 255, 0.26)); color: #041224; }
.pill.outline { border-color: rgba(255, 255, 255, 0.28); }

/* ---------- Metrics ---------- */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 0.7rem;
    margin-top: 0.4rem;
}
.metric-card {
    position: relative;
    border: 1px solid var(--border);
    border-radius: 14px;
    background: var(--surface);
    padding: 0.85rem 0.95rem;
}
.metric-card.large { grid-column: span 2; }
.metric-label { margin: 0; font-size: 0.88rem; color: var(--muted); }
.metric-value { margin: 0.2rem 0 0.05rem; font-size: 1.6rem; font-weight: 800; letter-spacing: -0.2px; }
.metric-sub { margin: 0; font-size: 0.9rem; color: rgba(255, 255, 255, 0.72); }
.band { display: flex; flex-wrap: wrap; gap: 0.45rem; margin-top: 0.85rem; }

/* ---------- Feature grid ---------- */
.mode-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 0.8rem;
    margin: 1rem 0 1.4rem;
}
.mode-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 0.95rem 1rem;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.24);
}
.mode-card h4 { margin: 0.1rem 0 0.25rem; font-size: 1.05rem; }
.mode-card p { margin: 0; color: var(--muted); }
.mode-chips { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.55rem; }

/* ---------- Global components ---------- */
.section-heading { display: flex; align-items: center; justify-content: space-between; gap: 0.75rem; margin: 0.3rem 0 0.7rem; }
.section-heading h3 { margin: 0; letter-spacing: -0.1px; }
.section-note { color: var(--muted); margin: 0; }

div[data-testid="stFileUploaderDropzone"] { border: 1px dashed rgba(255, 255, 255, 0.32); border-radius: 1rem; background: rgba(255, 255, 255, 0.03); }
form[data-testid="stForm"] {
    background: linear-gradient(130deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.02));
    border: 1px solid var(--border);
    border-radius: 1rem;
    padding: 1rem 1.15rem 0.9rem;
    box-shadow: var(--shadow);
}
.stButton > button {
    border-radius: 12px;
    padding: 0.85rem 1.2rem;
    font-weight: 800;
    border: 1px solid rgba(255, 255, 255, 0.3);
    background: linear-gradient(120deg, rgba(109, 242, 195, 0.5), rgba(106, 181, 255, 0.52));
    color: #041224;
    font-size: 1rem;
    letter-spacing: 0.15px;
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.28), 0 0 24px rgba(106, 181, 255, 0.32);
    transition: transform 120ms ease, box-shadow 160ms ease, filter 120ms ease;
}
.stButton > button:hover { transform: translateY(-1px); filter: brightness(1.04); box-shadow: 0 18px 44px rgba(0, 0, 0, 0.32), 0 0 30px rgba(109, 242, 195, 0.28); }
.stButton > button:active { transform: translateY(0); }
.form-button-container { display: flex; align-items: center; justify-content: center; height: 100%; }
.form-button-container button { width: 100%; }
.result-frame {
    padding: 1rem;
    border-radius: 1.1rem;
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.09));
    border: 1px solid var(--border);
    min-height: 120px;
    box-shadow: 0 10px 34px rgba(0, 0, 0, 0.26);
}
.emotion-card {
    padding: 1rem 1.05rem;
    border-radius: 1rem;
    border: 1px solid var(--border);
    background: var(--surface);
    margin-bottom: 1rem;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.24);
}
.emotion-card h4 { margin: 0 0 0.45rem 0; font-size: 1.04rem; }
.emotion-card ul { padding-left: 1.05rem; margin: 0; }
.emotion-card li { margin-bottom: 0.3rem; }
.emotion-card.emotion-success { border-color: rgba(46, 204, 113, 0.6); }
.emotion-card.emotion-warning { border-color: rgba(241, 196, 15, 0.6); }
.emotion-card.emotion-error { border-color: rgba(235, 87, 87, 0.6); }
.emotion-card.emotion-info { border-color: rgba(52, 152, 219, 0.45); }
.preview-card {
    border-radius: 1.05rem;
    border: 1px dashed rgba(255, 255, 255, 0.2);
    padding: 0.9rem;
    min-height: 360px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(255, 255, 255, 0.03);
}
.preview-card img { border-radius: 1rem; }
.section-card { background: var(--surface); border: 1px solid var(--border); border-radius: 1rem; padding: 0.9rem 1rem; }

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.4rem;
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 0.35rem;
    border-radius: 999px;
    box-shadow: 0 10px 22px rgba(0, 0, 0, 0.2);
}
.stTabs [data-baseweb="tab"] {
    padding: 0.62rem 1.2rem;
    border-radius: 999px;
    border: 1px solid transparent;
    transition: 120ms ease;
}
.stTabs [data-baseweb="tab"]:hover { border-color: rgba(109, 242, 195, 0.35); }
.stTabs [aria-selected="true"] {
    background: linear-gradient(120deg, rgba(109, 242, 195, 0.26), rgba(106, 181, 255, 0.26));
    border-color: rgba(255, 255, 255, 0.42) !important;
    box-shadow: 0 8px 22px rgba(0, 0, 0, 0.22);
}

/* ---------- Responsive tweaks ---------- */
@media (max-width: 992px) {
    .block-container { padding-left: 1.05rem; padding-right: 1.05rem; }
    .hero-card h1 { font-size: 1.85rem; }
    .metric-grid { grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); }
    .metric-card.large { grid-column: span 1; }
    .stButton > button { width: 100%; }
}
@media (max-width: 640px) {
    .block-container { padding-left: 0.9rem; padding-right: 0.9rem; }
    .stTabs [data-baseweb="tab-list"] { flex-wrap: wrap; justify-content: center; }
    .stTabs [data-baseweb="tab"] { width: 100%; justify-content: center; }
    div[data-testid="stFileUploaderDropzone"] { padding: 1.5rem 0.5rem; }
    .result-frame { padding: 0.8rem; }
}
</style>
"""
st.markdown(CUSTOM_STYLES, unsafe_allow_html=True)

# ==========================
# FUNGSI BANTUAN UI
# ==========================
def display_card(placeholder, title, body_html, variant="info"):
    placeholder.markdown(
        f"""
        <div class="emotion-card emotion-{variant}">
            <h4>{title}</h4>
            <div class="emotion-card-body">
                {body_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_preview_card(container, renderer):
    container.empty()
    with container:
        st.markdown("<div class='preview-card'>", unsafe_allow_html=True)
        renderer()
        st.markdown("</div>", unsafe_allow_html=True)


# ==========================
# LOAD MODEL & ENCODER
# ==========================
with st.sidebar:
    try:
        svm_model, label_encoder = load_model(MODEL_PATH, ENCODER_PATH)
        st.success("Model berhasil dimuat!")
    except Exception as e:
        st.error(f"Gagal memuat model:\n{e}")
        st.stop()

    st.header("Panduan")
    st.markdown(
        "1. Pilih tab *Gambar* atau *Video*.\n"
        "2. Unggah file yang ingin dianalisis.\n"
        "3. Klik tombol proses lalu tunggu hasilnya."
    )
    st.caption(f"Model: `{MODEL_PATH}`")

    st.header("Pengaturan")
    st.info("Pengaturan Frame Skip Rate kini tersedia di tab Video dan Realtime masing-masing.")

# ==========================
# KONFIGURASI DETEKSI WAJAH
# ==========================
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)


# ==========================
# FUNGSI UTILITAS
# ==========================
def make_square_image(image):
    h, w = image.shape[:2]
    if h == w:
        return image
    min_side = min(h, w)
    y_start = (h - min_side) // 2
    x_start = (w - min_side) // 2
    cropped_img = image[y_start:y_start + min_side, x_start:x_start + min_side]
    return cropped_img


def draw_emotion_box(frame, x, y, w, h, emotion):
    color = EMOTION_COLORS.get(emotion.lower(), (0, 255, 0))
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)

    text = emotion.upper()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

    padding = 6
    label_width = text_w + padding * 2
    label_height = text_h + padding * 2
    label_x = x + 4
    label_y = y - label_height - 4

    if label_y < 0:
        label_y = y + 4

    cv2.rectangle(
        frame,
        (label_x - 2, label_y - 2),
        (label_x + label_width + 2, label_y + label_height + 2),
        (0, 0, 0),
        -1,
    )
    text_position = (label_x + padding, label_y + label_height - padding)
    cv2.putText(frame, text, text_position, font, font_scale, color, thickness, cv2.LINE_AA)


def suppress_overlapping_faces(faces, overlap_threshold=0.4, max_faces=None):
    if len(faces) == 0:
        return []

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in faces], dtype=float)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    idxs = np.argsort(y2)
    pick = []

    while idxs.size > 0:
        i = idxs[-1]
        pick.append(i)
        idxs = idxs[:-1]
        if idxs.size == 0:
            break

        xx1 = np.maximum(x1[i], x1[idxs])
        yy1 = np.maximum(y1[i], y1[idxs])
        xx2 = np.minimum(x2[i], x2[idxs])
        yy2 = np.minimum(y2[i], y2[idxs])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        overlap = (w * h) / areas[idxs]

        idxs = idxs[overlap <= overlap_threshold]

    filtered = []
    for idx in pick:
        x, y = int(x1[idx]), int(y1[idx])
        w = int(x2[idx] - x1[idx])
        h = int(y2[idx] - y1[idx])
        filtered.append((x, y, w, h))

    filtered.sort(key=lambda b: b[2] * b[3], reverse=True)
    if max_faces is not None:
        filtered = filtered[:max_faces]
    return filtered


def has_skin_tone(region, threshold=0.08):
    if region.size == 0:
        return False

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 48, 50], dtype=np.uint8)
    upper1 = np.array([20, 255, 255], dtype=np.uint8)
    lower2 = np.array([170, 48, 50], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    ratio = cv2.countNonZero(mask) / mask.size
    return ratio >= threshold


def is_valid_face_roi(face_roi_gray, face_roi_color=None):
    roi_gray = face_roi_gray if face_roi_gray.ndim == 2 else cv2.cvtColor(face_roi_gray, cv2.COLOR_BGR2GRAY)
    h, w = roi_gray.shape[:2]

    if min(h, w) < 32:
        return False

    aspect_ratio = w / float(h)
    if not (0.58 <= aspect_ratio <= 1.8):
        return False

    if roi_gray.std() < 6:
        return False

    laplacian_var = cv2.Laplacian(roi_gray, cv2.CV_64F).var()
    if laplacian_var < 18:
        return False

    upper_face = roi_gray[: max(8, int(h * 0.65))]
    eyes = eye_cascade.detectMultiScale(
        upper_face,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(14, 14),
    )
    if len(eyes) == 0:
        return False

    if face_roi_color is None:
        return True

    if has_skin_tone(face_roi_color, threshold=0.07):
        return True

    if roi_gray.std() >= 10 and laplacian_var >= 35:
        return True

    return False


def annotate_image(image, detections):
    annotated = image.copy()
    for result in detections:
        x, y, w, h = result["box"]
        draw_emotion_box(annotated, x, y, w, h, result["emotion"])
    return annotated


def summarize_emotions(detections):
    if not detections:
        return "<p>Tidak ada wajah terdeteksi.</p>"

    counts = Counter(result["emotion"] for result in detections)
    items = "".join(
        f"<li><strong>{emotion.title()}</strong>: {count} wajah</li>"
        for emotion, count in counts.most_common()
    )
    return f"<ul>{items}</ul>"


def summarize_video_counter(counter):
    if not counter:
        return "<p>Tidak ada wajah terdeteksi pada video.</p>"

    items = "".join(
        f"<li><strong>{emotion.title()}</strong>: {count} frame</li>"
        for emotion, count in counter.most_common()
    )
    return f"<ul>{items}</ul>"


def predict_emotion_from_image(image):
    gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray_full,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(32, 32),
    )

    if len(faces) == 0:
        return []

    faces = suppress_overlapping_faces(faces, overlap_threshold=0.35)
    results = []
    for (x, y, w, h) in faces:
        face_roi_gray = gray_full[y:y + h, x:x + w]
        face_roi_color = image[y:y + h, x:x + w]
        if not is_valid_face_roi(face_roi_gray, face_roi_color):
            continue
        face_resized = resize_image(face_roi_gray, 48, 48)
        features = extract_combined_features(face_resized).reshape(1, -1)
        prediction = svm_model.predict(features)
        emotion = label_encoder.inverse_transform(prediction)[0]
        results.append({"box": (x, y, w, h), "emotion": emotion})
    return results


def handle_image_upload(uploaded_image):
    uploaded_image.seek(0)
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Gagal membaca gambar. Format tidak didukung.")

    detections = predict_emotion_from_image(image)
    annotated = annotate_image(image, detections) if detections else image.copy()
    square_image = make_square_image(annotated)
    return square_image, detections


def process_video_upload(uploaded_video, video_placeholder, progress_placeholder, status_placeholder, summary_placeholder, frame_skip_rate=1, target_fps=None):
    uploaded_video.seek(0)
    filename = getattr(uploaded_video, "name", "video")
    _, ext = os.path.splitext(filename)
    suffix = ext if ext else ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
        tfile.write(uploaded_video.read())
        temp_video_path = tfile.name

    cap = cv2.VideoCapture(temp_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    effective_fps = float(target_fps) if target_fps else source_fps
    delay = 1.0 / effective_fps
    fps_note = f"FPS: {effective_fps:.1f} ({'custom' if target_fps else 'asli'})"

    emotion_counter = Counter()
    processed_frames = 0

    progress_bar = progress_placeholder.progress(0.0) if total_frames else None
    if not progress_bar:
        progress_placeholder.info("Jumlah frame tidak tersedia, memproses...")

    display_card(
        summary_placeholder,
        "Ringkasan Ekspresi Video",
        "<p>Memproses video, harap tunggu...</p>",
        "info",
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frames += 1

            if processed_frames % frame_skip_rate != 0:
                if progress_bar and processed_frames % 10 == 0:
                    progress_bar.progress(min(processed_frames / total_frames, 1.0))
                continue

            detections = predict_emotion_from_image(frame)

            if detections:
                for result in detections:
                    x, y, w, h = result["box"]
                    draw_emotion_box(frame, x, y, w, h, result["emotion"])
                    emotion_counter[result["emotion"]] += 1

            square_frame = make_square_image(frame)
            video_placeholder.image(
                cv2.cvtColor(square_frame, cv2.COLOR_BGR2RGB),
                caption=f"Frame {processed_frames}/{total_frames or '?'} | {fps_note}",
                use_container_width=True,
            )
            status_placeholder.info(
                f"Memproses frame {processed_frames} dari {total_frames or '?'} ({fps_note})"
            )

            if progress_bar:
                progress_bar.progress(min(processed_frames / total_frames, 1.0))

            time.sleep(delay)
    finally:
        cap.release()
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

    if progress_bar:
        progress_bar.progress(1.0)

    status_placeholder.success(f"Pemrosesan selesai ({processed_frames} frame).")
    summary_variant = "success" if emotion_counter else "warning"
    display_card(
        summary_placeholder,
        "Ringkasan Ekspresi Video",
        summarize_video_counter(emotion_counter),
        summary_variant,
    )


# ==========================
# UI STREAMLIT
# ==========================
st.markdown("<div class='page-shell'>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero-wrap">
        <div class="hero-card">
            <div class="chip chip-soft">Next Gen Expression Detection</div>
            <h1>Deteksi Ekspresi Manusia Berbasis <span class="text-glow">HOG, LBP, SVM</span></h1>
            <p>Klasifikasi ekspresi wajah berdasarkan gambar, video pendek, maupun realtime.</p>
            <div class="spacer-sm"></div>
            <div class="pill-row">
                <span class="pill outline">HOG + LBP</span>
                <span class="pill outline">SVM RBF</span>
                <span class="pill outline">Face Quality Gate</span>
            </div>
        </div>
        <div class="hero-card hero-side">
            <div class="metric-grid">
                <div class="metric-card large">
                    <p class="metric-label">Output Ekspresi</p>
                    <p class="metric-value">7 kelas</p>
                    <p class="metric-sub">Happy, Sad, Angry, Neutral, Surprise, Fear, Disgust</p>
                </div>
                <div class="metric-card">
                    <p class="metric-label">Performa</p>
                    <p class="metric-value">Ringan</p>
                    <p class="metric-sub">FPS custom & frame skip</p>
                </div>
                <div class="metric-card">
                    <p class="metric-label">Resolusi Input</p>
                    <p class="metric-value">48x48 px</p>
                    <p class="metric-sub">Grayscale, siap proses</p>
                </div>
            </div>
                <div class="band">
                <span class="chip chip-soft">Skin Tone Filter</span>
                <span class="chip chip-soft">Blur & Noise Gate</span>
                <span class="chip chip-soft">Multi-Face Ready</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="mode-grid">
        <div class="mode-card">
            <p class="chip chip-line">Gambar</p>
            <h4>Foto cepat</h4>
            <p>Unggah potret wajah lalu lihat sorotan ekspresi dengan highlight warna.</p>
            <div class="mode-chips">
                <span class="chip chip-soft">JPG/PNG</span>
                <span class="chip chip-soft">Auto Crop</span>
                <span class="chip chip-soft">Output Ekspresi</span>
            </div>
        </div>
        <div class="mode-card">
            <p class="chip chip-line">Video</p>
            <h4>Ringkas frame</h4>
            <p>Ideal untuk klip &lt; 1 menit dengan kontrol FPS dan frame skip.</p>
            <div class="mode-chips">
                <span class="chip chip-soft">MP4/AVI/MOV</span>
                <span class="chip chip-soft">Progress bar</span>
                <span class="chip chip-soft">Preview</span>
            </div>
        </div>
        <div class="mode-card">
            <p class="chip chip-line">Realtime</p>
            <h4>Streaming webcam</h4>
            <p>Pantau ekspresi wajah secara langsung dari kamera dengan highlight warna.</p>
            <div class="mode-chips">
                <span class="chip chip-soft">Pilih kamera</span>
                <span class="chip chip-soft">Frame skip</span>
                <span class="chip chip-soft">Status live</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_image, tab_video, tab_realtime = st.tabs(["Gambar", "Video", "Realtime"])

# ======== TAB GAMBAR ========
with tab_image:
    st.markdown(
        "<div class='section-heading'><h3>Deteksi Ekspresi dari Gambar</h3><p class='section-note'>Unggah potret wajah lalu jalankan deteksi untuk ringkasan ekspresi.</p></div>",
        unsafe_allow_html=True,
    )

    with st.form("image_form"):
        upload_col, button_col = st.columns([4, 1])
        with upload_col:
            st.caption("Pilih gambar (JPG, JPEG, PNG)")
            image_file = st.file_uploader(
                "Pilih gambar",
                type=["jpg", "jpeg", "png"],
                key="image_uploader",
                label_visibility="collapsed",
            )
        with button_col:
            st.write("")
            st.markdown("<div class='form-button-container'>", unsafe_allow_html=True)
            detect_image = st.form_submit_button("Deteksi Ekspresi")
            st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    image_display_col, image_info_col = st.columns([5, 2], gap="large")
    image_placeholder = image_display_col.empty()
    caption_placeholder = image_display_col.empty()
    image_placeholder.markdown(
        "<div class='result-frame'><p>Belum ada gambar yang dianalisis.</p></div>",
        unsafe_allow_html=True,
    )

    summary_placeholder = image_info_col.empty()
    display_card(
        summary_placeholder,
        "Ringkasan Ekspresi",
        "<p>Unggah gambar kemudian jalankan deteksi.</p>",
        "info",
    )

    if detect_image:
        if image_file is None:
            display_card(
                summary_placeholder,
                "Ringkasan Ekspresi",
                "<p>Silakan unggah file gambar terlebih dahulu.</p>",
                "warning",
            )
        else:
            try:
                processed_image, detections = handle_image_upload(image_file)
                image_placeholder.image(
                    cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                )
                caption_placeholder.caption(f"Hasil dari: {image_file.name}")

                if detections:
                    summary_html = (
                        f"<p><strong>{len(detections)}</strong> wajah terdeteksi.</p>"
                        + summarize_emotions(detections)
                    )
                    display_card(summary_placeholder, "Ringkasan Ekspresi", summary_html, "success")
                else:
                    display_card(
                        summary_placeholder,
                        "Ringkasan Ekspresi",
                        "<p>Tidak ada wajah terdeteksi pada gambar.</p>",
                        "warning",
                    )
            except Exception as err:
                display_card(
                    summary_placeholder,
                    "Ringkasan Ekspresi",
                    f"<p>Terjadi kesalahan: {err}</p>",
                    "error",
                )

# ======== TAB VIDEO ========
with tab_video:
    st.markdown(
        "<div class='section-heading'><h3>Deteksi Ekspresi dari Video</h3><p class='section-note'>Gunakan video berdurasi kurang dari 1 menit agar tetap responsif.</p></div>",
        unsafe_allow_html=True,
    )

    target_fps_value = None
    
    upload_col, button_col = st.columns([4, 1])
    with upload_col:
        st.caption("Pilih video (MP4, AVI, MOV)")
        video_file = st.file_uploader(
            "Pilih video",
            type=["mp4", "avi", "mov"],
            key="video_uploader",
            label_visibility="collapsed",
        )
        
        if video_file:
            fps_mode = st.radio(
                "Pengaturan FPS",
                ["Ikuti FPS asli", "Atur FPS manual"],
                index=0,
                horizontal=True,
                help="Gunakan FPS asli video atau tetapkan FPS target untuk pengaturan jeda antar-frame.",
            )
            if fps_mode == "Atur FPS manual":
                target_fps_value = st.slider(
                    "FPS target",
                    min_value=5,
                    max_value=60,
                    value=24,
                    help="Tidak mengubah jumlah frame yang dianalisis, hanya kecepatan pemutaran/pembaruan UI.",
                )
            
            frame_skip_rate = st.slider(
                "Frame Skip Rate",
                min_value=1,
                max_value=10,
                value=3,
                help="Proses setiap N frame. Nilai lebih tinggi = lebih cepat tapi kurang akurat.",
            )
        else:
            frame_skip_rate = 1

    with button_col:
        st.write("")
        st.markdown("<div class='form-button-container'>", unsafe_allow_html=True)
        if video_file:
            col_start, col_stop = st.columns(2)
            with col_start:
                if st.button("Proses Video", use_container_width=True):
                    st.session_state.processing_video = True
            with col_stop:
                if st.button("Stop Video", use_container_width=True):
                    st.session_state.processing_video = False
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    video_display_col, video_info_col = st.columns([5, 2], gap="large")
    video_placeholder = video_display_col.empty()
    video_placeholder.markdown(
        "<div class='result-frame'><p>Belum ada video yang diputar.</p></div>",
        unsafe_allow_html=True,
    )

    with video_info_col:
        progress_placeholder = st.container()
        status_placeholder = st.empty()
        summary_placeholder = st.empty()
        status_placeholder.info("Belum ada video yang diproses.")
        display_card(
            summary_placeholder,
            "Ringkasan Ekspresi Video",
            "<p>Unggah video kemudian jalankan pemrosesan.</p>",
            "info",
        )

    if "processing_video" not in st.session_state:
        st.session_state.processing_video = False

    if st.session_state.processing_video:
        if video_file is None:
            status_placeholder.warning("Silakan unggah file video terlebih dahulu.")
            st.session_state.processing_video = False
        else:
            process_video_upload(
                video_file,
                video_placeholder,
                progress_placeholder,
                status_placeholder,
                summary_placeholder,
                frame_skip_rate,
                target_fps_value,
            )

# ======== TAB REALTIME ========
with tab_realtime:
    st.markdown(
        "<div class='section-heading'><h3>Deteksi Ekspresi Realtime</h3><p class='section-note'>Gunakan webcam untuk mendeteksi ekspresi secara langsung.</p></div>",
        unsafe_allow_html=True,
    )

    # Helper untuk mendeteksi kamera
    def get_available_cameras(max_devices=6):
        available = []
        for i in range(max_devices):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(f"Camera {i}")
            cap.release()
        return available

    if "available_cameras" not in st.session_state:
        st.session_state.available_cameras = get_available_cameras()

    if "run_camera" not in st.session_state:
        st.session_state.run_camera = False

    camera_index = None
    col_cam, col_info = st.columns([3, 1])
    
    with col_info:
        st.write("")
        camera_options = st.session_state.available_cameras
        if camera_options:
            camera_option = st.selectbox(
                "Pilih Kamera",
                camera_options,
                index=0,
                key="camera_select",
            )
            camera_index = int(camera_option.split(" ")[1])
        else:
            st.selectbox(
                "Pilih Kamera",
                ["Tidak ada kamera terdeteksi"],
                index=0,
                disabled=True,
            )
            st.info("Webcam tidak ditemukan. Hubungkan perangkat lalu klik tombol refresh kamera.")
            st.session_state.run_camera = False

        if st.button("Refresh Kamera", key="refresh_cam"):
            st.session_state.available_cameras = get_available_cameras()
            st.session_state.run_camera = False
            st.rerun()

        realtime_skip_rate = st.slider(
            "Frame Skip Rate",
            min_value=1,
            max_value=10,
            value=3,
            key="realtime_skip",
            help="Deteksi ekspresi setiap N frame. Video tetap lancar, deteksi lebih ringan."
        )

        st.write("")
        start_disabled = camera_index is None
        if st.button("Mulai Kamera", key="start_cam", disabled=start_disabled):
            st.session_state.run_camera = True
        
        if st.button("Stop Kamera", key="stop_cam"):
            st.session_state.run_camera = False
            st.rerun()
        
        st.markdown("---")
        st.caption("Status:")
        status_text = st.empty()
        if st.session_state.run_camera:
            status_text.success("Kamera aktif.")
        elif camera_index is None:
            status_text.warning("Tidak ada kamera yang tersedia.")
        else:
            status_text.info("Siap.")

    with col_cam:
        cam_placeholder = st.empty()
        if not st.session_state.run_camera:
            cam_placeholder.markdown(
                "<div class='result-frame'><p>Kamera belum aktif.</p></div>",
                unsafe_allow_html=True,
            )

    if st.session_state.run_camera and camera_index is not None:
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            status_text.error(f"Gagal membuka {camera_option}.")
            st.session_state.run_camera = False
        else:
            frame_count = 0
            last_detections = []
            
            while st.session_state.run_camera:
                ret, frame = cap.read()
                if not ret:
                    status_text.error("Gagal membaca frame.")
                    break
                
                frame_count += 1
                
                # Flip frame agar seperti cermin
                frame = cv2.flip(frame, 1)
                
                # Deteksi ekspresi (skip frame logic)
                if frame_count % realtime_skip_rate == 0 or frame_count == 1:
                    last_detections = predict_emotion_from_image(frame)
                
                # Gambar kotak (gunakan hasil deteksi terakhir)
                if last_detections:
                    for result in last_detections:
                        x, y, w, h = result["box"]
                        draw_emotion_box(frame, x, y, w, h, result["emotion"])
                
                # Tampilkan
                square_frame = make_square_image(frame)
                cam_placeholder.image(
                    cv2.cvtColor(square_frame, cv2.COLOR_BGR2RGB),
                    caption=f"Realtime Feed ({camera_option})",
                    use_container_width=True,
                )
                
                # Sedikit delay untuk mengurangi beban CPU
                time.sleep(0.03)
            
            cap.release()
    elif st.session_state.run_camera and camera_index is None:
        status_text.error("Tidak ada kamera yang dapat digunakan.")
        st.session_state.run_camera = False


st.markdown("</div>", unsafe_allow_html=True)
