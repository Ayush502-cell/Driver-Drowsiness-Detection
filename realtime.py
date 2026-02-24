# ============================================================
#  FILE: realtime.py
#  TEAMMATE 2 — Real-Time Drowsiness Detection
#  Purpose: Load saved model and detect drowsiness via webcam
#  Dependency: drowsy_model.h5 (produced by model.py)
# ============================================================

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import os

# ─── CONFIG ────────────────────────────────────────────────
MODEL_PATH = "drowsy_model.h5"
IMG_SIZE = (128, 128)
THRESHOLD = 0.5  # above 0.5 = Alert, below = Drowsy
ALERT_FRAMES = 20  # how many consecutive drowsy frames before alarm
# ────────────────────────────────────────────────────────────

# Color scheme (BGR format for OpenCV)
COLOR_ALERT = (0, 255, 0)  # Green
COLOR_DROWSY = (0, 0, 255)  # Red
COLOR_TEXT = (255, 255, 255)  # White


def load_drowsy_model():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("   Please run model.py first to train and save the model.")
        exit()
    model = load_model(MODEL_PATH)
    print(f"✔ Model loaded from: {MODEL_PATH}")
    return model


def preprocess_frame(frame):
    """
    Prepares a webcam frame for model prediction.
    Matches the same preprocessing used during training.
    """
    img = cv2.resize(frame, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR, model expects RGB
    img = img / 255.0  # normalize to [0,1]
    img = np.expand_dims(img, axis=0)  # add batch dimension → (1,128,128,3)
    return img


def draw_overlay(frame, label, confidence, is_drowsy, drowsy_frame_count):
    """
    Draws label, confidence bar and alert on the video frame.
    """
    h, w = frame.shape[:2]
    color = COLOR_DROWSY if is_drowsy else COLOR_ALERT

    # Status box at top
    cv2.rectangle(frame, (0, 0), (w, 60), color, -1)
    cv2.putText(
        frame,
        f"Status: {label}",
        (15, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        COLOR_TEXT,
        2,
        cv2.LINE_AA,
    )

    # Confidence score
    conf_text = f"Confidence: {confidence:.1%}"
    cv2.putText(
        frame,
        conf_text,
        (15, h - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )

    # Confidence bar
    bar_width = int(confidence * (w - 30))
    cv2.rectangle(frame, (15, h - 35), (w - 15, h - 15), (80, 80, 80), -1)
    cv2.rectangle(frame, (15, h - 35), (15 + bar_width, h - 15), color, -1)

    # ALERT warning if drowsy for too long
    if is_drowsy and drowsy_frame_count >= ALERT_FRAMES:
        cv2.rectangle(frame, (0, h // 2 - 40), (w, h // 2 + 40), (0, 0, 200), -1)
        cv2.putText(
            frame,
            "⚠ DROWSINESS ALERT! ⚠",
            (w // 2 - 200, h // 2 + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 0),
            3,
            cv2.LINE_AA,
        )

    return frame


def run_realtime_detection(model):
    """
    Main loop: opens webcam, predicts each frame, shows result.
    Press 'q' to quit.
    """
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    if not cap.isOpened():
        print("❌ Could not open webcam. Check your camera connection.")
        return

    print("\n🎥 Webcam started. Press 'q' to quit.")

    drowsy_frame_count = 0
    fps_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame from webcam.")
            break

        # Preprocess and predict
        processed = preprocess_frame(frame)
        prediction = model.predict(processed, verbose=0)[0][0]

        # class_indices from training: Alert=0, Drowsy=1
        # Sigmoid output: closer to 1 = Alert, closer to 0 = Drowsy
        # Adjust based on your class mapping from data_pipeline.py
        is_drowsy = prediction < THRESHOLD
        label = "ALERT 😊" if not is_drowsy else "DROWSY 😴"
        confidence = (1 - prediction) if is_drowsy else prediction

        # Count consecutive drowsy frames
        if is_drowsy:
            drowsy_frame_count += 1
        else:
            drowsy_frame_count = 0

        # Draw overlay on frame
        frame = draw_overlay(frame, label, confidence, is_drowsy, drowsy_frame_count)

        # Show FPS
        frame_count += 1
        if time.time() - fps_time >= 1.0:
            fps = frame_count / (time.time() - fps_time)
            frame_count = 0
            fps_time = time.time()
        cv2.putText(
            frame,
            f"FPS: {frame_count}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
        )

        cv2.imshow("DrowsyDetectNet — Real-Time Detection", frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n👋 Detection stopped by user.")
            break

    cap.release()
    cv2.destroyAllWindows()


# ─── MAIN ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("   DrowsyDetectNet — REAL-TIME DETECTION")
    print("   TEAMMATE 2")
    print("=" * 50)

    model = load_drowsy_model()
    run_realtime_detection(model)
