# face_rec.py
import cv2
import os
import numpy as np
import pyttsx3
import time
import queue
from threading import Thread

# ── Config ────────────────────────────────────────────────────────────────────
HAAR_CASCADE         = 'haarcascade_frontalface_default.xml'
DATABASE_DIR         = 'database'
IM_WIDTH, IM_HEIGHT  = 112, 92
CONFIDENCE_THRESHOLD = 85    # LBPH: lower = stricter. <85 = known, else Unknown
ANNOUNCE_COOLDOWN    = 5     # seconds between voice announcements per person
FRAME_SKIP           = 2     # only detect every Nth frame (speeds up a lot)
DETECT_SCALE         = 0.5   # shrink frame for detection (0.5 = half size = 4x faster)
# ──────────────────────────────────────────────────────────────────────────────


def load_training_data(data_dir):
    images, labels, names = [], [], {}
    person_id = 0
    for subdir in sorted(os.listdir(data_dir)):
        subject_path = os.path.join(data_dir, subdir)
        if not os.path.isdir(subject_path):
            continue
        names[person_id] = subdir
        for filename in os.listdir(subject_path):
            if filename.startswith('.'):
                continue
            img = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(person_id)
        person_id += 1
    return np.array(images), np.array(labels), names


# ── TTS worker — one thread, one engine, processes a queue ───────────────────
tts_queue = queue.Queue()

def tts_worker():
    """Single background thread that owns the TTS engine exclusively."""
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    while True:
        text = tts_queue.get()
        if text is None:       # None = shutdown signal
            break
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass
        tts_queue.task_done()

def announce(name, last_announced):
    """Queue an announcement if cooldown has passed — never blocks."""
    now = time.time()
    if now - last_announced.get(name, 0) > ANNOUNCE_COOLDOWN:
        last_announced[name] = now
        if tts_queue.empty():          # don't pile up if already speaking
            tts_queue.put(f"Hello {name}")
# ──────────────────────────────────────────────────────────────────────────────


def main():
    # ── Validate ──────────────────────────────────────────────────────────────
    if not os.path.exists(HAAR_CASCADE):
        print(f"Error: '{HAAR_CASCADE}' not found.")
        return
    if not os.path.isdir(DATABASE_DIR) or not os.listdir(DATABASE_DIR):
        print(f"Error: Database '{DATABASE_DIR}' is empty. Run create_database.py first.")
        return

    # ── Train ─────────────────────────────────────────────────────────────────
    print("Loading training data...")
    images, labels, names = load_training_data(DATABASE_DIR)
    if len(images) == 0:
        print("Error: No images found in database.")
        return

    print(f"Training LBPH on {len(images)} images across {len(names)} person(s)...")
    model = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8
    )
    model.train(images, labels)
    print("✅ Training complete.\n")

    # ── Start TTS thread ──────────────────────────────────────────────────────
    tts_thread = Thread(target=tts_worker, daemon=True)
    tts_thread.start()

    # ── Setup webcam ──────────────────────────────────────────────────────────
    haar_cascade = cv2.CascadeClassifier(HAAR_CASCADE)

    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("Error: Could not open webcam.")
        return
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    webcam.set(cv2.CAP_PROP_FPS, 30)

    print("Starting face recognition. Press ESC to quit.")
    print("─" * 45)

    last_announced = {}
    cached_faces   = []
    frame_count    = 0
    fps_time       = time.time()
    fps            = 0

    while True:
        ret, frame = webcam.read()
        if not ret:
            continue

        frame      = cv2.flip(frame, 1)
        gray_full  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_count += 1

        # ── Detect every FRAME_SKIP frames ────────────────────────────────────
        if frame_count % FRAME_SKIP == 0:
            small = cv2.resize(gray_full, (0, 0), fx=DETECT_SCALE, fy=DETECT_SCALE)
            detected = haar_cascade.detectMultiScale(
                small,
                scaleFactor=1.2,
                minNeighbors=4,
                minSize=(20, 20),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            cached_faces = (
                [[int(v / DETECT_SCALE) for v in face] for face in detected]
                if len(detected) > 0 else []
            )

        # ── Recognize all faces in frame ──────────────────────────────────────
        for (x, y, w, h) in cached_faces:
            x, y = max(0, x), max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
            if w <= 0 or h <= 0:
                continue

            face_resized = cv2.resize(gray_full[y:y+h, x:x+w], (IM_WIDTH, IM_HEIGHT))

            try:
                label, confidence = model.predict(face_resized)
            except Exception as e:
                print(f"Prediction error: {e}")
                continue

            if confidence < CONFIDENCE_THRESHOLD:
                name  = names.get(label, "Unknown")
                color = (0, 220, 0)
                text  = f"{name}  ({int(confidence)})"
                announce(name, last_announced)
            else:
                name  = "Unknown"
                color = (0, 0, 220)
                text  = "Unknown"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x, y - 26), (x + w, y), color, -1)
            cv2.putText(frame, text, (x + 4, y - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        # ── FPS ───────────────────────────────────────────────────────────────
        if frame_count % 15 == 0:
            fps = 15 / (time.time() - fps_time + 1e-6)
            fps_time = time.time()

        cv2.putText(frame,
                    f"FPS: {fps:.1f}  |  Faces: {len(cached_faces)}  |  "
                    f"People: {len(names)}  |  ESC to quit",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) == 27:
            break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    tts_queue.put(None)    # shutdown TTS thread cleanly
    webcam.release()
    cv2.destroyAllWindows()
    print("Session ended.")


if __name__ == '__main__':
    main()