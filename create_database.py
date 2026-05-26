# create_database.py
import cv2
import sys
import os
import time

# ── Config ────────────────────────────────────────────────────────────────────
TOTAL_IMAGES = 45
SIZE_FACTOR   = 4
IM_WIDTH, IM_HEIGHT = 112, 92
DATABASE_DIR  = 'database'
HAAR_CASCADE  = 'haarcascade_frontalface_default.xml'
CAPTURE_DELAY = 0.38   # seconds between captures
# ──────────────────────────────────────────────────────────────────────────────

def get_next_index(folder_path):
    """Return the next available image index in the folder."""
    existing = [
        int(f.split('.')[0])
        for f in os.listdir(folder_path)
        if f[0] != '.' and f.split('.')[0].isdigit()
    ]
    return max(existing, default=0) + 1


def main():
    if len(sys.argv) < 2:
        print("Usage: python create_database.py <person_name>")
        sys.exit(1)

    person_name = sys.argv[1]
    save_path = os.path.join(DATABASE_DIR, person_name)
    os.makedirs(save_path, exist_ok=True)

    # Load Haar cascade
    if not os.path.exists(HAAR_CASCADE):
        print(f"Error: '{HAAR_CASCADE}' not found. Download it from OpenCV GitHub.")
        sys.exit(1)
    haar_cascade = cv2.CascadeClassifier(HAAR_CASCADE)

    # Open webcam
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)

    print("─" * 55)
    print(f"  Capturing 45 images for: {person_name}")
    print("  Look at the camera and vary your expressions!")
    print("  Press ESC to stop early.")
    print("─" * 55)

    count = 0
    while count < TOTAL_IMAGES:
        ret, frame = webcam.read()
        if not ret:
            print("Warning: Failed to read frame. Retrying...")
            continue

        frame   = cv2.flip(frame, 1)
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mini    = cv2.resize(gray, (gray.shape[1] // SIZE_FACTOR,
                                    gray.shape[0] // SIZE_FACTOR))

        faces = haar_cascade.detectMultiScale(
            mini,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) > 0:
            # Pick the largest face
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            (x, y, w, h) = [v * SIZE_FACTOR for v in faces[0]]

            face_gray   = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face_gray, (IM_WIDTH, IM_HEIGHT))

            # Save image
            idx = get_next_index(save_path)
            img_path = os.path.join(save_path, f"{idx}.png")
            cv2.imwrite(img_path, face_resized)
            count += 1

            # Draw on preview
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{person_name} [{count}/{TOTAL_IMAGES}]",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            time.sleep(CAPTURE_DELAY)

        else:
            cv2.putText(frame, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Progress bar in terminal
        progress = int((count / TOTAL_IMAGES) * 30)
        bar = f"[{'█' * progress}{'░' * (30 - progress)}] {count}/{TOTAL_IMAGES}"
        print(f"\r  {bar}", end='', flush=True)

        cv2.imshow('Create Database - Press ESC to quit', frame)
        if cv2.waitKey(10) == 27:
            print("\n  Stopped early by user.")
            break

    webcam.release()
    cv2.destroyAllWindows()
    print(f"\n✅  {count} images saved to '{save_path}'")


if __name__ == '__main__':
    main()