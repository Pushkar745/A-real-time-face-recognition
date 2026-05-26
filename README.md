# Real-Time Face Recognition 👁️

A real-time face recognition system that identifies or verifies a person from a live video frame using Python and OpenCV.

## 🎯 What it does

- Detects faces in real time from a webcam or video stream
- Identifies or verifies a person against a known set of faces
- Runs locally — no cloud API required

## 🛠️ Tech Stack

- **Python**
- **OpenCV** — video capture and image processing
- **face_recognition** library (dlib-based)

## ⚙️ Getting Started

### Prerequisites

```bash
pip install opencv-python face_recognition numpy
```

### Run the app

```bash
git clone https://github.com/Pushkar745/A-real-time-face-recognition.git
cd A-real-time-face-recognition
python face_recognition_app.py
```

> Make sure your webcam is connected and accessible.

## 📁 Project Structure

```
├── known_faces/        # Folder with reference images
├── face_recognition_app.py
└── README.md
```

## 🔮 Future improvements

- [ ] Add a GUI for managing known faces
- [ ] Support video file input (not just webcam)
- [ ] Export recognition logs to CSV

## 📄 License

MIT
