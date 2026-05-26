# Real-Time Face Recognition 👁️

A real-time face recognition system built with Python and OpenCV that detects and identifies multiple faces simultaneously from a live webcam feed. Features voice announcements, live FPS display, and a clean database management workflow.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-green?style=flat&logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)

---

## 📸 Features

- ✅ Real-time detection of **multiple faces** in a single frame
- ✅ **LBPH algorithm** for fast, accurate recognition
- ✅ **Voice announcements** via pyttsx3 (non-blocking, queue-based)
- ✅ Live **FPS counter** and face count HUD
- ✅ **Frame skipping + downscale** for smooth performance
- ✅ Easy database management — add any number of people
- ✅ Works fully **offline** — no cloud API required

---

## 🗂️ Project Structure

```
A-real-time-face-recognition/
│
├── create_database.py        # Capture face images for a person
├── face_rec.py               # Main real-time recognition script
├── requirements.txt          # Python dependencies
├── .gitignore
│
├── database/                 # Auto-created — stores face images
│   ├── Pushkar/
│   │   ├── 1.png
│   │   └── ...
│   └── AnotherPerson/
│
└── haarcascade_frontalface_default.xml   # Downloaded separately (see setup)
```

---

## ⚙️ Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/Pushkar745/A-real-time-face-recognition.git
cd A-real-time-face-recognition
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the Haar Cascade file
```bash
curl -o haarcascade_frontalface_default.xml https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
```

---

## 🚀 Usage

### Step 1 — Add a person to the database
```bash
python create_database.py YourName
```
- A webcam window opens
- Look at the camera and vary your expressions
- **45 images** are captured automatically
- Repeat for each person you want to recognize

### Step 2 — Start face recognition
```bash
python face_rec.py
```
- The system trains on your database and starts the webcam
- Recognized faces get a **green box** with the person's name
- Unknown faces get a **red box**
- Press **ESC** to quit

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.8+ |
| Computer Vision | OpenCV 4.9 |
| Face Detection | Haar Cascade Classifier |
| Face Recognition | LBPH (Local Binary Pattern Histogram) |
| Text-to-Speech | pyttsx3 |
| Numerical Computing | NumPy |

---

## ⚡ Performance Optimizations

- **Frame skipping** — face detection runs every 2nd frame; recognition uses cached coordinates in between
- **Downscale detection** — frames are halved in size before detection (4× fewer pixels to scan)
- **Async TTS** — voice announcements run in a dedicated background thread via a queue, so they never freeze the video
- **Webcam locked** to 640×480 @ 30fps for consistent performance

---

## 🔧 Configuration

All key settings are at the top of `face_rec.py`:

```python
CONFIDENCE_THRESHOLD = 85    # Lower = stricter matching
ANNOUNCE_COOLDOWN    = 5     # Seconds between voice announcements
FRAME_SKIP           = 2     # Detect every Nth frame
DETECT_SCALE         = 0.5   # Shrink factor for detection (speed)
```

---

## 🔮 Roadmap

- [ ] Save/load trained model (skip retraining on startup)
- [ ] Swap Haar cascade for OpenCV DNN detector (better accuracy)
- [ ] Attendance logging to CSV with timestamps
- [ ] Liveness detection (anti-spoofing)
- [ ] Flask REST API + web dashboard
- [ ] Package as standalone executable (.exe)

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙋 Author

**Pushkar Baviskar**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://in.linkedin.com/in/pushkarbaviskar)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-blue?style=flat&logo=google-chrome)](https://pushkarbaviskar.azurewebsites.net/)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?style=flat&logo=twitter)](https://twitter.com/hackepushkar)
