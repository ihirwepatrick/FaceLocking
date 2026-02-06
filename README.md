# Face Recognition with Real-time Tracking

A real-time face recognition system with face locking capabilities, designed to identify and track specific individuals while monitoring their actions.

## Core Features
- **Face Recognition**: 
  - Real-time face detection and identification
  - Multi-person support
  - High accuracy with ArcFace embeddings

- **Face Locking**:
  - Track specific individuals in real-time
  - Monitor head movements (left/right)
  - Detect smiles and facial expressions
  - Automatic action logging with timestamps

## Quick Start

### 1. Setup Environment
```bash
# Create virtual environment (recommended for isolation)
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Models
The system requires two models in the `models/` directory:
- `embedder_arcface.onnx` - Face recognition model
- `face_landmarker.task` - Facial landmark detection

Download links:
- **ArcFace**: [HuggingFace](https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx) (rename to `embedder_arcface.onnx`)
- **FaceMesh**: [Google Storage](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)

### 3. Initialize Project
```bash
python init_project.py
```

## Usage

### Enroll New Faces
Capture face samples and add them to the database:
```bash
python -m src.enroll
```
**Controls**: `SPACE` = capture | `a` = auto-capture | `s` = save | `q` = quit

### Run Face Recognition
Start real-time face recognition with tracking:
```bash
python -m src.recognize
```
**Controls**: `l` = lock face | `+/-` = adjust threshold | `r` = reload DB | `d` = debug | `q` = quit

### Rebuild Database
Rebuild the face database from existing enrollment crops:
```bash
python -m src.rebuild_db
```

### Evaluate Performance
Test recognition accuracy and find optimal threshold:
```bash
python -m src.evaluate
```

## Face Locking Features

When you press `l` during recognition:
- **Locks onto** the currently recognized person
- **Tracks** their movements even with multiple people
- **Detects actions**:
  - Head movements (left/right)
  - Smile detection
  - Face lock/unlock events
- **Logs everything** to `logs/[Name]_history_[timestamp].txt`

**Example log entries**:
```
2026-01-31 13:20:11.324267 - FACE_LOCKED: Face locked: Patrick
2026-01-31 13:20:20.225528 - HEAD_RIGHT: Moved right by 31.9px
2026-01-31 13:20:21.684933 - SMILE: Smile detected (ratio: 14.97)
```

## Tips

**Recognition Accuracy**:
- Use `+` key to increase threshold if faces aren't recognized
- Use `-` key to decrease threshold if getting false positives
- Default threshold (0.40) works well for most cases
- System uses temporal smoothing for stable recognition

**Face Enrollment**:
- Capture 10-20 samples per person
- Vary expressions and angles slightly
- Ensure good, stable lighting
- Use `a` for auto-capture mode

**Removing Faces**:
1. Delete the person's folder from `data/enroll/[Name]`
2. Run `python -m src.rebuild_db`

## Technical Details
- **Face Recognition**: ArcFace embeddings (512-dimensional)
- **Face Detection**: Haar Cascade + MediaPipe FaceMesh
- **Landmark Tracking**: 5-point facial landmarks
- **Performance**: Optimized for CPU, runs at 15-30 FPS
- **Python**: Compatible with 3.10, 3.11, 3.12, 3.13
