# AI-Driven DeepFake Detection System
### Team 31 — Chitkara University Institute of Engineering & Technology

> Implementation of the paper:  
> **"AI-Driven Detection of Deepfake Content for Enhancing Digital Trust"**  
> Sukhpreet Singh, Dhruv Shukla, Dhruv Kinger, Gourav Rana

---

## Tech Stack (Exact as per Paper)

| Component | Technology | Paper Section |
|-----------|-----------|---------------|
| CNN Backbone | PyTorch + Keras | §III-C |
| Transfer Learning | EfficientNet-B4 | §III-C |
| Face Detection | OpenCV Haar Cascade + dlib HOG | §III-B |
| Image Processing | PIL (Pillow) + scikit-image | §III-B |
| Browser Pre-filter | Brain.js MLP | §III-D |
| Browser Inference | TensorFlow.js | §III-D |
| Backend Server | Flask (Python) | — |

---

## Project Structure

```
deepfake_detector/
├── backend/
│   └── app.py              ← Flask server, PyTorch CNN, EfficientNet-B4 ensemble
├── frontend/
│   └── templates/
│       └── index.html      ← TensorFlow.js + Brain.js browser UI
├── models/                 ← Place trained .pth weights here
├── requirements.txt
└── README.md
```

---

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Download pre-trained weights
Place your trained model weights in `models/`:
- `models/cnn_deepfake.pth`       ← Custom CNN (~8.2M params)
- `models/efficientnet_b4.pth`    ← EfficientNet-B4 (~19.3M params)

If no weights are provided, models run with random initialization  
(useful for testing the pipeline / UI).

### 3. Start the server
```bash
cd backend
python app.py
```

### 4. Open browser
```
http://localhost:5000
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /` | GET | Main UI |
| `/api/analyze/image` | POST | Analyze single image (base64 JSON) |
| `/api/analyze/video` | POST | Analyze video (multipart, 5fps extraction) |
| `/api/model/info` | GET | Full model architecture & paper metrics |

### Example — Image Analysis
```python
import requests, base64

with open("face.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

res = requests.post("http://localhost:5000/api/analyze/image",
                    json={"image": f"data:image/jpeg;base64,{b64}"})
print(res.json())
# {
#   "verdict": "FAKE",
#   "fake_probability": 0.832,
#   "confidence": 83.2,
#   "cnn_score": 0.791,
#   "efficientnet_score": 0.856,
#   "face_detected": true,
#   "stage_used": "full_ensemble",
#   "inference_ms": 312.4
# }
```

---

## Two-Stage Inference Pipeline (Paper §III-D)

```
Input Image/Video
      ↓
OpenCV Haar Cascade → Face Detection & Crop (224×224)
      ↓
Custom CNN → 512-dim Embedding
      ↓
Brain.js MLP Gate → Confidence Score
   ↙          ↘
Confident    Ambiguous (0.3–0.7)
  ↓               ↓
Use gate      Full EfficientNet-B4 Ensemble
  score       (40% CNN + 60% EfficientNet)
      ↓
Majority Voting (video) / Single verdict (image)
      ↓
Final: REAL / FAKE + Confidence %
```

---

## Model Performance (from paper)

| Metric | Score |
|--------|-------|
| Accuracy | 94.2% |
| Precision | 93.7% |
| Recall | 92.9% |
| F1 Score | 93.3% |
| AUC-ROC | 0.971 |
| Browser Inference | ~340ms/frame |
| Cross-Validation (5-fold) | 93.5% ± 0.5% |

### Training Setup
- **Optimizer**: Adam (lr=1e-4, cosine annealing to 1e-6)
- **Batch size**: 32
- **Epochs**: 50 (CNN), 30 (EfficientNet fine-tune)
- **Hardware**: NVIDIA RTX 3080 (10 GB VRAM)
- **Datasets**: FaceForensics++ (~420K crops) + DFDC (15K subset)

---

## Training Your Own Weights

To train on FaceForensics++:
1. Download dataset from https://github.com/ondyari/FaceForensics
2. Extract frames: `python backend/extract_frames.py --dataset /path/to/ff++`
3. Train: `python backend/train.py --model efficientnet --epochs 30`

---

*Paper: AI-Driven Detection of Deepfake Content for Enhancing Digital Trust*  

