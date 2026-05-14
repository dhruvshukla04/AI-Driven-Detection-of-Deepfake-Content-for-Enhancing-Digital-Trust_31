# ─────────────────────────────────────────────
# DeepFake Detector — Startup Script
# Team 31, Chitkara University
# ─────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   AI-Driven DeepFake Detection System            ║"
echo "║   Team 31 · Chitkara University                  ║"
echo "║   Stack: PyTorch · TF.js · Brain.js · OpenCV     ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 not found. Please install Python 3.9+"
    exit 1
fi

# Install dependencies
echo "[SETUP] Installing dependencies..."
pip install -r requirements.txt -q

echo ""
echo "[START] Starting DeepFake Detection Server..."
echo "[INFO]  Open your browser at: http://localhost:5000"
echo "[INFO]  Press Ctrl+C to stop"
echo ""

cd backend
python3 app.py
