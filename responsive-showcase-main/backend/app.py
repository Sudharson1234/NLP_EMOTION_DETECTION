import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from datetime import datetime
import uuid
import onnxruntime as ort

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load ONNX Model (Emotion FERPlus - pre-trained on FER+ dataset)
print("Loading ONNX emotion model...")
try:
    session = ort.InferenceSession("emotion_model.onnx", providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"✅ ONNX model loaded! Input: {input_name}, Output: {output_name}")
    print(f"   Input shape: {session.get_inputs()[0].shape}")
except Exception as e:
    print(f"❌ Failed to load ONNX model: {e}")
    session = None

# FERPlus emotions (8 classes)
emotion_labels = ["Neutral", "Happy", "Surprise", "Sad", "Angry", "Disgust", "Fear", "Contempt"]

# Haar face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CAPTURED_FOLDER = "captured"
os.makedirs(CAPTURED_FOLDER, exist_ok=True)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    filename = secure_filename(file.filename)

    if filename == "":
        filename = f"{uuid.uuid4().hex}.jpg"

    img_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(img_path)

    img_color = cv2.imread(img_path)
    if img_color is None:
        return jsonify({"error": "Invalid image"}), 400

    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        return jsonify({
            "face_detected": False,
            "emotion": "No Face Detected",
            "confidence": 0
        })
    
    (x, y, w, h) = faces[0]
    
    # Extract face region
    face = gray[y:y+h, x:x+w]
    
    # Preprocess for FERPlus ONNX model (64x64 grayscale)
    face = cv2.resize(face, (64, 64))
    face = face.astype(np.float32)
    
    # Normalize to [0, 1]
    face = face / 255.0
    
    # Reshape to NCHW format: (1, 1, 64, 64)
    face = np.expand_dims(face, axis=0)  # (1, 64, 64)
    face = np.expand_dims(face, axis=0)  # (1, 1, 64, 64)

    try:
        if session is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        # Run ONNX inference
        outputs = session.run([output_name], {input_name: face})
        prediction = outputs[0][0]
        
        # Apply softmax
        exp_pred = np.exp(prediction - np.max(prediction))
        probabilities = exp_pred / exp_pred.sum()
        
        max_index = int(np.argmax(probabilities))
        emotion = emotion_labels[max_index]
        confidence = float(probabilities[max_index] * 100)
        
        print(f"✅ Detected: {emotion} ({confidence:.1f}%)")

        return jsonify({
            "face_detected": True,
            "emotion": emotion,
            "confidence": round(confidence, 2)
        })
    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/capture", methods=["POST"])
def capture():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    emotion = request.form.get("emotion", "Unknown").replace(" ", "_")
    confidence = request.form.get("confidence", "0")

    try:
        confidence = int(float(confidence))
    except:
        confidence = 0

    filename = f"{emotion}_{confidence}%_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}.jpg"
    save_path = os.path.join(CAPTURED_FOLDER, filename)
    file.save(save_path)

    return jsonify({"saved": True, "file": filename})

@app.route("/")
def home():
    return "✅ Flask Backend with ONNX Model is RUNNING!"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    emotion = data.get("emotion", "Neutral")
    
    replies = {
        "Happy": "Great to see you happy! What made your day good?",
        "Sad": "I'm here with you — what happened?",
        "Angry": "Take a breath. What triggered this?",
        "Fear": "You are safe. What are you worried about?",
        "Surprise": "That looks unexpected! Tell me more.",
        "Neutral": "How are you feeling right now?",
        "Disgust": "Something seems off. Want to share?",
        "Contempt": "I sense some skepticism. What's on your mind?"
    }

    return jsonify({"reply": replies.get(emotion, "I'm here to help you.")})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
