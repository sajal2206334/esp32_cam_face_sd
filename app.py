from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
import time
import mediapipe as mp

app = Flask(__name__)

# Ensure static folder exists
STATIC_DIR = 'static'
os.makedirs(STATIC_DIR, exist_ok=True)
LATEST_IMAGE_PATH = os.path.join(STATIC_DIR, 'latest.jpg')

# MediaPipe setup
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

@app.route('/')
def index():
    return 'ESP32-CAM Server Running'

@app.route('/upload', methods=['POST'])
def upload():
    image = cv2.imdecode(np.frombuffer(request.data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    # Detect face
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_image)

    # Save latest image
    cv2.imwrite(LATEST_IMAGE_PATH, image)

    print(f"[UPLOAD] Image received at {time.ctime()}")
    if results.detections:
        print(f"[DETECTED] {len(results.detections)} face(s) found")
        return jsonify({"face_detected": True}), 200
    else:
        print("[NOT DETECTED] No face found")
        return jsonify({"face_detected": False}), 200

@app.route('/view')
def view():
    return '''
    <html>
    <head>
        <title>ESP32-CAM Live Image Viewer</title>
        <style>
            body { font-family: sans-serif; text-align: center; background: #f9f9f9; }
            img { max-width: 90vw; border: 4px solid #333; margin-top: 20px; }
        </style>
        <script>
            function refreshImage() {
                const img = document.getElementById("cam");
                img.src = "/static/latest.jpg?t=" + new Date().getTime();
            }
            function handleImageError() {
                const img = document.getElementById("cam");
                img.alt = "No image received yet.";
            }
            setInterval(refreshImage, 2000);
        </script>
    </head>
    <body>
        <h1>Latest Image from ESP32-CAM</h1>
        <img id="cam" src="/static/latest.jpg" alt="Latest Frame" onerror="handleImageError()">
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)