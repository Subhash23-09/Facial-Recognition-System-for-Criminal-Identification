import numpy as np
import mysql.connector
import cv2
from database import db_config

# Simulated deep learning functions (Replace with ViT/FaceNet)
def detect_faces(image):
    """Simulate face detection (returns bounding box)."""
    return [50, 50, 200, 200]  # Fake coordinates (x, y, w, h)

def detect_landmarks(image):
    """Simulate landmark detection."""
    return [(60, 60), (80, 60), (70, 80)]  # Fake landmark points

def extract_features(image):
    """Simulate deep feature extraction (e.g., ViT, FaceNet)."""
    return np.random.rand(512).astype(np.float32)  # Fake 512D embedding

# Function to store image data in database
def store_embedding(image_name, image):
    face_box = detect_faces(image)
    landmarks = detect_landmarks(image)
    features = extract_features(image)

    # Convert to storable formats
    face_box_str = ",".join(map(str, face_box))
    landmarks_str = ";".join([f"{x},{y}" for x, y in landmarks])
    feature_blob = features.tobytes()

    # Store in MySQL database
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO Embeddings (image_name, face_box, landmarks, features) VALUES (%s, %s, %s, %s)",
                   (image_name, face_box_str, landmarks_str, feature_blob))
    conn.commit()
    cursor.close()
    conn.close()

# Function to retrieve stored embeddings
def fetch_stored_embeddings():
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute("SELECT image_name, face_box, landmarks, features FROM Embeddings")
    results = cursor.fetchall()
    cursor.close()
    conn.close()

    stored_data = []
    for row in results:
        name = row[0]
        face_box = list(map(int, row[1].split(","))) if row[1] else []
        landmarks = [tuple(map(int, point.split(","))) for point in row[2].split(";")] if row[2] else []
        feature_vector = np.frombuffer(row[3], dtype=np.float32) if row[3] else None

        stored_data.append({
            "image_name": name,
            "face_box": face_box,
            "landmarks": landmarks,
            "features": feature_vector
        })

    return stored_data
