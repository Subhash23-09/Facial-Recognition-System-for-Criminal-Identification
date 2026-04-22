#  dfd70ddd897c43b9fa54e7798933807d35c1029e9f766d797d58039dc704dc84

from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import torch
import numpy as np
import mysql.connector
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms
from transformers import ViTImageProcessor, ViTModel
from scipy.spatial.distance import cosine
from werkzeug.security import check_password_hash
from werkzeug.utils import secure_filename
from database import db_config
from align_enhance import align_and_enhance_faces  # ✅ Using your external enhancement logic
from datetime import datetime

app = Flask(__name__)
app.secret_key = "dfd70ddd897c43b9fa54e7798933807d35c1029e9f766d797d58039dc704dc84ey"

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Models
mtcnn = MTCNN(keep_all=True, device=device)
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)
vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# === DB connection ===
def get_db_connection():
    return mysql.connector.connect(**db_config)

def insert_log(action_name, details=None):
    timestamp = datetime.now()  # Get current timestamp using Python
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO logs (action_name, details, timestamp) VALUES (%s, %s, %s)",
        (action_name, details, timestamp)
    )
    conn.commit()
    cursor.close()
    conn.close()

# === ViT Embedding ===
def get_vit_embedding(face_pil):
    inputs = vit_processor(face_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = vit_model(**inputs).last_hidden_state[:, 0, :]
    return outputs.cpu().numpy().flatten()



# === Matching ===
def match_criminal_mysql(new_embedding):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT image_name, features, image_data FROM embeddings1")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    best_match = None
    best_similarity = 0.0
    best_match_image_data = None

    for row in rows:
        stored_name, stored_feature_blob, stored_image_data = row
        stored_feature = np.frombuffer(stored_feature_blob, dtype=np.float32)

        similarity = 1 - cosine(stored_feature, new_embedding)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = stored_name
            best_match_image_data = stored_image_data

    if best_match:
        matched_path = f"static/uploads/matched_{best_match}.jpg"
        with open(matched_path, "wb") as f:
            f.write(best_match_image_data)
        return {
            "matched_image": matched_path,
            "similarity": round(best_similarity * 100, 2),
            "matched_name": best_match
        }
    return None

# === Routes ===

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Database connection
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM Users WHERE username=%s", (username,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user and check_password_hash(user[0], password):
            session['username'] = username
            insert_log('login', f'User {username} logged in successfully')  # Log login event
            return redirect(url_for('dashboard'))

        flash("Invalid credentials", "danger")
    return render_template('login.html')

@app.route('/logout')
def logout():
    if 'username' in session:
        username = session['username']
        session.pop('username', None)  # Log out the user
        insert_log('logout', f'User {username} logged out')  # Log logout event
        flash("Logged out successfully.", "info")
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    # Assuming 'username' is stored in session upon login
    if 'username' not in session:
        return redirect(url_for('login'))  # Redirect to login if not authenticated
    return render_template('dashboard.html', username=session['username'])


@app.route('/missing_person', methods=['GET', 'POST'])
def missing_person():
    if 'username' not in session:
        return redirect(url_for('login'))

    # Just redirect to the /upload route
    return redirect(url_for('upload'))




@app.route('/add_criminal', methods=['GET', 'POST'])
def add_criminal():
    if request.method == 'POST':
        # Get the form data
        name = request.form['name']
        description = request.form['description']
        file = request.files['image']
        
        if not name or not description or not file:
            flash("Please provide all the details (name, description, and image).", "warning")
            return redirect(url_for('add_criminal'))
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Read image as binary to store in database
        with open(filepath, 'rb') as f:
            image_data = f.read()

        # Connect to the database and insert the data
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO criminals (name, description, image_name, image_data) VALUES (%s, %s, %s, %s)",
            (name, description, filename, image_data)
        )
        conn.commit()
        cursor.close()
        conn.close()

        flash("Criminal added successfully!", "success")
        return redirect(url_for('dashboard'))

    return render_template('add_criminal.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files['file']
        if not file:
            flash("No file uploaded", "warning")
            return redirect(url_for('upload'))

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        image = Image.open(filepath).convert("RGB")
        boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)
        if boxes is None:
            flash("No face detected", "warning")
            return redirect(url_for('upload'))

        enhanced_faces = align_and_enhance_faces(image, boxes, landmarks, filepath)

        best_match_data = None
        enhanced_filename = None  # Initialize the variable outside the loop

        for i, (enhanced_face_pil, (x1, y1, x2, y2)) in enumerate(enhanced_faces):
            enhanced_filename = f"enhanced_{i}_{filename}"
            enhanced_path = os.path.join(app.config['UPLOAD_FOLDER'], enhanced_filename)
            enhanced_face_pil.save(enhanced_path)

            embedding = get_vit_embedding(enhanced_face_pil)
            best_match_data = match_criminal_mysql(embedding)

        # Check if best_match_data exists before proceeding
        if not best_match_data:
            flash("No match found", "warning")
            return redirect(url_for('upload'))

        return redirect(url_for('result',
                                original=filename,
                                enhanced=enhanced_filename,
                                matched=os.path.basename(best_match_data["matched_image"]) if best_match_data else '',
                                percentage=best_match_data["similarity"] if best_match_data else 0,
                                name=best_match_data["matched_name"] if best_match_data else "Unknown"))
                                


    return render_template('index.html')


@app.route('/result')
def result():
    original = request.args.get('original')
    enhanced = request.args.get('enhanced')
    matched = request.args.get('matched')
    percentage = request.args.get('percentage')
    matched_name = request.args.get('name')

    return render_template('result.html',
                           original_url=url_for('static', filename=f"uploads/{original}"),
                           enhanced_url=url_for('static', filename=f"uploads/{enhanced}"),
                           matched_url=url_for('static', filename=f"uploads/{matched}") if matched else None,
                           match_percentage=percentage,
                           matched_name=matched_name)

if __name__ == '__main__':
    app.run(debug=True)







