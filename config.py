import os

db_config = {
    "host": "localhost",
    "user": "root",
    "password": "admin23",
    "database": "FaceRecognitionDB",
    "port": 3306
}

UPLOAD_FOLDER = os.path.join(os.getcwd(), "static/uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
