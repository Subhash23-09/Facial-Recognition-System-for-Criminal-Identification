import mysql.connector

# Database Configuration
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "admin23",
    "database": "FaceRecognitionDB"
}

# Function to initialize database
def initialize_database():
    conn = mysql.connector.connect(host="localhost", user="root", password="admin23")
    cursor = conn.cursor()

    # Create Database
    cursor.execute("CREATE DATABASE IF NOT EXISTS FaceRecognitionDB")
    cursor.execute("USE FaceRecognitionDB")

    # Create Users Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(50) UNIQUE NOT NULL,
        password VARCHAR(255) NOT NULL
    )""")

    # Create Embeddings Table with fields from the image
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Embeddings1 (
        id INT AUTO_INCREMENT PRIMARY KEY,
        image_name VARCHAR(255),
        face_box TEXT,
        landmarks TEXT,
        features BLOB
    )""")

    conn.commit()
    cursor.close()
    conn.close()

# Initialize Database
initialize_database()
