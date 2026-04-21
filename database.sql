CREATE DATABASE FaceRecognitionDB;
USE FaceRecognitionDB;

CREATE TABLE Users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password VARCHAR(100) NOT NULL
);

INSERT INTO Users (username, password) VALUES ('admin', 'admin123');

CREATE TABLE Faces (
    id INT AUTO_INCREMENT PRIMARY KEY,
    image_name VARCHAR(255) NOT NULL,
    vit_embeddings BLOB NOT NULL
);
