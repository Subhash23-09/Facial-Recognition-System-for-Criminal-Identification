import cv2
import torch
import numpy as np
from PIL import Image, ImageEnhance
from facenet_pytorch import MTCNN

mtcnn = MTCNN(keep_all=True)

def align_and_enhance_faces(image, boxes, landmarks, image_path):
    """Aligns, enhances and returns cropped face images."""
    aligned_faces = []
    image_cv = cv2.imread(image_path)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)

        # Crop the detected face
        face = image_cv[y1:y2, x1:x2]
        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

        # Enhance brightness and contrast
        enhancer = ImageEnhance.Contrast(face_pil)
        face_pil = enhancer.enhance(1.5)  # Increase contrast
        enhancer = ImageEnhance.Brightness(face_pil)
        face_pil = enhancer.enhance(1.2)  # Increase brightness

        aligned_faces.append((face_pil, (x1, y1, x2, y2)))

    return aligned_faces
