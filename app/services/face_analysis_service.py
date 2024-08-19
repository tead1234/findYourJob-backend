import os
import sys

import dlib
import cv2
import io

from ..repositories import face_image_repository
from ..entity.face_image import *
import numpy as np
from io import BytesIO

MODEL_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(MODEL_FILE_PATH)


def process(gender, received_face_path):
    pass


def save_face_landmarks_db(file, gender, job1, job2, job3):
    new_face = face(
        gender=gender,
        job1=job1,
        job2=job2,
        job3=job3
    )

    file_id = face_image_repository.save_face_image(new_face)
    return file_id


# def get_face_landmarks(file_path):
#     # Read the image file
#     with open(file_path, 'rb') as f:
#         file_bytes = np.frombuffer(f.read(), np.uint8)
#
#     # Decode the image
#     load_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#
#     # Convert to grayscale
#     gray = cv2.cvtColor(load_image, cv2.COLOR_BGR2GRAY)
#
#     # Detect faces
#     faces = detector(gray)
#     landmarks = []
#     for face_image in faces:
#         shape = predictor(gray, face_image)
#         landmarks.append([(point.x, point.y) for point in shape.parts()])
#     return landmarks

def get_face_landmarks(file_stream: BytesIO) -> np.ndarray:
    # BytesIO 객체에서 이미지 로드
    file_bytes = np.frombuffer(file_stream.read(), np.uint8)
    load_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if load_image is None:
        raise ValueError("Unable to decode image from file bytes")

    # 그레이스케일로 변환
    gray = cv2.cvtColor(load_image, cv2.COLOR_BGR2GRAY)

    # 얼굴 탐지
    faces = detector(gray)
    landmarks = []

    for face in faces:
        shape = predictor(gray, face)
        landmarks.append([(point.x, point.y) for point in shape.parts()])

    if not landmarks:
        raise ValueError("No faces detected")

    landmarks_np = np.array(landmarks).flatten().reshape(1, -1)
    return landmarks_np