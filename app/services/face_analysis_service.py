import os
import sys
import logging
import dlib
import cv2
import io
from PIL import Image
from ..repositories import face_image_repository
from ..entity.face_image import *
import numpy as np
from io import BytesIO

logger = logging.getLogger(__name__)

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
    try:
        # BytesIO 객체에서 이미지 로드
        file_bytes = np.frombuffer(file_stream.read(), np.uint8)
        
        # PIL을 사용하여 이미지 검증
        try:
            pil_image = Image.open(BytesIO(file_bytes))
            # 이미지를 RGB로 변환
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            # PIL 이미지를 numpy 배열로 변환
            rgb_image = np.array(pil_image)
        except Exception as e:
            logger.error(f"Error loading image with PIL: {str(e)}")
            raise ValueError(f"Invalid image format: {str(e)}")

        # RGB에서 그레이스케일로 변환
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

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
    except Exception as e:
        logger.error(f"Error in get_face_landmarks: {str(e)}")
        raise ValueError(f"Error processing image: {str(e)}")