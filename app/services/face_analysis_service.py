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
        logger.debug(f"Loaded file bytes: {len(file_bytes)} bytes")
        
        # PIL을 사용하여 이미지 검증
        try:
            pil_image = Image.open(BytesIO(file_bytes))
            logger.debug(f"PIL Image format: {pil_image.format}, mode: {pil_image.mode}, size: {pil_image.size}")
            
            # 이미지 크기 조정 (너무 큰 이미지 처리)
            max_size = 800
            if max(pil_image.size) > max_size:
                ratio = max_size / max(pil_image.size)
                new_size = tuple(int(dim * ratio) for dim in pil_image.size)
                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
                logger.debug(f"Resized image to: {new_size}")
            
            # 이미지를 RGB로 변환
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
                logger.debug(f"Converted image to RGB mode")
            
            # PIL 이미지를 numpy 배열로 변환
            rgb_image = np.array(pil_image)
            logger.debug(f"RGB image shape: {rgb_image.shape}, dtype: {rgb_image.dtype}, min: {rgb_image.min()}, max: {rgb_image.max()}")
            
            # 이미지가 올바른 범위에 있는지 확인
            if rgb_image.max() > 255 or rgb_image.min() < 0:
                rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)
                logger.debug("Clipped image values to valid range")
            
        except Exception as e:
            logger.error(f"Error loading image with PIL: {str(e)}")
            raise ValueError(f"Invalid image format: {str(e)}")

        # RGB에서 그레이스케일로 변환
        try:
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            logger.debug(f"Grayscale image shape: {gray.shape}, dtype: {gray.dtype}, min: {gray.min()}, max: {gray.max()}")
        except Exception as e:
            logger.error(f"Error converting to grayscale: {str(e)}")
            raise ValueError(f"Error converting image to grayscale: {str(e)}")

        # 이미지가 8비트인지 확인
        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)
            logger.debug("Converted image to uint8")

        # 얼굴 탐지
        try:
            # dlib 호환성을 위해 이미지 복사
            gray_dlib = gray.copy()
            faces = detector(gray_dlib)
            logger.debug(f"Number of faces detected: {len(faces)}")
        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}")
            raise ValueError(f"Error detecting faces: {str(e)}")
        
        landmarks = []
        for face in faces:
            try:
                shape = predictor(gray_dlib, face)
                landmarks.append([(point.x, point.y) for point in shape.parts()])
            except Exception as e:
                logger.error(f"Error getting landmarks for face: {str(e)}")
                continue

        if not landmarks:
            raise ValueError("No faces detected")

        landmarks_np = np.array(landmarks).flatten().reshape(1, -1)
        return landmarks_np
    except Exception as e:
        logger.error(f"Error in get_face_landmarks: {str(e)}")
        raise ValueError(f"Error processing image: {str(e)}")