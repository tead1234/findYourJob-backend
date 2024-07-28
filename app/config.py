import os

MODEL_FILE_PATH = os.getenv("MODEL_FILE_PATH", "shape_predictor_68_face_landmarks.dat")
DB_URL = os.getenv("DB_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "mydatabase")