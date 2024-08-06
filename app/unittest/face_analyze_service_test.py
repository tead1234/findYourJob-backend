import unittest
from app.services.face_analysis_service import *


class face_analysis_test(unittest.TestCase):
    def landmark_test(self):
        result = get_face_landmarks("/findYourJob-backend/app/face_samples\\17111302_S001_L03_E01_C7.jpg")
