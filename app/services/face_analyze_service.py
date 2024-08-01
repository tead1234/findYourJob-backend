import dlib
import cv2

MODEL_FILE_PATH = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(MODEL_FILE_PATH)


class face_analyze_service():

    def get_face_landmarks(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        landmarks = []
        for face in faces:
            shape = predictor(gray, face)
            landmarks.append([(point.x, point.y) for point in shape.parts()])
        return landmarks


face_analyze_service().get_face_landmarks()
