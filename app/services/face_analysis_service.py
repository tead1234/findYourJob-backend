import dlib
import cv2

MODEL_FILE_PATH = 'D:\\silde_project_workspace\\findYourJob-backend\\app\\shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(MODEL_FILE_PATH)


class face_analyze_service():

    def get_face_landmarks(self, image_path):
        load_image_path = cv2.imread(image_path)
        gray = cv2.cvtColor(load_image_path, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        landmarks = []
        for face in faces:
            shape = predictor(gray, face)
            landmarks.append([(point.x, point.y) for point in shape.parts()])
        print(landmarks)
        return landmarks


face_analyze_service().get_face_landmarks("D:\\silde_project_workspace\\findYourJob-backend\\face_samples\\17111302_S001_L03_E01_C7.jpg")
