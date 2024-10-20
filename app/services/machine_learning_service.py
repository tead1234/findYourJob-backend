import asyncio
import os

from app.entity.face_image import face
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from app.repositories.face_image_repository import face_image_repository
from app.services.face_analysis_service import get_face_landmarks
import joblib
import pandas as pd


class machine_learning_service:
    def __init__(self):
        self.gender_encoder = LabelEncoder()
        self.job_encoder = LabelEncoder()
        self.model_gender = RandomForestClassifier()
        self.model_job1 = RandomForestClassifier()
        self.model_job2 = RandomForestClassifier()
        self.model_job3 = RandomForestClassifier()
        self.translation_df = pd.read_csv("app/job_100_1.csv", encoding="utf-8")

    def train_models(self, faces: list[face]):
        landmarks = []
        genders = []
        jobs1 = []
        jobs2 = []
        jobs3 = []

        for f in faces:
            landmarks.append(np.array(f.landmarks).flatten())
            genders.append(f.gender)
            jobs1.append(f.job1)
            jobs2.append(f.job2)
            jobs3.append(f.job3)

        X = np.array(landmarks)
        y_gender = self.gender_encoder.fit_transform(genders)
        y_job1 = self.job_encoder.fit_transform(jobs1)
        y_job2 = self.job_encoder.fit_transform(jobs2)
        y_job3 = self.job_encoder.fit_transform(jobs3)

        self.model_gender.fit(X, y_gender)
        self.model_job1.fit(X, y_job1)
        self.model_job2.fit(X, y_job2)
        self.model_job3.fit(X, y_job3)

        joblib.dump(self.model_gender, 'gender_model.pkl')
        joblib.dump(self.model_job1, 'job1_model.pkl')
        joblib.dump(self.model_job2, 'job2_model.pkl')
        joblib.dump(self.model_job3, 'job3_model.pkl')
        joblib.dump(self.gender_encoder, 'gender_encoder.pkl')
        joblib.dump(self.job_encoder, 'job_encoder.pkl')

    async def self_learn(self):

        faces = []
        print("학습 실시")
        cursor = face_image_repository().collection.find()
        async for document in cursor:
            faces.append(face(
                id=str(document['_id']),
                gender=document['gender'],
                landmarks=document['landmarks'],
                job1=document['job1'],
                job2=document['job2'],
                job3=document['job3'],
            ))

        self.train_models(faces)

    def received_face_learn(self, received_face: face):
        pass

    def translate_job(self, job_name):
        if job_name in self.translation_df['English'].values:
            return self.translation_df.loc[self.translation_df['English'] == job_name, 'Korean'].values[0]
        else:
            return job_name

    def predict(self, landmarks: np.ndarray) -> dict:
        self.model_gender = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gender_model.pkl'))
        self.model_job1 = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'job1_model.pkl'))
        self.model_job2 = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'job2_model.pkl'))
        self.model_job3 = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'job3_model.pkl'))
        self.gender_encoder = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gender_encoder.pkl'))
        self.job_encoder = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'job_encoder.pkl'))

        # 성별 예측
        gender_pred = self.model_gender.predict(landmarks)
        gender = self.gender_encoder.inverse_transform(gender_pred)[0].strip()

        # 직업 예측
        job1_pred = self.model_job1.predict(landmarks)
        job2_pred = self.model_job2.predict(landmarks)
        job3_pred = self.model_job3.predict(landmarks)

        job1 = self.job_encoder.inverse_transform(job1_pred)[0].strip()
        job2 = self.job_encoder.inverse_transform(job2_pred)[0].strip()
        job3 = self.job_encoder.inverse_transform(job3_pred)[0].strip()

        job1_korean = self.translate_job(job1)
        job2_korean = self.translate_job(job2)
        job3_korean = self.translate_job(job3)

        return {
            "predicted_gender": gender,
            "predicted_job1": job1,
            "predicted_job2": job2,
            "predicted_job3": job3,
            "predicted_job1_translated": job1_korean,
            "predicted_job2_translated": job2_korean,
            "predicted_job3_translated": job3_korean
        }


# predeict_model =  machine_learning_service().predict("../face_samples/example.png")
# # predeict_model
#
# print(predeict_model)
# async def main():
#
#     await machine_learning_service().self_learn()
# # 메인 이벤트 루프에서 실행
# if __name__ == "__main__":
#     asyncio.run(main())