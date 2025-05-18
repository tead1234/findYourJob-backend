import asyncio
import os
import logging
from app.entity.face_image import face
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from app.repositories.face_image_repository import face_image_repository
from app.services.face_analysis_service import get_face_landmarks
import joblib
import pandas as pd
import json
import uuid
from io import BytesIO

logger = logging.getLogger(__name__)

class machine_learning_service:
    def __init__(self):
        self.gender_encoder = LabelEncoder()
        self.job_encoder = LabelEncoder()
        self.model_gender = RandomForestClassifier()
        self.model_job1 = RandomForestClassifier()
        self.model_job2 = RandomForestClassifier()
        self.model_job3 = RandomForestClassifier()
        self.translation_df = pd.read_csv("app/job_100_1.csv", encoding="utf-8")
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        
        # 모델 파일이 없으면 초기 모델 생성
        if not self._check_model_files():
            self._create_initial_models()

    def _check_model_files(self):
        """모델 파일들이 존재하는지 확인"""
        required_files = [
            'gender_model.pkl',
            'job1_model.pkl',
            'job2_model.pkl',
            'job3_model.pkl',
            'gender_encoder.pkl',
            'job_encoder.pkl'
        ]
        return all(os.path.exists(os.path.join(self.model_dir, f)) for f in required_files)

    def _create_initial_models(self):
        """초기 모델 생성"""
        try:
            logger.info("Creating initial models...")
            # init_landmark_data.json에서 초기 데이터 로드
            init_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'init_landmark_data.json')
            with open(init_data_path, 'r') as f:
                init_data = json.load(f)

            # 데이터 준비
            landmarks = []
            genders = []
            jobs1 = []
            jobs2 = []
            jobs3 = []

            for face_data in init_data.values():
                landmarks.append(np.array(face_data['landmarks']).flatten())
                genders.append(face_data['gender'])
                jobs1.append(face_data['job1'])
                jobs2.append(face_data['job2'])
                jobs3.append(face_data['job3'])

            X = np.array(landmarks)
            y_gender = self.gender_encoder.fit_transform(genders)
            y_job1 = self.job_encoder.fit_transform(jobs1)
            y_job2 = self.job_encoder.fit_transform(jobs2)
            y_job3 = self.job_encoder.fit_transform(jobs3)

            # 모델 학습
            self.model_gender.fit(X, y_gender)
            self.model_job1.fit(X, y_job1)
            self.model_job2.fit(X, y_job2)
            self.model_job3.fit(X, y_job3)

            # 모델 저장
            joblib.dump(self.model_gender, os.path.join(self.model_dir, 'gender_model.pkl'))
            joblib.dump(self.model_job1, os.path.join(self.model_dir, 'job1_model.pkl'))
            joblib.dump(self.model_job2, os.path.join(self.model_dir, 'job2_model.pkl'))
            joblib.dump(self.model_job3, os.path.join(self.model_dir, 'job3_model.pkl'))
            joblib.dump(self.gender_encoder, os.path.join(self.model_dir, 'gender_encoder.pkl'))
            joblib.dump(self.job_encoder, os.path.join(self.model_dir, 'job_encoder.pkl'))
            
            logger.info("Initial models created successfully")
        except Exception as e:
            logger.error(f"Error creating initial models: {str(e)}")
            raise

    def train_models(self, faces: list[face]):
        if not faces:
            raise ValueError("No face data available for training")

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
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        y_gender = self.gender_encoder.fit_transform(genders)
        y_job1 = self.job_encoder.fit_transform(jobs1)
        y_job2 = self.job_encoder.fit_transform(jobs2)
        y_job3 = self.job_encoder.fit_transform(jobs3)

        logger.info(f"Training data shape: X={X.shape}, y_gender={y_gender.shape}")
        
        self.model_gender.fit(X, y_gender)
        self.model_job1.fit(X, y_job1)
        self.model_job2.fit(X, y_job2)
        self.model_job3.fit(X, y_job3)

        joblib.dump(self.model_gender, os.path.join(self.model_dir, 'gender_model.pkl'))
        joblib.dump(self.model_job1, os.path.join(self.model_dir, 'job1_model.pkl'))
        joblib.dump(self.model_job2, os.path.join(self.model_dir, 'job2_model.pkl'))
        joblib.dump(self.model_job3, os.path.join(self.model_dir, 'job3_model.pkl'))
        joblib.dump(self.gender_encoder, os.path.join(self.model_dir, 'gender_encoder.pkl'))
        joblib.dump(self.job_encoder, os.path.join(self.model_dir, 'job_encoder.pkl'))

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

        if not faces:
            logger.warning("No face data found in database, using face_samples data")
            # face_samples 데이터 로드
            face_sample_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'face_samples')
            csv_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'face_specific_data.csv')
            
            try:
                face_data_df = pd.read_csv(csv_file, header=None, names=["filename", "gender", "job1", "job2", "job3"])
                
                for image_file in os.listdir(face_sample_dir):
                    if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        try:
                            # 이미지 파일명에 해당하는 데이터 찾기
                            filename = os.path.splitext(image_file)[0]
                            face_info = face_data_df[face_data_df['filename'] == filename]
                            
                            if not face_info.empty:
                                # 이미지에서 랜드마크 추출
                                image_path = os.path.join(face_sample_dir, image_file)
                                with open(image_path, 'rb') as f:
                                    file_stream = BytesIO(f.read())
                                    landmarks = get_face_landmarks(file_stream)
                                
                                if landmarks is not None and len(landmarks) > 0:
                                    faces.append(face(
                                        id=str(uuid.uuid4()),
                                        gender=face_info['gender'].values[0],
                                        landmarks=landmarks.flatten().tolist(),
                                        job1=face_info['job1'].values[0],
                                        job2=face_info['job2'].values[0],
                                        job3=face_info['job3'].values[0]
                                    ))
                        except Exception as e:
                            logger.error(f"Error processing {image_file}: {str(e)}")
                            continue
                
                if not faces:
                    logger.error("No valid face data found in face_samples")
                    raise ValueError("No valid face data found in face_samples")
                
                logger.info(f"Loaded {len(faces)} face images from face_samples")
            except Exception as e:
                logger.error(f"Error loading face_samples data: {str(e)}")
                raise

        logger.info(f"Training with {len(faces)} face images")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.train_models, faces)

    def received_face_learn(self, received_face: face):
        pass

    def translate_job(self, job_name):
        if job_name in self.translation_df['English'].values:
            return self.translation_df.loc[self.translation_df['English'] == job_name, 'Korean'].values[0]
        else:
            return job_name

    def predict(self, landmarks: np.ndarray) -> dict:
        self.model_gender = joblib.load(os.path.join(self.model_dir, 'gender_model.pkl'))
        self.model_job1 = joblib.load(os.path.join(self.model_dir, 'job1_model.pkl'))
        self.model_job2 = joblib.load(os.path.join(self.model_dir, 'job2_model.pkl'))
        self.model_job3 = joblib.load(os.path.join(self.model_dir, 'job3_model.pkl'))
        self.gender_encoder = joblib.load(os.path.join(self.model_dir, 'gender_encoder.pkl'))
        self.job_encoder = joblib.load(os.path.join(self.model_dir, 'job_encoder.pkl'))

        # 성별 예측
        gender_pred = self.model_gender.predict(landmarks)
        gender = self.gender_encoder.inverse_transform(gender_pred)[0].strip()

        # 각 모델의 예측 확률 가져오기
        job1_probs = self.model_job1.predict_proba(landmarks)[0]
        job2_probs = self.model_job2.predict_proba(landmarks)[0]
        job3_probs = self.model_job3.predict_proba(landmarks)[0]

        # 모든 가능한 직업 목록
        all_jobs = self.job_encoder.classes_
        
        # 각 직업의 평균 확률 계산
        job_probabilities = {}
        for i, job in enumerate(all_jobs):
            avg_prob = (job1_probs[i] + job2_probs[i] + job3_probs[i]) / 3
            job_probabilities[job] = avg_prob

        # 확률이 높은 순서대로 정렬
        sorted_jobs = sorted(job_probabilities.items(), key=lambda x: x[1], reverse=True)
        
        # 상위 3개 직업 선택
        selected_jobs = [job.strip() for job, _ in sorted_jobs[:3]]
        
        job1 = selected_jobs[0]
        job2 = selected_jobs[1]
        job3 = selected_jobs[2]

        job1_korean = self.translate_job(job1)
        job2_korean = self.translate_job(job2)
        job3_korean = self.translate_job(job3)

        return {
            "predicted_gender": gender,
            "predicted_job1": job1_korean,
            "predicted_job2": job2_korean,
            "predicted_job3": job3_korean,
            "predicted_job1_translated": job1,
            "predicted_job2_translated": job2,
            "predicted_job3_translated": job3
        }

