import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from pathlib import Path
import json
from io import BytesIO
import pandas as pd
from app.services.face_analysis_service import get_face_landmarks

class database_initialize_util:
   def __init__(self):
       pass

   @staticmethod
   def init_database():
       base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
       face_sample_dir = os.path.join(base_dir, 'face_samples')
       face_sample_dir = Path(face_sample_dir)
       
       output_dir = Path('app')
       output_dir.mkdir(exist_ok=True)
       
       # CSV 파일 경로를 face_samples 디렉토리 하위로 수정
       csv_file = os.path.join(face_sample_dir, 'face_specific_data.csv')
       face_data_df = pd.read_csv(csv_file, header=None, names=["filename", "gender", "job1", "job2", "job3"])

       all_face_data = {}

       for image_file in face_sample_dir.iterdir():
           if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
               landmarks = get_face_landmarks(str(image_file))
               if landmarks:
                   face_data = {
                       "landmarks": landmarks
                   }

                   filename = image_file.stem
                   additional_info = face_data_df[face_data_df['filename'] == filename]

                   if not additional_info.empty:
                       face_data["gender"] = additional_info["gender"].values[0]
                       face_data["job1"] = additional_info["job1"].values[0]
                       face_data["job2"] = additional_info["job2"].values[0]
                       face_data["job3"] = additional_info["job3"].values[0]

                   all_face_data[image_file.stem] = face_data
               else:
                   print("이미지가 너무 작습니다.")
                   sys.exit(1)

       output_file = output_dir / "init_landmark_data.json"
       with open(output_file, 'w') as f:
           json.dump(all_face_data, f, indent=4)

database_initialize_util.init_database()
